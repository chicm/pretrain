"""Unit test for EpochMixtureDataset (OLMo-style without-replacement mixture).

Builds tiny synthetic sources on disk (encoding each instance's start offset in
its tokens so we can detect duplicates), then checks:
  1. within a single pass each source yields its instances WITHOUT replacement
     (100% coverage, no repeats) for a single replica;
  2. the training-visible source mixture converges to the target weights;
  3. across DDP ranks, instance ownership is disjoint (no instance drawn by two
     ranks in the same pass).
"""
import os, sys, json, tempfile, shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import EpochMixtureDataset  # noqa: E402

BS = 8                    # block_size
INST = BS                 # stride
DT = "uint32"


def make_source(root, name, n_instances, tag):
    """Write one shard whose token stream encodes, per instance, a unique id
    (tag*1e6 + instance_index) in its first token, so drawn (x,y) reveals which
    instance produced it. Length = n_instances*BS + 1."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    n_tok = n_instances * BS + 1
    arr = np.zeros(n_tok, dtype=np.dtype(DT))
    for i in range(n_instances):
        arr[i * BS] = tag * 1_000_000 + i   # marker at each instance start
    arr.tofile(os.path.join(d, "shard_0000.bin"))
    with open(os.path.join(d, "index.json"), "w") as f:
        json.dump({"dtype": DT, "eot": 0, "vocab_size": 32,
                   "total_tokens": n_tok,
                   "shards": [{"path": "shard_0000.bin", "tokens": n_tok}]}, f)
    return d


def draw(ds, n):
    it = iter(ds)
    xs = []
    for _ in range(n):
        x, y = next(it)
        xs.append(int(x[0].item()))   # marker token = instance id
    return xs


def test_coverage_no_repeat():
    tmp = tempfile.mkdtemp()
    try:
        # single source, 100 instances, one replica -> a full pass must cover
        # all 100 distinct ids exactly once before any repeat.
        make_source(tmp, "solo", 100, tag=1)
        ds = EpochMixtureDataset({os.path.join(tmp, "solo"): 1.0}, BS,
                                 seed=42, rank=0, world=1)
        ids = draw(ds, 100)
        assert len(set(ids)) == 100, f"expected 100 unique, got {len(set(ids))}"
        # next 100 = second pass, again unique
        it = iter(ds); [next(it) for _ in range(100)]
        second = [int(next(it)[0][0].item()) for _ in range(100)]
        assert len(set(second)) == 100
        print("PASS coverage_no_repeat: 100/100 unique per pass")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_weight_ratio():
    tmp = tempfile.mkdtemp()
    try:
        make_source(tmp, "big", 5000, tag=1)
        make_source(tmp, "small", 5000, tag=2)
        ds = EpochMixtureDataset(
            {os.path.join(tmp, "big"): 0.7,
             os.path.join(tmp, "small"): 0.3}, BS,
            seed=7, rank=0, world=1, count_sources=True)
        it = iter(ds)
        for _ in range(4000):
            next(it)
        c = ds.source_counts()
        frac_big = c.get("big", 0) / 4000
        assert abs(frac_big - 0.7) < 0.03, f"big frac {frac_big:.3f} != 0.70"
        print(f"PASS weight_ratio: big={frac_big:.3f} (target 0.70)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ddp_disjoint():
    tmp = tempfile.mkdtemp()
    try:
        make_source(tmp, "solo", 120, tag=1)
        world = 4
        seen = []
        for r in range(world):
            ds = EpochMixtureDataset({os.path.join(tmp, "solo"): 1.0}, BS,
                                     seed=99, rank=r, world=world)
            # each replica owns 120/4 = 30 instances per pass
            ids = draw(ds, 30)
            assert len(set(ids)) == 30, f"rank {r} repeats within pass"
            seen.append(set(ids))
        # pairwise disjoint
        for i in range(world):
            for j in range(i + 1, world):
                inter = seen[i] & seen[j]
                assert not inter, f"ranks {i},{j} overlap: {sorted(inter)[:5]}"
        union = set().union(*seen)
        assert len(union) == 120, f"union {len(union)} != 120 (full coverage)"
        print("PASS ddp_disjoint: 4 ranks x 30 = 120 unique, no overlap")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_coverage_no_repeat()
    test_weight_ratio()
    test_ddp_disjoint()
    print("ALL TESTS PASSED")
