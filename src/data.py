"""Data pipeline: tokenize a HF dataset into a single packed uint16/uint32 .bin,
then stream fixed-length blocks for causal LM training.

Two steps:
  1) prepare_data(): tokenize -> data/<name>_train.bin (+ val.bin). Run once.
  2) PackedDataset: memory-maps the .bin and yields (x, y) blocks.
"""
import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


# ----------------------------------------------------------------------------
# Sharded tokenize output (for 1T multi-source corpus). Each source is written
# as shard_XXXX.bin files + an index.json describing them, enabling resumable
# tokenize and weighted multi-source sampling at train time.
# ----------------------------------------------------------------------------
def read_index(src_dir):
    """Load a source directory's index.json (or synthesize one from *.bin +
    a legacy single train.bin/meta.json)."""
    idx_path = os.path.join(src_dir, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return json.load(f)
    # legacy fallback: a single train.bin + meta.json
    meta_path = os.path.join(src_dir, "meta.json")
    train_bin = os.path.join(src_dir, "train.bin")
    if os.path.exists(train_bin) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return {"dtype": meta["dtype"], "eot": meta.get("eot"),
                "vocab_size": meta.get("vocab_size"),
                "total_tokens": meta.get("total_tokens"),
                "shards": [{"path": "train.bin",
                            "tokens": meta.get("total_tokens")}]}
    # last resort: glob shard_*.bin, assume uint32
    shards = sorted(glob.glob(os.path.join(src_dir, "shard_*.bin")))
    if not shards:
        raise FileNotFoundError(f"no index.json / train.bin / shard_*.bin in {src_dir}")
    return {"dtype": "uint32", "eot": None, "vocab_size": None,
            "total_tokens": None,
            "shards": [{"path": os.path.basename(s), "tokens": None} for s in shards]}


class WeightedMultiSourceDataset(IterableDataset):
    """Infinite iterable that samples fixed-length (x, y) blocks from multiple
    tokenized sources according to per-source weights.

    sources: dict {source_dir: weight}. Weights are normalized. Each step:
      pick a source ~ weight, pick a shard ~ shard-token-count, read a random
      contiguous block of (block_size+1) tokens from that shard's memmap.

    DDP-aware: each rank uses a distinct RNG stream (seed + rank). Also splits
    across DataLoader workers. Yields torch int64 (x, y) like PackedDataset.
    """
    def __init__(self, sources, block_size, seed=1337, rank=0, world=1,
                 count_sources=False):
        super().__init__()
        self.block_size = block_size
        self.seed = seed
        self.rank = rank
        self.world = world
        self.count_sources = count_sources
        self._counts = {}
        # resolve each source: list of (memmap, n_blocks) + source weight
        self.src_names = []
        self.src_weights = []
        self.src_shards = []   # per source: list of dict(mm, n_tokens)
        self.src_shard_w = []  # per source: normalized shard weights (by tokens)
        for src_dir, w in sources.items():
            if w <= 0:
                continue
            idx = read_index(src_dir)
            dtype = np.dtype(idx["dtype"])
            shards = []
            tok_counts = []
            for sh in idx["shards"]:
                p = os.path.join(src_dir, sh["path"])
                mm = np.memmap(p, dtype=dtype, mode="r")
                if len(mm) <= block_size + 1:
                    continue
                shards.append(mm)
                tok_counts.append(len(mm))
            if not shards:
                continue
            tok_counts = np.array(tok_counts, dtype=np.float64)
            self.src_names.append(os.path.basename(src_dir.rstrip("/")))
            self.src_weights.append(float(w))
            self.src_shards.append(shards)
            self.src_shard_w.append(tok_counts / tok_counts.sum())
        if not self.src_shards:
            raise ValueError("WeightedMultiSourceDataset: no usable sources")
        sw = np.array(self.src_weights, dtype=np.float64)
        self.src_weights = sw / sw.sum()

    def source_counts(self):
        return dict(self._counts)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        # unique stream per (rank, worker)
        rng = np.random.default_rng(
            self.seed + self.rank * 100003 + worker_id * 10007)
        bs1 = self.block_size + 1
        n_src = len(self.src_shards)
        while True:
            si = rng.choice(n_src, p=self.src_weights)
            shards = self.src_shards[si]
            shi = rng.choice(len(shards), p=self.src_shard_w[si])
            mm = shards[shi]
            start = int(rng.integers(0, len(mm) - bs1))
            chunk = np.asarray(mm[start:start + bs1]).astype(np.int64)
            if self.count_sources:
                name = self.src_names[si]
                self._counts[name] = self._counts.get(name, 0) + 1
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
            yield x, y


def prepare_data(dataset_name, out_dir, tokenizer, split="train",

                 text_key="text", num_proc=32, val_frac=0.0005,
                 hf_config=None, streaming=False, max_docs=None, eot_id=None):
    """Tokenize a HF dataset into packed .bin files.
    vocab < 65536 -> uint16, else uint32 (Qwen3 vocab 151669 -> uint32)."""
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    if eot_id is None:
        eot_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    eot = eot_id
    dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32

    ds = load_dataset(dataset_name, name=hf_config, split=split, streaming=streaming)

    def tok(example):
        ids = tokenizer(example[text_key])["input_ids"]
        ids.append(eot)
        return {"ids": ids, "len": len(ids)}

    if streaming:
        # simple single-process path for streaming
        all_ids = []
        for i, ex in enumerate(ds):
            all_ids.extend(tokenizer(ex[text_key])["input_ids"] + [eot])
            if max_docs and i + 1 >= max_docs:
                break
        arr = np.array(all_ids, dtype=dtype)
        n_val = int(len(arr) * val_frac)
        if n_val > 0:
            arr[:-n_val].tofile(os.path.join(out_dir, "train.bin"))
            arr[-n_val:].tofile(os.path.join(out_dir, "val.bin"))
        else:
            arr.tofile(os.path.join(out_dir, "train.bin"))
        total_tokens = int(len(arr))
    else:
        ds = ds.map(tok, remove_columns=ds.column_names, num_proc=num_proc,
                    desc="tokenizing")
        total = int(np.sum(ds["len"], dtype=np.int64))
        n_val = int(total * val_frac)
        n_train = total - n_val

        # Write directly to disk-backed memmaps, filling in large batched
        # concatenations (vectorized) instead of a per-doc Python loop.
        train_path = os.path.join(out_dir, "train.bin")
        val_path = os.path.join(out_dir, "val.bin") if n_val > 0 else None
        train_mm = np.memmap(train_path, dtype=dtype, mode="w+", shape=(n_train,))
        val_mm = (np.memmap(val_path, dtype=dtype, mode="w+", shape=(n_val,))
                  if val_path else None)

        ds = ds.with_format("numpy")
        write_batch = 8192  # docs per flush; concat then bulk-assign
        tidx = vidx = 0
        buf, buf_len = [], 0
        FLUSH = 1 << 24  # ~16M tokens per flush chunk

        def flush(chunk):
            nonlocal tidx, vidx
            # route into train / val by absolute position (val is the tail)
            done = tidx + vidx
            for seg_dst, seg in _split_train_val(chunk, done, n_train):
                if seg_dst == "train":
                    train_mm[tidx: tidx + len(seg)] = seg
                    tidx += len(seg)
                else:
                    val_mm[vidx: vidx + len(seg)] = seg
                    vidx += len(seg)

        for batch in ds.iter(batch_size=write_batch):
            # batch["ids"] is a list of 1-D numpy arrays -> one C-level concat
            buf.append(np.concatenate(list(batch["ids"])))
            buf_len += len(buf[-1])
            if buf_len >= FLUSH:
                flush(np.concatenate(buf))
                buf, buf_len = [], 0
        if buf:
            flush(np.concatenate(buf))

        train_mm.flush()
        if val_mm is not None:
            val_mm.flush()
        del train_mm, val_mm
        total_tokens = total
    meta = {"dtype": np.dtype(dtype).name, "vocab_size": tokenizer.vocab_size,
            "eot": eot, "total_tokens": int(total_tokens)}
    import json
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare_data] wrote {total_tokens:,} tokens to {out_dir} (dtype={dtype})")
    return meta


def prepare_data_sharded(dataset_name, out_dir, tokenizer, split="train",
                         text_key="text", hf_config=None,
                         eot_id=None, shard_tokens=1_000_000_000,
                         target_tokens=None, num_proc=32, data_files=None,
                         streaming=True):
    """Streaming, resumable, sharded tokenizer for very large sources (1T corpus).

    Writes shard_XXXX.bin (uint16/uint32) of ~shard_tokens each + index.json.
    Resumable: on restart, completed shards in index.json are kept and we skip
    ahead by their token count (approximate doc-level skip via HF streaming).
    Stops when target_tokens reached (or dataset exhausted).
    """
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    if eot_id is None:
        eot_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32
    idx_path = os.path.join(out_dir, "index.json")

    # load / init index for resume
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
    else:
        index = {"dtype": np.dtype(dtype).name, "eot": int(eot_id),
                 "vocab_size": int(tokenizer.vocab_size),
                 "dataset": dataset_name, "hf_config": hf_config,
                 "total_tokens": 0, "shards": []}
    done_tokens = index.get("total_tokens", 0)
    shard_id = len(index["shards"])
    print(f"[shard] resume: {shard_id} shards, {done_tokens:,} tokens done; "
          f"target={target_tokens}")
    if target_tokens and done_tokens >= target_tokens:
        print("[shard] target already reached; nothing to do.")
        return index

    load_kwargs = dict(name=hf_config, split=split, streaming=streaming)
    if data_files is not None:
        load_kwargs["data_files"] = data_files
    ds = load_dataset(dataset_name, **load_kwargs)

    def save_index():
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)

    # skip already-consumed docs approximately by token count
    skip_tokens = done_tokens
    buf = []
    buf_len = 0
    total = done_tokens
    skipped = 0
    for ex in ds:
        txt = ex.get(text_key)
        if not txt:
            continue
        ids = tokenizer(txt)["input_ids"]
        ids.append(eot_id)
        if skipped < skip_tokens:
            skipped += len(ids)
            continue
        buf.append(np.asarray(ids, dtype=dtype))
        buf_len += len(ids)
        if buf_len >= shard_tokens:
            arr = np.concatenate(buf)
            # write exactly shard_tokens, carry remainder
            cut = shard_tokens
            path = f"shard_{shard_id:04d}.bin"
            arr[:cut].tofile(os.path.join(out_dir, path))
            index["shards"].append({"path": path, "tokens": int(cut)})
            total += cut
            index["total_tokens"] = int(total)
            save_index()
            print(f"[shard] wrote {path} ({cut:,} tok); total={total:,}")
            shard_id += 1
            rem = arr[cut:]
            buf = [rem] if len(rem) else []
            buf_len = len(rem)
            if target_tokens and total >= target_tokens:
                break
    # final partial shard
    if buf_len > 0 and (not target_tokens or total < target_tokens):
        arr = np.concatenate(buf)
        if target_tokens:
            arr = arr[:max(0, target_tokens - total)]
        if len(arr):
            path = f"shard_{shard_id:04d}.bin"
            arr.tofile(os.path.join(out_dir, path))
            index["shards"].append({"path": path, "tokens": int(len(arr))})
            total += len(arr)
            index["total_tokens"] = int(total)
            save_index()
            print(f"[shard] wrote final {path} ({len(arr):,} tok)")
    print(f"[shard] DONE {out_dir}: {total:,} tokens in {len(index['shards'])} shards")
    return index


def _split_train_val(chunk, abs_start, n_train):
    """Route a token chunk (starting at absolute position abs_start) into
    train (positions < n_train) and val (>= n_train) segments."""
    end = abs_start + len(chunk)
    if end <= n_train:
        return [("train", chunk)]
    if abs_start >= n_train:
        return [("val", chunk)]
    cut = n_train - abs_start
    return [("train", chunk[:cut]), ("val", chunk[cut:])]


class PackedDataset(Dataset):
    """Memory-mapped packed token stream -> (x, y) blocks of length block_size."""
    def __init__(self, bin_path, block_size, dtype="uint16"):
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")
        self.n = (len(self.data) - 1) // block_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.block_size
        chunk = self.data[s: s + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
