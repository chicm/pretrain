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
    a legacy single train.bin/meta.json). Also merges parallel-worker
    subdirectories part_*/ (each with its own index.json) into one view;
    shard 'path' entries are made relative to src_dir."""
    # merged multi-part layout: src_dir/part_K/index.json
    parts = sorted(glob.glob(os.path.join(src_dir, "part_*")))
    def _part_index(p):
        # prefer index_v2.json (rebuilt, cache-consistent) over index.json
        v2 = os.path.join(p, "index_v2.json")
        v1 = os.path.join(p, "index.json")
        return v2 if os.path.exists(v2) else (v1 if os.path.exists(v1) else None)
    part_idx = [p for p in parts if _part_index(p)]
    if part_idx:
        merged = None
        shards = []
        for p in part_idx:
            with open(_part_index(p)) as f:
                pi = json.load(f)
            if merged is None:
                merged = {k: pi.get(k) for k in
                          ("dtype", "eot", "vocab_size")}
                merged["total_tokens"] = 0
            base = os.path.basename(p)
            for sh in pi["shards"]:
                shards.append({"path": os.path.join(base, sh["path"]),
                               "tokens": sh.get("tokens")})
            merged["total_tokens"] += pi.get("total_tokens", 0) or 0
        merged["shards"] = shards
        return merged

    # flat layout: prefer index_v2.json over index.json (cache-consistent rebuild)
    idx_path = os.path.join(src_dir, "index_v2.json")
    if not os.path.exists(idx_path):
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
            # .copy() -> own resizable storage; a from_numpy view of a slice
            # shares non-resizable storage and breaks DataLoader collate/pin.
            x = torch.from_numpy(chunk[:-1].copy())
            y = torch.from_numpy(chunk[1:].copy())
            yield x, y


class EpochMixtureDataset(IterableDataset):
    """OLMo-style multi-source loader: WITHOUT-replacement, shuffle-and-traverse.

    Contrast with WeightedMultiSourceDataset (random start, WITH replacement),
    which only covers ~63% of each source per 1 epoch of sampling and repeats
    the rest. This class instead partitions every source into non-overlapping
    fixed-length instances and traverses a shuffled permutation of them, so a
    source is fully covered (100%, no repeats) before any instance is seen
    twice -- matching OLMo's NumpyFSLDataset "concatenate & chunk + global
    shuffle" behaviour, extended to a weighted multi-source mixture.

    Semantics:
      * Each source is chunked into instances of length (block_size+1), stride
        block_size (consecutive instances share one boundary token, nanoGPT
        style). n_instances(shard) = (len(mm) - 1) // block_size.
      * A per-source cursor yields instances WITHOUT replacement: shards are
        visited in a shuffled order and each shard's instances are shuffled;
        when a source is exhausted it reshuffles and starts a new pass (so a
        small source cycles through full epochs, a large source may not finish
        even one pass -- both with zero mid-pass repetition).
      * Which source to draw from each step is chosen ~ per-source weight, so
        the training-visible mixture converges to the target ratios while every
        drawn instance is fresh within the current pass.
      * DDP/worker safe: instances are disjointly partitioned across every
        (rank, worker) by a global stride, so no instance is seen by two
        replicas within the same pass.

    Memory-light: no giant global permutation is materialised; shuffling is
    done at shard granularity plus a per-shard instance permutation
    (<= a few MB per shard).
    """
    def __init__(self, sources, block_size, seed=1337, rank=0, world=1,
                 count_sources=False, resume_skip=0):
        super().__init__()
        self.block_size = block_size
        self.seed = seed
        self.rank = rank
        self.world = world
        self.count_sources = count_sources
        self.resume_skip = int(resume_skip)
        self._counts = {}
        self.src_names = []
        self.src_weights = []
        self.src_shards = []   # per source: list of (abs_path, np.dtype, n_tokens)
        self._mm_cache = {}    # lazy per-worker memmap cache: (si,shi)->memmap
        for src_dir, w in sources.items():
            if w <= 0:
                continue
            idx = read_index(src_dir)
            dtype = np.dtype(idx["dtype"])
            itemsize = dtype.itemsize
            shards = []
            for sh in idx["shards"]:
                p = os.path.join(src_dir, sh["path"])
                # LAZY: do NOT memmap here (cold blobfuse open of hundreds of
                # shards x 64 ranks hangs for tens of minutes). Derive length
                # from index tokens or file size; open on first access in __iter__.
                ntok = sh.get("tokens")
                if not ntok:
                    try:
                        ntok = os.path.getsize(p) // itemsize
                    except OSError:
                        continue
                if ntok <= block_size + 1:
                    continue
                shards.append((p, dtype, int(ntok)))
            if not shards:
                continue
            self.src_names.append(os.path.basename(src_dir.rstrip("/")))
            self.src_weights.append(float(w))
            self.src_shards.append(shards)
        if not self.src_shards:
            raise ValueError("EpochMixtureDataset: no usable sources")
        sw = np.array(self.src_weights, dtype=np.float64)
        self.src_weights = sw / sw.sum()

    def source_counts(self):
        return dict(self._counts)

    def _get_mm(self, si, shi):
        """Lazily memmap shard (si,shi) on first access; cache per worker."""
        key = (si, shi)
        mm = self._mm_cache.get(key)
        if mm is None:
            path, dtype, _ = self.src_shards[si][shi]
            mm = np.memmap(path, dtype=dtype, mode="r")
            self._mm_cache[key] = mm
        return mm

    def _replica_owns(self, si, global_worker, n_global):
        """True if this (rank,worker) replica owns >=1 instance of source si.
        Pure metadata (no memmap open)."""
        bs = self.block_size
        for (_, _, ntok) in self.src_shards[si]:
            n_inst = (ntok - 1) // bs
            if n_inst > global_worker and n_inst > 0:
                # stride arange(global_worker, n_inst, n_global) non-empty
                return True
        return False

    def _source_cursor(self, si, rng, global_worker, n_global):
        """INFINITE generator of (shard_memmap, start) for source si, drawn
        without replacement within each pass, restricted to this (rank,worker)
        slice via a global stride, reshuffling on each new pass. Callers MUST
        only build cursors for sources this replica actually owns (see
        _replica_owns), otherwise this loops forever emitting nothing."""
        shards = self.src_shards[si]
        bs = self.block_size
        while True:
            # shuffle shard visitation order for this pass
            shard_order = rng.permutation(len(shards))
            for shi in shard_order:
                _, _, ntok = shards[shi]         # metadata only; no open yet
                n_inst = (ntok - 1) // bs
                if n_inst <= 0:
                    continue
                # this (rank,worker)'s disjoint slice of instances in this shard
                local = np.arange(global_worker, n_inst, n_global) \
                    if global_worker < n_inst else np.empty(0, dtype=np.int64)
                if local.size == 0:
                    continue
                rng.shuffle(local)
                mm = self._get_mm(si, shi)       # LAZY open only when we read
                for k in local:
                    yield mm, int(k) * bs

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        n_global = self.world * num_workers
        global_worker = self.rank * num_workers + worker_id
        # source-selection RNG (same across shards); per (rank,worker) stream
        sel_rng = np.random.default_rng(
            self.seed + self.rank * 100003 + worker_id * 10007)
        n_src = len(self.src_shards)
        # Determine which sources THIS replica actually owns instances of.
        # Small sources (owm/infimath) split across world*num_workers may leave
        # some replicas with zero instances. We zero their weight ONCE upfront
        # (deterministic, no mid-stream weight mutation) so every replica emits
        # an INFINITE, well-defined batch stream -> fixed step count keeps all
        # ranks in NCCL lockstep (fixes collective-timeout SIGABRT).
        owned = [si for si in range(n_src)
                 if self._replica_owns(si, global_worker, n_global)]
        if not owned:
            owned = list(range(n_src))  # degenerate safety; shouldn't happen
        w = np.asarray(self.src_weights, dtype=np.float64).copy()
        keep = np.zeros(n_src, dtype=bool)
        keep[owned] = True
        w = np.where(keep, w, 0.0)
        wsum = w.sum()
        sel_p = (w / wsum) if wsum > 0 else None
        # infinite cursors ONLY for owned sources
        cursors = {si: self._source_cursor(
                       si,
                       np.random.default_rng(self.seed + 777 * (si + 1)
                                             + self.rank * 100003
                                             + worker_id * 10007),
                       global_worker, n_global)
                   for si in owned}
        bs1 = self.block_size + 1
        # deterministic fast-forward for resume: this replica's total skip is
        # split evenly across its workers (main loop consumes round-robin).
        skip = self.resume_skip // max(1, num_workers)
        produced = 0
        while True:
            si = int(sel_rng.choice(n_src, p=sel_p))
            mm, start = next(cursors[si])   # cursors are infinite -> never stops
            # fast-forward: advance the SAME deterministic stream but don't yield
            if produced < skip:
                produced += 1
                continue
            chunk = np.asarray(mm[start:start + bs1]).astype(np.int64)
            if self.count_sources:
                name = self.src_names[si]
                self._counts[name] = self._counts.get(name, 0) + 1
            x = torch.from_numpy(chunk[:-1].copy())
            y = torch.from_numpy(chunk[1:].copy())
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


def _config_prefix(dataset_name, hf_config):
    """Best-effort mapping from (dataset, config) -> parquet path prefix, so we
    can list & partition just this config's files across parallel workers."""
    if dataset_name == "HuggingFaceFW/fineweb-edu" and hf_config:
        # sample-350BT -> sample/350BT/
        if hf_config.startswith("sample-"):
            return "sample/" + hf_config[len("sample-"):] + "/"
    if dataset_name == "HuggingFaceTB/finemath" and hf_config:
        return hf_config + "/"          # finemath-3plus/
    # dclm / finephrase / finepdfs-edu: single default config, use all parquet
    return None


def list_config_files(dataset_name, hf_config):
    """Return sorted list of parquet file paths for this dataset+config.

    Caches the (expensive) list_repo_files enumeration to a shared-disk JSON so
    that many parallel file-shard workers don't each hammer the HF API (a repo
    like dclm has ~28k files -> ~280 paginated requests per enumeration; N
    workers * retries trivially exceeds the 1000-req/5min quota). First worker
    populates the cache; the rest read it with zero API calls.
    """
    import os, json, hashlib, time
    pref_key = f"{dataset_name}::{hf_config}"
    cache_dir = os.environ.get("FILELIST_CACHE_DIR", "/tmp")
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5(pref_key.encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"filelist_{h}.json")
    # fast path: valid cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                data = json.load(fh)
            if data.get("key") == pref_key and data.get("files"):
                return data["files"]
        except Exception:
            pass
    # lock so only one worker enumerates; others wait for the cache file
    lock_path = cache_path + ".lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        got_lock = True
    except FileExistsError:
        got_lock = False
    if not got_lock:
        # wait up to 10 min for the enumerating worker to publish the cache
        for _ in range(600):
            if os.path.exists(cache_path):
                try:
                    with open(cache_path) as fh:
                        data = json.load(fh)
                    if data.get("key") == pref_key and data.get("files"):
                        return data["files"]
                except Exception:
                    pass
            time.sleep(1)
        # fall through to enumerate ourselves as a last resort
    from huggingface_hub import HfApi
    api = HfApi()
    files = [f for f in api.list_repo_files(dataset_name, repo_type="dataset")
             if f.endswith(".parquet")]
    pref = _config_prefix(dataset_name, hf_config)
    if pref:
        files = [f for f in files if f.startswith(pref)]
    files = sorted(files)
    try:
        tmp = cache_path + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump({"key": pref_key, "files": files}, fh)
        os.replace(tmp, cache_path)
    except Exception:
        pass
    finally:
        if got_lock:
            try:
                os.remove(lock_path)
            except Exception:
                pass
    return files



def prepare_data_sharded(dataset_name, out_dir, tokenizer, split="train",
                         text_key="text", hf_config=None,
                         eot_id=None, shard_tokens=1_000_000_000,
                         target_tokens=None, num_proc=32, data_files=None,
                         streaming=True, file_shards=1, file_shard_id=0):
    """Streaming, resumable, sharded tokenizer for very large sources (1T corpus).

    Writes shard_XXXX.bin (uint16/uint32) of ~shard_tokens each + index.json.
    Resumable: on restart, completed shards in index.json are kept and we skip
    ahead by their token count (approximate doc-level skip via HF streaming).
    Stops when target_tokens reached (or dataset exhausted).

    Parallelism: when file_shards>1, only this worker's slice (file_shard_id of
    file_shards) of the dataset's parquet files is processed. Each worker should
    use a distinct out_dir (e.g. <dir>/part_K). Resume-by-skip then only skips
    within this worker's own file subset, which is correct.
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

    load_kwargs = dict(split=split, streaming=streaming)
    if file_shards > 1:
        # partition this config's parquet files across workers
        all_files = list_config_files(dataset_name, hf_config)
        my_files = all_files[file_shard_id::file_shards]
        print(f"[shard] file-shard {file_shard_id}/{file_shards}: "
              f"{len(my_files)}/{len(all_files)} parquet files", flush=True)
        load_kwargs["data_files"] = my_files
        ds = load_dataset(dataset_name, data_files=my_files, split=split,
                          streaming=streaming)
    else:
        load_kwargs["name"] = hf_config
        if data_files is not None:
            load_kwargs["data_files"] = data_files
        ds = load_dataset(dataset_name, **load_kwargs)

    def save_index():
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)

    # skip already-consumed docs approximately by token count
    import time
    skip_tokens = done_tokens
    buf = []
    buf_len = 0
    total = done_tokens
    skipped = 0
    ENC_BATCH = 1024          # docs encoded per fast-tokenizer call (GIL released)
    t0 = time.time()
    last_log = t0
    since_log = 0

    def encode_batch(texts):
        out = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return out

    txt_batch = []

    def process_ids_list(ids_list):
        nonlocal buf, buf_len, total, skipped, shard_id, since_log
        for ids in ids_list:
            ids.append(eot_id)
            n = len(ids)
            if skipped < skip_tokens:
                skipped += n
                continue
            buf.append(np.asarray(ids, dtype=dtype))
            buf_len += n
            since_log += n
            if buf_len >= shard_tokens:
                arr = np.concatenate(buf)
                cut = shard_tokens
                path = f"shard_{shard_id:04d}.bin"
                arr[:cut].tofile(os.path.join(out_dir, path))
                index["shards"].append({"path": path, "tokens": int(cut)})
                total += cut
                index["total_tokens"] = int(total)
                save_index()
                print(f"[shard] wrote {path} ({cut:,} tok); total={total:,}", flush=True)
                shard_id += 1
                rem = arr[cut:]
                buf = [rem] if len(rem) else []
                buf_len = len(rem)
                if target_tokens and total >= target_tokens:
                    return True
        return False

    stop = False
    for ex in ds:
        txt = ex.get(text_key)
        if not txt:
            continue
        txt_batch.append(txt)
        if len(txt_batch) >= ENC_BATCH:
            if process_ids_list(encode_batch(txt_batch)):
                stop = True
            txt_batch = []
            now = time.time()
            if now - last_log >= 30:
                rate = since_log / (now - last_log)
                done_now = total + buf_len
                eta_h = ((target_tokens - done_now) / rate / 3600
                         if target_tokens and rate > 0 else -1)
                print(f"[rate] {done_now:,} tok | {rate/1000:.1f}K tok/s | "
                      f"eta {eta_h:.1f}h", flush=True)
                last_log = now
                since_log = 0
            if stop:
                break
    if not stop and txt_batch:
        process_ids_list(encode_batch(txt_batch))

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
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y
