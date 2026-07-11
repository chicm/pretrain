"""Data pipeline: tokenize a HF dataset into a single packed uint16/uint32 .bin,
then stream fixed-length blocks for causal LM training.

Two steps:
  1) prepare_data(): tokenize -> data/<name>_train.bin (+ val.bin). Run once.
  2) PackedDataset: memory-maps the .bin and yields (x, y) blocks.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def prepare_data(dataset_name, out_dir, tokenizer, split="train",
                 text_key="text", num_proc=32, val_frac=0.0005,
                 hf_config=None, streaming=False, max_docs=None):
    """Tokenize a HF dataset into packed .bin files. GPT2 vocab -> uint16."""
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    eot = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
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
