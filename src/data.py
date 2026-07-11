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
    else:
        ds = ds.map(tok, remove_columns=ds.column_names, num_proc=num_proc,
                    desc="tokenizing")
        total = int(np.sum(ds["len"], dtype=np.int64))
        arr = np.zeros(total, dtype=dtype)
        idx = 0
        for batch in ds.iter(batch_size=1000):
            for ids in batch["ids"]:
                arr[idx: idx + len(ids)] = ids
                idx += len(ids)

    n_val = int(len(arr) * val_frac)
    if n_val > 0:
        arr[:-n_val].tofile(os.path.join(out_dir, "train.bin"))
        arr[-n_val:].tofile(os.path.join(out_dir, "val.bin"))
    else:
        arr.tofile(os.path.join(out_dir, "train.bin"))
    meta = {"dtype": np.dtype(dtype).name, "vocab_size": tokenizer.vocab_size,
            "eot": eot, "total_tokens": int(len(arr))}
    import json
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare_data] wrote {len(arr):,} tokens to {out_dir} (dtype={dtype})")
    return meta


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
