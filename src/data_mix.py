"""Named data-source mixes for multi-source weighted pretraining.

A mix maps a *source key* -> weight. Source keys resolve to directories under
--data_root (e.g. key 'dclm' -> <data_root>/dclm_tok). Weights need not sum to
1 (they are normalized by the loader).

Use via train.py:  --data_root $S/data --data_mix mix_1t
Or pass an inline JSON:  --data_mix '{"dclm":0.39,"fineweb_edu":0.26}'
"""

# maps source key -> tokenized directory suffix
SOURCE_DIRS = {
    "dclm":        "dclm_tok",
    "fineweb_edu": "fineweb_edu_240bt_tok",   # fresh 240B tokenization for 1T mix
    "finepdfs":    "finepdfs_edu_tok",
    "code":        "starcoder_tok",           # bigcode/starcoderdata (gated, token)
    "math":        "math_tok",                # finemath-3plus
    "finephrase":  "finephrase_tok",
}

# 1T target mix (see docs/data_scaling_1T_design.md §2).
# User-approved final mix (plan B): web(dclm+fineweb+finepdfs)=55%, synthetic=20%,
# code=15%, math=10%.
MIX_1T = {
    "dclm":        0.32,   # 320B  high-quality web (CC-BY)
    "fineweb_edu": 0.18,   # 180B  educational web (ODC-By)
    "finepdfs":    0.05,   #  50B  PDF-sourced (ODC-By)
    "finephrase":  0.20,   # 200B  synthetic rewrite (ODC-By)
    "code":        0.15,   # 150B  starcoderdata multi-lang (HF token)
    "math":        0.10,   # 100B  finemath-3plus
}


# smaller validation mix that only needs sources tokenized so far; the loader
# silently drops sources whose dir is missing/empty, so this also works during
# incremental tokenization.
MIX_SMOKE = MIX_1T

MIXES = {
    "mix_1t": MIX_1T,
    "smoke":  MIX_SMOKE,
}


def resolve_mix(name_or_json, data_root):
    """Return {abs_source_dir: weight}, dropping sources whose dir is absent."""
    import os
    import json
    if name_or_json in MIXES:
        mix = MIXES[name_or_json]
    else:
        mix = json.loads(name_or_json)
    out = {}
    for key, w in mix.items():
        suffix = SOURCE_DIRS.get(key, key + "_tok")
        d = os.path.join(data_root, suffix)
        # accept dir with index.json OR legacy train.bin OR shard_*.bin
        import glob
        has = (os.path.exists(os.path.join(d, "index.json")) or
               os.path.exists(os.path.join(d, "train.bin")) or
               glob.glob(os.path.join(d, "shard_*.bin")) or
               glob.glob(os.path.join(d, "part_*", "index.json")))
        if has:
            out[d] = w
        else:
            print(f"[data_mix] skip missing source '{key}' -> {d}")
    if not out:
        raise ValueError(f"resolve_mix: no available sources for {name_or_json} under {data_root}")
    return out
