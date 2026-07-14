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
    "code":        "code_tok",                # codeparrot-clean (Python, non-gated)
    "math":        "math_tok",                # finemath-3plus + open-web-math
    "finephrase":  "finephrase_tok",
}

# 1T target mix (see docs/data_scaling_1T_design.md §2, revised for non-gated sources).
# NOTE: bigcode/StarCoder code sets are HF-gated and no token is available on the
# cluster, so code is limited to codeparrot-clean (Python). If a token is later
# provided, swap in the-stack-v2 and raise the code share.
MIX_1T = {
    "dclm":        0.42,   # 420B  high-quality web (CC-BY)
    "fineweb_edu": 0.24,   # 240B  educational web (ODC-By)
    "finepdfs":    0.08,   #  80B  PDF-sourced (ODC-By), quality >= Nemotron-CC-v2
    "math":        0.12,   # 120B  finemath-3plus + open-web-math
    "code":        0.04,   #  40B  codeparrot-clean Python (non-gated)
    "finephrase":  0.10,   # 100B  synthetic rewrite (ODC-By)
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
               glob.glob(os.path.join(d, "shard_*.bin")))
        if has:
            out[d] = w
        else:
            print(f"[data_mix] skip missing source '{key}' -> {d}")
    if not out:
        raise ValueError(f"resolve_mix: no available sources for {name_or_json} under {data_root}")
    return out
