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
    "infimath":    "infimath_tok",            # infiwebmath-3plus
    "owm":         "owm_tok",                 # open-web-math/open-web-math
    "finephrase":  "finephrase_tok",
}

# 1T target mix (see docs/data_scaling_1T_design.md §2, §9).
# Plan B (user-approved): math limited to ~1 epoch of the actually-available unique
# math tokens (~66B total across 3 sources) instead of forcing 100B via repetition.
# The freed ~3.4% is reallocated to dclm (largest, safest to upweight).
#   web(dclm+fineweb+finepdfs) = 58.4%, synthetic(finephrase) = 20%, code = 15%,
#   math(3 sources, ~1 epoch each) = 6.6%.
# Math split is proportional to each source's on-disk unique tokens (1 epoch):
#   math_tok(finemath-3plus 31.2B)=3.12%, infimath(infiwebmath 21.6B)=2.16%,
#   owm(OpenWebMath 13.2B)=1.32%.
MIX_1T = {
    "dclm":        0.354,  # ~354B  high-quality web (CC-BY); absorbs freed math budget
    "fineweb_edu": 0.18,   #  180B  educational web (ODC-By)
    "finepdfs":    0.05,   #   50B  PDF-sourced (ODC-By)
    "finephrase":  0.20,   #  200B  synthetic rewrite (ODC-By)
    "code":        0.15,   #  150B  starcoderdata multi-lang (HF token)
    "math":        0.0312, #  ~31B  finemath-3plus (~1 epoch)
    "infimath":    0.0216, #  ~22B  infiwebmath-3plus (~1 epoch)
    "owm":         0.0132, #  ~13B  OpenWebMath (~1 epoch)
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
