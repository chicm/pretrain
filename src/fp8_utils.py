"""FP8 training via torchao float8 (MI300X-validated).

Key findings (ROCm 7.1, MI300X, torchao 0.15) — see docs/fp8_results.md:
  * MI300 hipBLASLt supports the *fnuz* fp8 variants (e4m3fnuz/e5m2fnuz),
    NOT the OCP e4m3fn. torchao auto-selects fnuz when torchao.utils.is_MI300()
    is True, so no manual dtype override is needed on MI300X.
  * torch.compile is MANDATORY: eager-mode fp8 is ~2x SLOWER than bf16 due to
    unfused quant/scale ops. With compile, tensorwise fp8 is ~1.5x faster than
    bf16 on 8B MLP shapes.
  * We convert only the large Linear layers (attn/MLP projections) and skip the
    lm_head / tok_emb (tied, vocab-sized, numerically sensitive).

Usage: call convert_model_to_fp8(model) AFTER model build, BEFORE fully_shard
and torch.compile.
"""
import torch


def _is_mi300():
    try:
        from torchao.utils import is_MI300
        return bool(is_MI300())
    except Exception:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        return "MI300" in name


def convert_model_to_fp8(model, recipe: str = "tensorwise", log=print):
    """In-place convert eligible nn.Linear layers to torchao Float8Linear.

    recipe: 'tensorwise' (fastest on MI300X) or 'rowwise' (more accurate, ~1.4x).
    Returns the (mutated) model. Requires torch.compile downstream to be fast.
    """
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    from torchao.float8.config import Float8LinearRecipeName

    if recipe == "tensorwise":
        cfg = Float8LinearConfig()
    elif recipe == "rowwise":
        cfg = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE)
    else:
        raise ValueError(f"unknown fp8 recipe: {recipe!r}")

    # Skip lm_head/tok_emb: vocab-sized, tied weights, numerically sensitive.
    # Also skip any Linear whose in/out dims aren't multiples of 16 (fp8 gemm req).
    skip_substr = ("lm_head",)

    def _filter(mod, fqn: str):
        if any(s in fqn for s in skip_substr):
            return False
        if isinstance(mod, torch.nn.Linear):
            if (mod.in_features % 16) or (mod.out_features % 16):
                return False
            return True
        return False

    n_before = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    convert_to_float8_training(model, config=cfg, module_filter_fn=_filter)
    from torchao.float8.float8_linear import Float8Linear
    n_fp8 = sum(isinstance(m, Float8Linear) for m in model.modules())
    mi = _is_mi300()
    log(f"[fp8] recipe={recipe} converted {n_fp8}/{n_before} Linear layers "
        f"(is_MI300={mi}; dtype={'fnuz' if mi else 'ocp-e4m3fn'}). "
        f"torch.compile REQUIRED for speedup.")
    if not mi:
        log("[fp8] WARNING: not detected as MI300 -> fp8 gemm may be unsupported/slow.")
    return model
