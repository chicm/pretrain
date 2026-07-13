"""Export a Chimera .pt checkpoint to Hugging Face Qwen3 format.

Chimera is architecturally a Qwen3 dense model (GQA + RoPE + RMSNorm pre-norm +
SwiGLU + QK-Norm + no bias + tied embeddings, Qwen3 vocab). Its RoPE uses the
same split-half ("rotate_half"/NeoX) convention as HF Qwen3, so NO weight
permutation of q/k is required -- names map 1:1.

Output: a directory with config.json + model.safetensors (+ tokenizer copied
from Qwen/Qwen3-8B) that vLLM / transformers can load as Qwen3ForCausalLM.

Usage:
    python export_hf.py --ckpt /path/ckpt_8000.pt --out /path/hf_8b_step8000
    # then verify logits parity:
    python export_hf.py --ckpt ... --out ... --verify
"""
import argparse
import json
import os
import shutil

import torch

from model import Chimera, ModelArgs
from configs import MODELS


def build_model_args(cfg: dict) -> ModelArgs:
    preset = cfg.get("model", "1b")
    if preset not in MODELS:
        raise ValueError(f"unknown model preset in ckpt cfg: {preset!r}")
    return MODELS[preset]()


def load_chimera(path: str):
    print(f"[export] loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    step = ckpt.get("step", "?")
    margs = build_model_args(cfg)
    print(f"[export] preset={cfg.get('model','?')} step={step} dim={margs.dim} "
          f"n_layers={margs.n_layers} n_heads={margs.n_heads} "
          f"n_kv_heads={margs.n_kv_heads} ffn={margs.ffn_hidden} "
          f"vocab={margs.vocab_size} max_seq_len={margs.max_seq_len} "
          f"qk_norm={margs.qk_norm} tie={margs.tie_embeddings}")
    model = Chimera(margs)
    sd = ckpt["model"]
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing:
        print(f"[export][warn] missing keys: {missing[:8]}")
    if unexpected:
        print(f"[export][warn] unexpected keys: {unexpected[:8]}")
    model.eval()
    return model, margs, cfg, step


def chimera_to_hf_state_dict(model: Chimera, margs: ModelArgs) -> dict:
    """Rename Chimera params to HF Qwen3 param names. No permutation needed."""
    src = model.state_dict()
    out = {}

    def take(name):
        if name not in src:
            raise KeyError(f"expected param missing from Chimera sd: {name}")
        return src[name]

    # embeddings
    out["model.embed_tokens.weight"] = take("tok_emb.weight")

    for i in range(margs.n_layers):
        p = f"layers.{i}."
        h = f"model.layers.{i}."
        # attention
        out[h + "self_attn.q_proj.weight"] = take(p + "attn.wq.weight")
        out[h + "self_attn.k_proj.weight"] = take(p + "attn.wk.weight")
        out[h + "self_attn.v_proj.weight"] = take(p + "attn.wv.weight")
        out[h + "self_attn.o_proj.weight"] = take(p + "attn.wo.weight")
        if margs.qk_norm:
            out[h + "self_attn.q_norm.weight"] = take(p + "attn.q_norm.weight")
            out[h + "self_attn.k_norm.weight"] = take(p + "attn.k_norm.weight")
        # norms
        out[h + "input_layernorm.weight"] = take(p + "attn_norm.weight")
        out[h + "post_attention_layernorm.weight"] = take(p + "ffn_norm.weight")
        # mlp (SwiGLU): w1=gate, w3=up, w2=down
        out[h + "mlp.gate_proj.weight"] = take(p + "ffn.w1.weight")
        out[h + "mlp.up_proj.weight"] = take(p + "ffn.w3.weight")
        out[h + "mlp.down_proj.weight"] = take(p + "ffn.w2.weight")

    out["model.norm.weight"] = take("norm.weight")
    # lm_head (tied to embeddings in Chimera) -- write explicit weight; HF with
    # tie_word_embeddings=True will ignore/retie, but writing it is harmless.
    lm_head = src.get("lm_head.weight", src["tok_emb.weight"])
    out["lm_head.weight"] = lm_head
    return out


def build_hf_config(margs: ModelArgs) -> dict:
    head_dim = margs.dim // margs.n_heads
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": margs.dim,
        "intermediate_size": margs.ffn_hidden,
        "num_hidden_layers": margs.n_layers,
        "num_attention_heads": margs.n_heads,
        "num_key_value_heads": margs.n_kv_heads,
        "head_dim": head_dim,
        "vocab_size": margs.vocab_size,
        "max_position_embeddings": margs.max_seq_len,
        "rms_norm_eps": margs.norm_eps,
        "rope_theta": margs.rope_theta,
        "tie_word_embeddings": bool(margs.tie_embeddings),
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "torch_dtype": "bfloat16",
        "bos_token_id": None,
        "eos_token_id": 151643,
        "use_cache": True,
    }


def export(args):
    model, margs, cfg, step = load_chimera(args.ckpt)
    os.makedirs(args.out, exist_ok=True)

    hf_sd = chimera_to_hf_state_dict(model, margs)
    hf_sd = {k: v.to(torch.bfloat16).contiguous() for k, v in hf_sd.items()}

    from safetensors.torch import save_file
    # tied weights share storage; safetensors needs distinct tensors.
    if margs.tie_embeddings:
        # drop lm_head.weight to avoid shared-storage error; HF reties it.
        hf_sd.pop("lm_head.weight", None)
    save_file(hf_sd, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})

    conf = build_hf_config(margs)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(conf, f, indent=2)
    print(f"[export] wrote config.json + model.safetensors to {args.out}")

    # copy tokenizer files from the HF cache / hub
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.save_pretrained(args.out)
    print(f"[export] saved tokenizer ({args.tokenizer}) to {args.out}")

    if args.verify:
        verify_parity(model, margs, args.out, args.tokenizer)


@torch.no_grad()
def verify_parity(model, margs, out_dir, tokenizer_name, n_tok=32):
    """Compare Chimera native logits vs HF-loaded Qwen3 logits on a sample."""
    print("[verify] loading exported model with transformers ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    text = "The quick brown fox jumps over the lazy dog. In this paper we"
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tok]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    chi_logits, _ = model(ids.to(dev))
    chi_logits = chi_logits.float().cpu()

    hf = AutoModelForCausalLM.from_pretrained(
        out_dir, torch_dtype=torch.float32).to(dev).eval()
    hf_logits = hf(ids.to(dev)).logits.float().cpu()

    diff = (chi_logits - hf_logits).abs()
    print(f"[verify] logits shape chi={tuple(chi_logits.shape)} "
          f"hf={tuple(hf_logits.shape)}")
    print(f"[verify] max|diff|={diff.max().item():.4e} "
          f"mean|diff|={diff.mean().item():.4e}")
    # argmax agreement on next-token predictions
    agree = (chi_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    print(f"[verify] argmax agreement={agree*100:.2f}%")
    if diff.max().item() < 1e-1 and agree > 0.99:
        print("[verify] PASS: exported model matches Chimera.")
    else:
        print("[verify] WARN: mismatch too large -- check mapping/RoPE/qk_norm.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--verify", action="store_true",
                    help="after export, compare logits vs native Chimera")
    export(ap.parse_args())


if __name__ == "__main__":
    main()
