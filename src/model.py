"""Chimera: a decoder-only transformer.

Qwen3-style dense backbone (GQA + RoPE + RMSNorm + SwiGLU, no bias) plus two
Gemma-inspired capabilities:
  * QK-Norm  : RMSNorm on per-head query/key before attention (stabilizes
               training, suppresses loss spikes). Enabled by default.
  * Hybrid attention : each layer can be "full" (global causal) or "sliding"
               (local causal window). Wired for a Gemma-style 5:1 local:global
               interleave with a forced global last layer, but DISABLED by
               default (all layers full) for pretraining at 4K-8K context.
               Enable later for long-context (32K+) extension.

Kept dependency-light so it works with plain PyTorch + FSDP2.
"""
import math
from dataclasses import dataclass, field
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    vocab_size: int = 151936
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8          # GQA: n_kv_heads < n_heads
    ffn_hidden: int = 5632       # SwiGLU intermediate
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    # --- Gemma-inspired extras ---
    qk_norm: bool = True         # RMSNorm on per-head Q/K (default on)
    sliding_window: int = 0      # local window size; 0 = no sliding (all full)
    layer_types: Optional[List[str]] = None  # per-layer "full"/"sliding";
                                              # None -> all "full" (pretrain default)

    def resolved_layer_types(self):
        """Return a list of length n_layers of 'full'/'sliding'.
        If layer_types is None -> all full. If sliding_window<=0, everything is
        forced to full regardless (so the hybrid path is fully off by default)."""
        if self.layer_types is None or self.sliding_window <= 0:
            return ["full"] * self.n_layers
        assert len(self.layer_types) == self.n_layers
        return list(self.layer_types)


def make_gemma_layer_types(n_layers: int, ratio: int = 6, global_last: bool = True):
    """Helper for long-context stage: Gemma-style interleave with 1 global layer
    every `ratio` layers (i.e. 5:1 local:global when ratio=6) and a forced
    global last layer. NOT used during pretraining."""
    types = ["sliding"] * n_layers
    for i in range(ratio - 1, n_layers, ratio):
        types[i] = "full"
    if global_last:
        types[-1] = "full"
    return types


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def precompute_rope(dim: int, seq_len: int, theta: float, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)          # (seq, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    cos = cos[:T].view(1, 1, T, D // 2)
    sin = sin[:T].view(1, 1, T, D // 2)
    rx1 = x1 * cos - x2 * sin
    rx2 = x2 * cos + x1 * sin
    return torch.cat([rx1, rx2], dim=-1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str = "full"):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.rep = self.n_heads // self.n_kv_heads
        self.layer_type = layer_type
        self.sliding_window = args.sliding_window
        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        # QK-Norm: RMSNorm over head_dim, applied per head to Q and K.
        self.qk_norm = args.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, args.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        if self.layer_type == "sliding" and self.sliding_window > 0:
            # local causal window: token t attends to (t-window, t].
            attn_mask = _sliding_causal_mask(T, self.sliding_window, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


_mask_cache = {}


def _sliding_causal_mask(T, window, device):
    """Boolean additive mask (True=keep) for a causal sliding window.
    Cached per (T, window, device)."""
    key = (T, window, str(device))
    m = _mask_cache.get(key)
    if m is None:
        i = torch.arange(T, device=device).view(T, 1)
        j = torch.arange(T, device=device).view(1, T)
        keep = (j <= i) & (j > i - window)     # causal AND within window
        m = keep.view(1, 1, T, T)
        _mask_cache[key] = m
    return m


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, dim, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str = "full"):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = Attention(args, layer_type=layer_type)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn = SwiGLU(args.dim, args.ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Chimera(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        types = args.resolved_layer_types()
        self.tok_emb = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(args, layer_type=t) for t in types])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = precompute_rope(args.dim // args.n_heads, args.max_seq_len,
                                   args.rope_theta, device="cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        # Depth-scaled init for residual output projections (GPT-2/Llama style):
        # each residual branch's output std is divided by sqrt(2 * n_layers) so
        # the residual-stream variance does not grow with depth. Without this,
        # deep models (e.g. 8B/32L) diverge to NaN early in warmup.
        residual_std = 0.02 / math.sqrt(2 * args.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.wo.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.ffn.w2.weight, mean=0.0, std=residual_std)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        # cos/sin are registered buffers, so model.to(device) already moved them.
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if getattr(self, "fused_ce", False):
                # flash-attn Triton fused CE: online softmax, no fp32 logits
                # materialization. ~14GB less peak mem at 8B vocab (enables bigger
                # micro_bsz). Internally computes in fp32 -> preserves precision floor.
                from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _fa_ce
                flat_logits = logits.view(-1, logits.size(-1))
                flat_tgt = targets.view(-1)
                per_tok, _ = _fa_ce(flat_logits, flat_tgt, ignore_index=-100)
                mask = flat_tgt != -100
                loss = per_tok.sum() / mask.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    targets.view(-1),
                    ignore_index=-100,
                )
        return logits, loss

    def num_params(self):
        # parameters() already de-duplicates shared tensors, so a tied
        # lm_head/tok_emb weight is counted exactly once.
        return sum(p.numel() for p in self.parameters())


# Backwards-compat alias (older code / checkpoints referenced "Transformer").
Transformer = Chimera
