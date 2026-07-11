"""Llama-style decoder-only transformer (GQA + RoPE + RMSNorm + SwiGLU).
Kept dependency-light so it works with plain PyTorch + FSDP2.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    vocab_size: int = 49152
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8          # GQA: n_kv_heads < n_heads
    ffn_hidden: int = 5632       # SwiGLU intermediate (~2.7x dim, multiple of 256)
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True


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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.rep = self.n_heads // self.n_kv_heads
        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, dim, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = Attention(args)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn = SwiGLU(args.dim, args.ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = precompute_rope(args.dim // args.n_heads, args.max_seq_len,
                                   args.rope_theta, device="cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

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
