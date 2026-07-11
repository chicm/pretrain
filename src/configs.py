"""Model + training configs. Small (~1B) for smoke test, plus a proxy tiny one."""
from dataclasses import dataclass
from model import ModelArgs


# --- Model presets ---
def _tiny():   # ~50M, for TinyStories smoke test on 1 GPU
    return ModelArgs(vocab_size=50304, dim=512, n_layers=8, n_heads=8,
                     n_kv_heads=4, ffn_hidden=1408, max_seq_len=1024)

def _1b():     # ~1.1B, real pipeline validation on FineWeb-10BT
    return ModelArgs(vocab_size=50304, dim=2048, n_layers=24, n_heads=16,
                     n_kv_heads=8, ffn_hidden=5632, max_seq_len=2048)

def _8b():     # ~8B, production (for later, on MI300)
    return ModelArgs(vocab_size=50304, dim=4096, n_layers=32, n_heads=32,
                     n_kv_heads=8, ffn_hidden=14336, max_seq_len=4096)

MODELS = {"tiny": _tiny, "1b": _1b, "8b": _8b}


@dataclass
class TrainConfig:
    model: str = "1b"
    data_dir: str = "./data/fineweb_tok"
    out_dir: str = "./checkpoints"
    block_size: int = 2048
    micro_bsz: int = 8            # per-GPU micro batch
    grad_accum: int = 8          # -> global batch = micro*accum*world_size*block
    max_steps: int = 20000
    warmup_steps: int = 200
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 50       # how many val batches to average per eval
    ckpt_every: int = 200
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 1337
