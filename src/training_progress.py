"""World-size-independent training progress helpers."""
import math


def cosine_lr(position: float, warmup: float, total: float,
              peak_lr: float, min_lr: float) -> float:
    """Warmup + cosine LR at a step or token position."""
    if total <= warmup:
        raise ValueError("total must be greater than warmup")
    if position < warmup:
        return peak_lr * position / max(1.0, warmup)
    ratio = min(1.0, (position - warmup) / (total - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def tokens_before_step(step: int, resume_step: int,
                       resume_consumed_tokens: int,
                       current_tokens_per_step: int) -> int:
    """Cumulative tokens before ``step`` in a resumed run."""
    if step < resume_step:
        raise ValueError("step cannot precede resume_step")
    return resume_consumed_tokens + (step - resume_step) * current_tokens_per_step


def resume_skip_for_rank(consumed_tokens: int, block_size: int,
                         world_size: int, rank: int) -> int:
    """Distribute globally consumed sequences across a new world size."""
    if consumed_tokens < 0:
        raise ValueError("consumed_tokens must be non-negative")
    if consumed_tokens % block_size:
        raise ValueError("consumed_tokens must be divisible by block_size")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    global_sequences = consumed_tokens // block_size
    base, remainder = divmod(global_sequences, world_size)
    return base + int(rank < remainder)
