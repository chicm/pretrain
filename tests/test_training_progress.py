import math

import pytest

from training_progress import cosine_lr, resume_skip_for_rank, tokens_before_step


OLD_TOKENS_PER_STEP = 64 * 4 * 4 * 4096
NEW_TOKENS_PER_STEP = 120 * 4 * 2 * 4096
RESUME_STEP = 18001
RESUME_TOKENS = RESUME_STEP * OLD_TOKENS_PER_STEP


def test_world_size_change_preserves_global_sequence_depth():
    skips = [resume_skip_for_rank(RESUME_TOKENS, 4096, 120, rank)
             for rank in range(120)]
    assert skips[:64] == [153609] * 64
    assert skips[64:] == [153608] * 56
    assert sum(skips) == RESUME_TOKENS // 4096


def test_resumed_token_progress_is_continuous():
    assert tokens_before_step(RESUME_STEP, RESUME_STEP, RESUME_TOKENS,
                              NEW_TOKENS_PER_STEP) == RESUME_TOKENS
    assert tokens_before_step(RESUME_STEP + 7, RESUME_STEP, RESUME_TOKENS,
                              NEW_TOKENS_PER_STEP) == RESUME_TOKENS + 7 * NEW_TOKENS_PER_STEP


def test_token_lr_matches_old_schedule_at_resume():
    total = 238000 * OLD_TOKENS_PER_STEP
    warmup = 1500 * OLD_TOKENS_PER_STEP
    token_lr = cosine_lr(RESUME_TOKENS, warmup, total, 2.8e-4, 2.8e-5)
    step_lr = cosine_lr(RESUME_STEP, 1500, 238000, 2.8e-4, 2.8e-5)
    assert math.isclose(token_lr, step_lr, rel_tol=0, abs_tol=1e-15)


def test_resume_tokens_must_align_to_sequence():
    with pytest.raises(ValueError, match="divisible"):
        resume_skip_for_rank(RESUME_TOKENS + 1, 4096, 120, 0)
