"""FSDP2 pretraining loop. Plain PyTorch (torch>=2.4) using torch.distributed.fsdp.fully_shard.

Launch with torchrun, e.g.:
  torchrun --nproc_per_node=8 train.py --model 1b --data_dir /path/tok

Multi-node is handled by run_multinode.sh / launch_multinode.sh (sets MASTER_ADDR / rendezvous).
"""
import os
import math
import json
import time
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
)
from torch.utils.checkpoint import (
    create_selective_checkpoint_contexts, CheckpointPolicy,
)

from model import Transformer
from model import Block as TransformerBlock
from data import PackedDataset
from configs import MODELS, TrainConfig


def is_master():
    return int(os.environ.get("RANK", 0)) == 0


def log(*a):
    if is_master():
        print(*a, flush=True)


def setup_dist():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def save_ckpt(model, opt, step, cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    # gather full state dicts — COLLECTIVE: all ranks must participate
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict, get_optimizer_state_dict, StateDictOptions)
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd = get_model_state_dict(model, options=opts)
    opt_sd = get_optimizer_state_dict(model, opt, options=opts)
    if not is_master():
        return
    path = os.path.join(cfg.out_dir, f"ckpt_{step}.pt")
    torch.save({"model": model_sd, "opt": opt_sd, "step": step,
                "cfg": cfg.__dict__}, path)
    log(f"[ckpt] saved {path}")


def load_ckpt(model, opt, path):
    """Restore model + optimizer from a full-state-dict checkpoint. Returns next step."""
    from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict, set_optimizer_state_dict, StateDictOptions)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    set_model_state_dict(model, ckpt["model"], options=opts)
    set_optimizer_state_dict(model, opt, ckpt["opt"], options=opts)
    log(f"[ckpt] resumed from {path} at step {ckpt['step']}")
    return ckpt["step"] + 1


@torch.no_grad()
def evaluate(model, loader, max_batches):
    """Average loss over a few val batches. Returns None if no val data."""
    if loader is None:
        return None
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)
        _, loss = model(x, y)
        total += loss.item()
        n += 1
        if n >= max_batches:
            break
    model.train()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--micro_bsz", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--activation_checkpoint", action="store_true",
                    help="enable per-Block non-reentrant activation checkpointing (full recompute)")
    ap.add_argument("--selective_ac", action="store_true",
                    help="selective activation checkpointing: save expensive matmul/SDPA outputs, "
                         "recompute cheap elementwise/norm. Overrides --activation_checkpoint.")
    ap.add_argument("--resume", default=None, help="path to a ckpt_*.pt to resume from")
    ap.add_argument("--reduce_bf16", action="store_true",
                    help="use bf16 gradient reduce_dtype (default fp32); speed A/B, watch grad_norm")
    ap.add_argument("--hsdp_shard", type=int, default=0,
                    help="if >0, use HSDP 2D mesh: shard group size = this (e.g. 8 = intra-node), "
                         "replicate group = world/shard. 0 = full FSDP2 sharding (default)")
    ap.add_argument("--fused_ce", action="store_true",
                    help="use flash-attn Triton fused cross-entropy (online softmax, no fp32 "
                         "logits materialization; ~14GB less peak mem at 8B vocab -> bigger micro_bsz)")
    args = ap.parse_args()

    cfg = TrainConfig()
    for k in ["model", "data_dir", "out_dir", "max_steps", "micro_bsz", "grad_accum"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.no_compile:
        cfg.compile = False
    if args.activation_checkpoint:
        cfg.activation_checkpoint = True
    cfg.selective_ac = bool(getattr(args, "selective_ac", False))
    if cfg.selective_ac:
        cfg.activation_checkpoint = True  # selective implies AC path

    # Per-model training-stability overrides. Larger/deeper models need a
    # smaller peak LR and longer warmup to stay numerically stable in bf16.
    _MODEL_OPT = {
        "8b": dict(lr=2e-4, min_lr=2e-5, warmup_steps=500),
    }
    for k, v in _MODEL_OPT.get(cfg.model, {}).items():
        setattr(cfg, k, v)

    local_rank = setup_dist()
    world = dist.get_world_size()
    torch.manual_seed(cfg.seed)

    # --- data ---
    meta = json.load(open(os.path.join(cfg.data_dir, "meta.json")))
    margs = MODELS[cfg.model]()
    margs.vocab_size = max(margs.vocab_size, meta["vocab_size"])
    cfg.block_size = margs.max_seq_len
    train_ds = PackedDataset(os.path.join(cfg.data_dir, "train.bin"),
                             cfg.block_size, meta["dtype"])
    sampler = DistributedSampler(train_ds, num_replicas=world,
                                 rank=dist.get_rank(), shuffle=True)
    loader = DataLoader(train_ds, batch_size=cfg.micro_bsz, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)

    # optional val set (written by prepare_data when val_frac > 0)
    val_loader = None
    val_path = os.path.join(cfg.data_dir, "val.bin")
    if os.path.exists(val_path):
        val_ds = PackedDataset(val_path, cfg.block_size, meta["dtype"])
        val_loader = DataLoader(val_ds, batch_size=cfg.micro_bsz, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=True)

    # --- model + FSDP2 ---
    model = Transformer(margs).to("cuda")
    model.fused_ce = bool(getattr(args, "fused_ce", False))
    if model.fused_ce:
        log("[ce] flash-attn Triton fused cross-entropy ON")
    log(f"[model] {cfg.model}: {model.num_params()/1e9:.3f}B params, "
        f"vocab={margs.vocab_size}, world={world}")
    reduce_dtype = torch.bfloat16 if getattr(args, "reduce_bf16", False) else torch.float32
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype)
    log(f"[precision] param_dtype=bf16 reduce_dtype={'bf16' if reduce_dtype==torch.bfloat16 else 'fp32'}")
    # HSDP: build a 2D device mesh (replicate, shard). shard group stays intra-node so
    # param all-gather is limited to the fast intra-node fabric; cross-node does grad all-reduce.
    fsdp_kw = {}
    hsdp_shard = getattr(args, "hsdp_shard", 0)
    if hsdp_shard and hsdp_shard > 0:
        assert world % hsdp_shard == 0, f"world {world} not divisible by hsdp_shard {hsdp_shard}"
        replicate = world // hsdp_shard
        mesh = init_device_mesh("cuda", (replicate, hsdp_shard),
                                mesh_dim_names=("replicate", "shard"))
        fsdp_kw["mesh"] = mesh
        log(f"[hsdp] 2D mesh replicate={replicate} x shard={hsdp_shard} (world={world})")
    else:
        log(f"[hsdp] full FSDP2 sharding (1D, world={world})")
    # Activation checkpointing: wrap each Block (non-reentrant) BEFORE sharding.
    # Trades recompute for a large drop in activation memory -> enables bigger micro_bsz.
    if cfg.activation_checkpoint:
        if getattr(cfg, "selective_ac", False):
            # Selective AC: SAVE expensive ops (matmul/addmm/SDPA) so they are NOT recomputed;
            # recompute the cheap elementwise/norm/RoPE. Best throughput/memory balance.
            _save_ops = {
                torch.ops.aten.mm.default,
                torch.ops.aten.addmm.default,
                torch.ops.aten.bmm.default,
                torch.ops.aten._scaled_dot_product_flash_attention.default,
                torch.ops.aten._scaled_dot_product_efficient_attention.default,
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
            }
            def _sac_policy(ctx, op, *a, **k):
                return (CheckpointPolicy.MUST_SAVE if op in _save_ops
                        else CheckpointPolicy.PREFER_RECOMPUTE)
            def _ctx_fn():
                return create_selective_checkpoint_contexts(_sac_policy)
            for i, layer in enumerate(model.layers):
                model.layers[i] = checkpoint_wrapper(
                    layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    preserve_rng_state=False, context_fn=_ctx_fn)
            log(f"[ac] SELECTIVE activation checkpointing ON for {len(model.layers)} blocks "
                f"(save matmul/SDPA, recompute elementwise)")
        else:
            for i, layer in enumerate(model.layers):
                model.layers[i] = checkpoint_wrapper(
                    layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    preserve_rng_state=False)
            log(f"[ac] FULL activation checkpointing ON for {len(model.layers)} blocks")
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp, **fsdp_kw)
    fully_shard(model, mp_policy=mp, **fsdp_kw)
    if cfg.compile:
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay, fused=True)

    # --- train loop ---
    # TensorBoard writer on master rank only (event files -> shared disk)
    writer = None
    if is_master() and getattr(cfg, "tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = cfg.tb_dir or os.path.join(cfg.out_dir, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
            log(f"[tb] TensorBoard logging to {tb_dir}")
        except Exception as e:
            log(f"[tb] disabled ({e})")
    model.train()
    step = 0
    skipped_steps = 0
    if args.resume:
        step = load_ckpt(model, opt, args.resume)
    t0 = time.time()
    tokens_per_step = cfg.micro_bsz * cfg.grad_accum * world * cfg.block_size
    data_iter = iter(loader)
    epoch = 0
    while step < cfg.max_steps:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_mean = 0.0   # accumulates already-divided losses -> equals the mean
        for micro in range(cfg.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)
            _, loss = model(x, y)
            loss = loss / cfg.grad_accum
            loss.backward()
            loss_mean += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # Skip the optimizer step on non-finite grads (inf/nan). Without this a
        # single spiky micro-batch permanently poisons the weights and every
        # subsequent step is NaN. Skipping keeps training recoverable.
        gn_finite = torch.isfinite(grad_norm).item() if hasattr(grad_norm, "item") \
            else math.isfinite(float(grad_norm))
        if gn_finite:
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)
            skipped_steps += 1
            log(f"[warn] step {step}: non-finite grad_norm, skipping optimizer step "
                f"(total skipped={skipped_steps})")
        if step % cfg.log_every == 0:
            dt = time.time() - t0
            tps = tokens_per_step * cfg.log_every / dt if step > 0 else 0
            gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
            mem = torch.cuda.max_memory_allocated() / 1e9
            # ETA from current throughput
            eta_s = (cfg.max_steps - step) * (dt / cfg.log_every) if step > 0 else 0
            eta_h = eta_s / 3600
            log(f"step {step:6d} | loss {loss_mean:.4f} | gnorm {gn:.3f} | "
                f"lr {lr:.2e} | {tps/1e3:.1f}K tok/s | mem {mem:.1f}G | eta {eta_h:.1f}h")
            if writer is not None and step > 0:
                writer.add_scalar("train/loss", loss_mean, step)
                writer.add_scalar("train/grad_norm", gn, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("perf/tokens_per_sec", tps, step)
                writer.add_scalar("perf/mem_gb", mem, step)
                writer.add_scalar("progress/tokens", step * tokens_per_step, step)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            val_loss = evaluate(model, val_loader, cfg.eval_batches)
            if val_loss is not None:
                log(f"step {step:6d} | val_loss {val_loss:.4f}")
                if writer is not None:
                    writer.add_scalar("val/loss", val_loss, step)
            t0 = time.time()   # don't count eval time in tok/s
        if step > 0 and step % cfg.ckpt_every == 0:
            save_ckpt(model, opt, step, cfg)
        step += 1

    save_ckpt(model, opt, step, cfg)
    if writer is not None:
        writer.close()
    log("[done] training complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
