"""FSDP2 pretraining loop. Plain PyTorch (torch>=2.4) using torch.distributed.fsdp.fully_shard.

Launch with torchrun, e.g.:
  torchrun --nproc_per_node=4 train.py --model 1b --data_dir /path/tok

Multi-node is handled by run.sh (sets MASTER_ADDR / rendezvous).
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

from model import Transformer
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
    ap.add_argument("--resume", default=None, help="path to a ckpt_*.pt to resume from")
    args = ap.parse_args()

    cfg = TrainConfig()
    for k in ["model", "data_dir", "out_dir", "max_steps", "micro_bsz", "grad_accum"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.no_compile:
        cfg.compile = False

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
    log(f"[model] {cfg.model}: {model.num_params()/1e9:.3f}B params, "
        f"vocab={margs.vocab_size}, world={world}")
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
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
        opt.step()
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
