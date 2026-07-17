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
from datetime import timedelta
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


def _stack_collate(batch):
    """Stack (x, y) samples with torch.stack, bypassing torch's default
    collate_tensor_fn shared-memory 'out=' path (elem.new(storage).resize_),
    which raises 'Trying to resize storage that is not resizable' on int64
    tensors derived from numpy memmaps under multi-worker DataLoaders on this
    ROCm build. torch.stack allocates a fresh contiguous output -> safe."""
    xs = torch.stack([b[0] for b in batch], 0)
    ys = torch.stack([b[1] for b in batch], 0)
    return xs, ys



def is_master():
    return int(os.environ.get("RANK", 0)) == 0


def log(*a):
    if is_master():
        print(*a, flush=True)


def setup_dist():
    # Slow/cold blobfuse-backed shards can stall a rank's first read of a large
    # shard for many minutes; the default 10-min NCCL watchdog then aborts the
    # whole job on the next collective. Raise the collective timeout so a
    # one-time cold read can complete (steady-state cached reads are fast).
    dist.init_process_group(backend="nccl",
                            timeout=timedelta(minutes=60))
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
    tmp = path + ".tmp"
    torch.save({"model": model_sd, "opt": opt_sd, "step": step,
                "cfg": cfg.__dict__}, tmp)
    os.replace(tmp, path)   # atomic: a crash mid-write never leaves a half ckpt
    # write/refresh the "latest" pointer so --resume latest just works
    with open(os.path.join(cfg.out_dir, "latest"), "w") as f:
        f.write(f"ckpt_{step}.pt\n")
    log(f"[ckpt] saved {path}")
    _prune_ckpts(cfg)


def _prune_ckpts(cfg):
    """Keep only the most recent cfg.keep_last_ckpts checkpoints on disk.
    Each 8B full-state ckpt is huge (~180GB incl optimizer), so unbounded
    saves would fill the disk over a 238K-step run. Master-only."""
    keep = getattr(cfg, "keep_last_ckpts", 3)
    if keep <= 0:
        return
    import glob, re
    files = glob.glob(os.path.join(cfg.out_dir, "ckpt_*.pt"))
    def _step_of(p):
        m = re.search(r"ckpt_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    files = sorted(files, key=_step_of)
    for p in files[:-keep]:
        try:
            os.remove(p)
            log(f"[ckpt] pruned old {os.path.basename(p)}")
        except OSError as e:
            log(f"[ckpt] prune failed {p}: {e}")


def resolve_resume_path(spec, out_dir):
    """Map --resume value to a concrete ckpt path.
      * 'latest'         -> read <out_dir>/latest pointer
      * a bare filename  -> <out_dir>/<filename>
      * an explicit path -> used as-is
    Returns None if nothing to resume from (fresh start)."""
    if not spec:
        return None
    if spec == "latest":
        ptr = os.path.join(out_dir, "latest")
        if not os.path.exists(ptr):
            log(f"[ckpt] --resume latest but no '{ptr}' found -> starting fresh")
            return None
        name = open(ptr).read().strip()
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            log(f"[ckpt] latest points to missing {path} -> starting fresh")
            return None
        return path
    if os.path.dirname(spec):
        return spec
    return os.path.join(out_dir, spec)


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
    ap.add_argument("--data_root", default=None,
                    help="parent dir of per-source tokenized dirs (for --data_mix)")
    ap.add_argument("--data_mix", default=None,
                    help="named mix (mix_1t/smoke) or inline JSON of source->weight; "
                         "enables multi-source weighted sampling (overrides --data_dir)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tb_dir", default=None,
                    help="TensorBoard log dir override (default <out_dir>/tb).")
    ap.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None,
                    help="enable/disable TensorBoard event writing (default from TrainConfig).")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--micro_bsz", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--activation_checkpoint", action="store_true",
                    help="enable per-Block non-reentrant activation checkpointing (full recompute)")
    ap.add_argument("--selective_ac", action="store_true",
                    help="selective activation checkpointing: save expensive matmul/SDPA outputs, "
                         "recompute cheap elementwise/norm. Overrides --activation_checkpoint.")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from: 'latest' (auto from <out_dir>/latest), "
                         "a bare filename (ckpt_5000.pt), or an explicit path. Restores "
                         "model+opt+step and fast-forwards the data stream so no data is "
                         "re-trained.")
    ap.add_argument("--keep_last_ckpts", type=int, default=None,
                    help="keep only the N most recent ckpt_*.pt on disk (default cfg=3). "
                         "0 = keep all. Prevents disk blow-up over long runs.")
    ap.add_argument("--reduce_bf16", action="store_true",
                    help="use bf16 gradient reduce_dtype (default fp32); speed A/B, watch grad_norm")
    ap.add_argument("--hsdp_shard", type=int, default=0,
                    help="if >0, use HSDP 2D mesh: shard group size = this (e.g. 8 = intra-node), "
                         "replicate group = world/shard. 0 = full FSDP2 sharding (default)")
    ap.add_argument("--fused_ce", action=argparse.BooleanOptionalAction, default=True,
                    help="use flash-attn Triton fused cross-entropy (online softmax, no fp32 "
                         "logits materialization; ~14GB less peak mem at 8B vocab -> bigger micro_bsz). "
                         "DEFAULT ON; use --no-fused_ce to opt out.")
    ap.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True,
                    help="convert attn/MLP Linear layers to torchao float8 training "
                         "(MI300X: auto fnuz dtype). DEFAULT ON (validated +24.5%% tput, "
                         "-28GB mem on MI300X). REQUIRES compile (do NOT pass --no_compile). "
                         "Pass --no-fp8 to disable (falls back to bf16).")
    ap.add_argument("--fp8_recipe", default="tensorwise", choices=["tensorwise", "rowwise"],
                    help="fp8 scaling recipe: tensorwise (fastest ~1.5x) or rowwise (accurate ~1.4x)")
    args = ap.parse_args()

    cfg = TrainConfig()
    for k in ["model", "data_dir", "out_dir", "tb_dir", "tensorboard", "max_steps", "micro_bsz", "grad_accum"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.no_compile:
        cfg.compile = False
    if args.activation_checkpoint:
        cfg.activation_checkpoint = True
    if args.keep_last_ckpts is not None:
        cfg.keep_last_ckpts = args.keep_last_ckpts
    cfg.selective_ac = bool(getattr(args, "selective_ac", False))
    if cfg.selective_ac:
        cfg.activation_checkpoint = True  # selective implies AC path

    # Per-model training-stability overrides. Larger/deeper models need a
    # smaller peak LR and longer warmup to stay numerically stable in bf16.
    # 8B: sqrt(2) LR bump (2e-4 -> 2.8e-4) for the doubled global batch on
    # 64 GPUs (2.10M -> 4.19M tok/step), warmup 500 -> 1500 to match.
    _MODEL_OPT = {
        "8b": dict(lr=2.8e-4, min_lr=2.8e-5, warmup_steps=1500),
    }
    for k, v in _MODEL_OPT.get(cfg.model, {}).items():
        setattr(cfg, k, v)

    local_rank = setup_dist()
    world = dist.get_world_size()
    torch.manual_seed(cfg.seed)

    # --- resolve resume EARLY so the data loader can fast-forward past already
    # consumed instances (avoids re-training on the same data after a restart) ---
    resume_path = resolve_resume_path(args.resume, cfg.out_dir)
    resume_step = 0
    if resume_path:
        _meta = torch.load(resume_path, map_location="cpu", weights_only=False)
        resume_step = int(_meta["step"]) + 1   # next step to run
        del _meta
    # instances THIS replica consumed before the resume point
    per_step_replica = cfg.micro_bsz * cfg.grad_accum
    resume_skip = resume_step * per_step_replica

    # --- data ---
    margs = MODELS[cfg.model]()
    cfg.block_size = margs.max_seq_len
    if getattr(args, "data_mix", None):
        # multi-source weighted sampling (1T corpus)
        from data import EpochMixtureDataset, read_index
        from data_mix import resolve_mix
        data_root = args.data_root or cfg.data_dir
        sources = resolve_mix(args.data_mix, data_root)
        # derive vocab from first source's index
        first_idx = read_index(list(sources.keys())[0])
        if first_idx.get("vocab_size"):
            margs.vocab_size = max(margs.vocab_size, first_idx["vocab_size"])
        train_ds = EpochMixtureDataset(
            sources, cfg.block_size, seed=cfg.seed,
            rank=dist.get_rank(), world=world, resume_skip=resume_skip)
        loader = DataLoader(train_ds, batch_size=cfg.micro_bsz,
                            num_workers=4, pin_memory=True, drop_last=True,
                            collate_fn=_stack_collate)
        val_loader = None
        if resume_skip:
            log(f"[data] resume: fast-forwarding {resume_skip} instances/replica "
                f"(step {resume_step})")
        log(f"[data] multi-source mix '{args.data_mix}': "
            + ", ".join(f"{os.path.basename(k)}={v}" for k, v in sources.items()))
    else:
        meta = json.load(open(os.path.join(cfg.data_dir, "meta.json")))
        margs.vocab_size = max(margs.vocab_size, meta["vocab_size"])
        train_ds = PackedDataset(os.path.join(cfg.data_dir, "train.bin"),
                                 cfg.block_size, meta["dtype"])
        sampler = DistributedSampler(train_ds, num_replicas=world,
                                     rank=dist.get_rank(), shuffle=True)
        loader = DataLoader(train_ds, batch_size=cfg.micro_bsz, sampler=sampler,
                            num_workers=4, pin_memory=True, drop_last=True,
                            collate_fn=_stack_collate)

        # optional val set (written by prepare_data when val_frac > 0)
        val_loader = None
        val_path = os.path.join(cfg.data_dir, "val.bin")
        if os.path.exists(val_path):
            val_ds = PackedDataset(val_path, cfg.block_size, meta["dtype"])
            val_loader = DataLoader(val_ds, batch_size=cfg.micro_bsz, shuffle=False,
                                    num_workers=2, pin_memory=True, drop_last=True,
                                    collate_fn=_stack_collate)

    # --- model + FSDP2 ---
    model = Transformer(margs).to("cuda")
    model.fused_ce = bool(getattr(args, "fused_ce", True))
    if model.fused_ce:
        # DEFAULT ON now — probe flash-attn availability so a missing dep on any
        # node degrades to plain F.cross_entropy instead of crashing the run.
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss  # noqa: F401
            log("[ce] flash-attn Triton fused cross-entropy ON")
        except Exception as e:
            model.fused_ce = False
            log(f"[ce] fused CE unavailable ({type(e).__name__}: {e}) -> "
                "falling back to F.cross_entropy")
    # FP8: convert eligible Linear layers BEFORE fully_shard + compile.
    # fp8 is ON by default; if compile is disabled it would be ~2x slower than
    # bf16, so gracefully fall back to bf16 instead of erroring.
    if getattr(args, "fp8", False):
        if not cfg.compile:
            log("[fp8] compile disabled (--no_compile) -> fp8 auto-disabled, "
                "falling back to bf16 (eager fp8 is ~2x slower).")
        else:
            from fp8_utils import convert_model_to_fp8
            convert_model_to_fp8(model, recipe=args.fp8_recipe, log=log)
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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay, fused=True)

    # --- train loop ---
    # Restore into the uncompiled FSDP2 module.  Loading through an OptimizedModule
    # wrapper can replace/re-layout sharded parameter storage after compile setup;
    # this is a suspected cause of persistent throughput loss on resumed FP8 runs.
    # The optimizer must already exist for its state to be restored, while compile
    # remains lazy and belongs after all model/optimizer state restoration.
    model.train()
    step = 0
    skipped_steps = 0
    if resume_path:
        step = load_ckpt(model, opt, resume_path)
    if cfg.compile:
        model = torch.compile(model)

    # TensorBoard writer on master rank only (event files -> shared disk).
    # On resume, keep history below `step` and purge orphaned events at/after it;
    # the resumed run then rewrites those global steps cleanly.
    writer = None
    if is_master() and getattr(cfg, "tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = cfg.tb_dir or os.path.join(cfg.out_dir, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            purge_step = step if resume_path else None
            writer = SummaryWriter(log_dir=tb_dir, purge_step=purge_step)
            log(f"[tb] TensorBoard logging to {tb_dir}"
                + (f" (purge_step={purge_step})" if purge_step is not None else ""))
        except Exception as e:
            log(f"[tb] disabled ({e})")
    # Deterministic GC: disable automatic collection and instead run a light
    # gen-1 collect on ALL ranks at the same step. Prevents random full-GC on
    # one rank from stalling the whole world at the next all-gather (straggler).
    # (Borrowed from OLMo train.py.) gc_collect_interval steps between collects.
    import gc
    gc.collect()
    gc.disable()
    gc_interval = getattr(cfg, "gc_collect_interval", 1000)
    log(f"[gc] automatic GC disabled; manual gc.collect(1) every {gc_interval} steps")
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
        if step % gc_interval == 0:
            gc.collect(1)   # light gen-1 collect, synchronized across all ranks

    save_ckpt(model, opt, step, cfg)
    if writer is not None:
        writer.close()
    log("[done] training complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
