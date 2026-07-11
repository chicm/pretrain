# Pretraining Project — Progress & Ops Notes (2026-07)

Hands-on progress, infrastructure, architecture decisions, and gotchas from a from-scratch
7B–10B dense LLM pretraining effort. Research background and model choices are in
[`pretrain-research-2026.md`](./pretrain-research-2026.md).

> Note: cluster names, paths, and account identifiers have been genericized. This doc focuses
> on transferable engineering lessons, not any specific environment.

---

## 1. Goal & Technical Direction

- **Goal**: pretrain a 7B–10B dense LLM from scratch.
- **Framework**: **FSDP2** (`torch.distributed.fsdp.fully_shard`). Migrate to Megatron only for MoE / 30B+.
- **Model**: **Chimera** — dense decoder-only, Qwen3-style backbone (GQA + RoPE + RMSNorm + SwiGLU + **QK-Norm**), optional hybrid sliding/full attention (off for pretraining). No MoE. No soft-capping.
- **Configs** (`src/configs.py`): `tiny` (~50M core) / `1b` (~1.444B incl. embeddings) / `8b`.
- **Tokenizer**: **Qwen3** (`Qwen/Qwen3-8B`), vocab padded to **151936**; data packed as uint32.
- **Data**: TinyStories (smoke) → FineWeb sample-10BT (real validation) → later FineWeb-Edu / Nemotron-CC.

---

## 2. Two Clusters

| | Cluster A (NVIDIA) | Cluster B (AMD, primary) |
|---|---|---|
| Accelerator | 4 nodes × 4× A100-80GB = 16 GPU | 4 nodes × 8× MI300X-192GB = 32 GPU |
| VM series | NC A100 v4 | ND MI300X v5 |
| **InfiniBand** | ❌ **none** (NC series = Ethernet accelerated networking only) | ✅ **yes** (8× IB HCA, `mlx5_ib0-7`) |
| Multi-node training | ❌ not viable | ✅ viable (primary) |
| conda env | torch 2.7.1+cu126 | torch 2.8.0a0, ROCm 6.4.3 |
| libs | transformers 5.13, datasets 5.0 | transformers 4.55, datasets 4.0 |

### 2.1 The InfiniBand dead end on the NVIDIA cluster (key lesson)

- The A100 job used an **NC-series VM that has no InfiniBand in hardware** (NC series only
  provides Ethernet accelerated networking — confirmed by vendor docs).
- Symptom: multi-node FSDP 1B training — cross-node all-reduce defaulted to **0.68 GB/s**,
  **210 s per step**, hung after step 0.
- **NCCL socket tuning** (`NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=8 NCCL_BUFFSIZE=8388608`)
  raised the pure-bandwidth microbenchmark to **6.17 GB/s (9×)**, but **did not rescue training**:
  FSDP does dozens of small per-layer all-gathers per step and is **latency-bound**. Socket tuning
  lifts bandwidth but not TCP latency → training still hung.
- **Conclusion**: **the bottleneck for multi-node large-model training is InfiniBand; Ethernet is
  not a substitute.** The NC A100 job is **single-node only** (4× A100 over NVLink — verified loss
  convergence + checkpointing). Real multi-node A100 needs an **ND-series VM** (8× IB, 1.6 TB/s).
- Attempt to mount a faster NFS from another region failed: capacity for the MI300X VM was not
  available in that region. Root cause: the NFS datastore lived in a different region, which forced
  the compute to be scheduled there, where the accelerator had no capacity.
  **Lesson: compute and any mounted datastore must be in the same region.**

---

## 3. Primary (AMD MI300X) Cluster Environment

- **GPU**: MI300X, gfx942, **192 GB HBM3 each**, 8 per node, 32 total.
- **Software**: conda env with torch 2.8.0a0, HIP 6.4 (ROCm 6.4.3), RCCL 2.22.3,
  FSDP2 `fully_shard` OK.
- **InfiniBand**: 8× IB HCA (`mlx5_ib0-7`, link_layer=InfiniBand), verbs devices present, real RDMA.
- **Inter-node SSH**: a dedicated trust user (invoked via `su <trust-user> -c 'ssh node-N ...'`).

### 3.1 Storage tiering (important)

| Path type | Backing | Use for |
|---|---|---|
| Local scratch (fast NVMe/RAID, per-node) | local disk | code, HF download cache |
| Shared mount (blob-backed FUSE, cross-node) | object storage | data `train.bin`, checkpoints, logs |

**Measured shared-mount IO characteristics**:
- Large sequential **read 7.0 GB/s**, sequential **write 816 MB/s** → fine for reading data / writing ckpts.
- **Small files are very slow** (~63 files/s) → HF Xet chunked downloads and git-with-many-small-files stall.
- **Rule**: large files (data, ckpt) on the shared mount; many-small-files (code, HF cache) on local scratch.

---

## 4. Code Deployment Workflow (code-sync)

**git is the single source of truth; code lives on each node's local disk to avoid FUSE cache
consistency issues.**

```
1. Edit code locally (workstation repo)
2. git commit + push
3. Each node: git pull to LOCAL fast disk (strong consistency, no cache staleness)
4. Run torchrun from the local checkout
```

- **Data** (`train.bin`) lives on the shared mount (fast large reads, no consistency issue);
  **checkpoints** are written to the shared mount.
- **Do not** keep code on the shared FUSE mount: cross-node reads may see stale versions,
  read-after-write can be stale, and `.pyc` caches go stale.
- **CRLF fix**: repo ships a `.gitattributes` (`*.sh eol=lf`, `*.py eol=lf`). Editing on Windows
  still produces CRLF locally, but git push→pull converts to LF on Linux. No manual conversion needed.

### 4.1 Multi-node launcher

Key points:
- Loop over nodes; per node: `git config --global --add safe.directory <local-checkout>`
  (avoids "dubious ownership") → fetch/reset or clone → activate env → torchrun.
- `torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$i --rdzv_backend=c10d --rdzv_endpoint=node-0:<port>`.
- Print each node's `git rev-parse --short HEAD` to confirm all ranks run the same commit.
- Launch as the inter-node trust user so SSH between nodes works.

---

## 5. Verified Milestones

| Milestone | Result |
|---|---|
| Environment check | torch / FSDP2 / GPU / libs OK (both clusters) |
| Data tokenize | TinyStories → **473,992,236 tokens** (uint16, gpt2); identical across clusters (ROCm-compatible) |
| A100 single-node smoke | 4× A100, loss 10.96→2.4, ~445K tok/s, FSDP2+bf16 OK |
| Checkpoint bug fix | `get_model_state_dict(full_state_dict=True)` is collective; original code returned early on non-master ranks → deadlock. Fixed: all ranks participate, only master writes. |
| **MI300X single-node smoke** | 8× MI300X, **~1,880K tok/s** (4.2× the 4-GPU A100), loss converges |
| **MI300X multi-node 1B** | **32 GPU, 500 steps, loss 11.26→1.44, steady ~928K tok/s, 11.3 s/step** |
| **MI300X checkpointing** | ckpt at 200/400/500, 15.2 GB each, mid-run + end, collective logic correct; ~43 s to write 15 GB to shared mount |
| code-sync workflow | all nodes pull same commit, run from local disk, zero perf loss |

### A100 vs MI300 multi-node — the decisive comparison

| | A100 (no IB) | MI300X (with IB) |
|---|---|---|
| per step | 210 s (hung) | **11.3 s** |
| tok/s | ~0 | **928K** |
| loss | never started | smooth convergence |

→ **~19× faster and stable. Proves the bottleneck is InfiniBand.**

---

## 6. Gotchas Cheat-Sheet

### Training / code
- `train.py` has **no `--ckpt_every` CLI flag**: `ckpt_every` is a config attribute (`configs.py`).
  train.py unconditionally saves a checkpoint at the end; mid-run saves follow `cfg.ckpt_every`.
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is **unsupported on ROCm** (harmless warning); drop it.
- `data.py` per-sample Python fill loop has been **vectorized** (memmap + batched `np.concatenate`)
  to handle FineWeb-10BT (~10B tokens) efficiently.

### git / permissions
- **Dubious ownership**: git refuses to operate on a repo owned by another user → add `safe.directory`.
- If a node's local checkout ends up owned by root (created via a root-privileged path), the trust
  user can't fetch and gets stuck on a stale commit. Fix: remove it with privilege, then re-clone as
  the trust user. **All nodes' local checkouts must be owned by the same (trust) user.**

### HF downloads
- Set `HF_HOME` to **local disk** + `HF_HUB_DISABLE_XET=1`. Don't put it on the shared FUSE mount
  (Xet writes many small chunks to object storage and stalls — minutes for a few hundred MB).
- Use `hf download`, not the deprecated `huggingface-cli`.

### Remote command execution (Windows workstation → cluster)
- Nested quoting from a Windows local shell breaks easily (especially with `xargs`/multi-level quotes).
  Prefer putting commands in a repo `.sh` and running `bash <local-checkout>/xxx.sh` after pull.
- Filter non-printable bytes from remote output (progress bars) to avoid decode errors: append
  `| tr -cd '[:print:]\n'`.
- Don't kill training with `pkill -f <pattern>` (it self-matches the ssh session carrying that
  pattern). Use `pgrep -x torchrun | xargs -r kill -9` plus GPU compute PIDs from `rocm-smi`/`nvidia-smi`.

---

## 7. Next Steps

1. ✅ GPU utilization monitoring (`rocm-smi` across all GPUs; `_gpumon.sh`).
2. ✅ Tokenize FineWeb sample-10BT (~10.2B tokens, uint32, Qwen3) → real-data run launched.
3. ✅ Training observability: text metrics (`step|loss|gnorm|lr|tok/s|mem|eta`) + TensorBoard
   (event files mirrored to local disk to work around blobfuse append-read; viewed via SSH tunnel).
4. Try an 8B model multi-node (192 GB/GPU leaves ample headroom).
5. Scale data (FineWeb-Edu / Nemotron-CC) and lengthen the run.
