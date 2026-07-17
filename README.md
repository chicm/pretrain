# pretrain

Pretraining an 8B dense, decoder-only LLM (**Chimera**, a Qwen3-style backbone) from scratch.

Current goal: get the full training pipeline running end-to-end with **FSDP2**
on the MI300 cluster (primary). A100 is single-node only. For now we care mostly about
"it runs + the loss curve is healthy", not peak performance.

## Compute

| Cluster | Size | Use |
|---|---|---|
| MI300 | up to 8 nodes × 8 GPUs (64 total) | Main production multi-node training (InfiniBand) |
| A100 | single-node only (no InfiniBand) | Data pipeline, 1B ablations, eval / post-training |

The A100 SKU (NC A100 v4) has **no InfiniBand**, so multi-node FSDP hangs after step 0;
all multi-node training runs on MI300. The two clusters are used independently; we do
not train across clusters.

## Layout

| Path | Contents |
|---|---|
| `src/` | Training code (model / data / train). See `src/README.md`. |
| `recipes/` | Git-tracked, experiment-specific launch recipes with exact runtime paths and arguments |
| `scripts/` | Generic, parameterized launch orchestration |
| `docs/` | Research notes, runbooks and design decisions |

## Quick start

```bash
cd src
bash download_data.sh   # download data
bash run_smoke.sh       # single-node smoke test (tiny model + TinyStories)
```

For production multi-node training, run a Git-tracked experiment recipe:

```bash
bash recipes/chimera_8b_1t.sh
```

The recipe records the exact topology, filesystem paths, source revision,
environment and training arguments. Generic orchestration lives in
`scripts/launch_multinode.sh` and contains no experiment-specific paths. Each
launch writes a timestamped manifest next to the checkpoint directory. See
[`docs/training_launch_reproducibility.md`](docs/training_launch_reproducibility.md).

The former root and `src/` multi-node launchers are deprecated and intentionally
refuse to start training, preventing stale configurations from being used.

## Requirements

- Python, torch >= 2.4 (FSDP2 `fully_shard`)
- transformers, datasets, numpy
