# pretrain

Pretraining an 8B dense, decoder-only LLM (**Chimera**, a Qwen3-style backbone) from scratch.

Current goal: get the full training pipeline running end-to-end with **FSDP2**
on the MI300 cluster (primary). A100 is single-node only. For now we care mostly about
"it runs + the loss curve is healthy", not peak performance.

## Compute

| Cluster | Size | Use |
|---|---|---|
| MI300 | 4 nodes × 8 GPUs (32 total) | Main production multi-node training (8× InfiniBand HCA) |
| A100 | single-node only (no InfiniBand) | Data pipeline, 1B ablations, eval / post-training |

The A100 SKU (NC A100 v4) has **no InfiniBand**, so multi-node FSDP hangs after step 0;
all multi-node training runs on MI300. The two clusters are used independently; we do
not train across clusters.

## Layout

| Path | Contents |
|---|---|
| `src/` | Training code (model / data / train + launch scripts). See `src/README.md`. |
| `docs/` | Research notes and design decisions |

## Quick start

```bash
cd src
bash download_data.sh   # download data
bash run_smoke.sh       # single-node smoke test (tiny model + TinyStories)
```

For multi-node training, path conventions, and dependencies, see [`src/README.md`](src/README.md).

## Requirements

- Python, torch >= 2.4 (FSDP2 `fully_shard`)
- transformers, datasets, numpy
