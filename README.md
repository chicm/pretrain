# pretrain

Pretraining an 8B dense, decoder-only LLM (Llama-style) from scratch.

Current goal: get the full training pipeline running end-to-end with **FSDP2**
on the A100 cluster, then scale to MI300 + full data. For now we only care about
"it runs + the loss curve is healthy", not performance.

## Compute

| Cluster | Size | Use |
|---|---|---|
| A100 | 8 nodes × 4 GPUs | Data pipeline, 1B ablations, eval / post-training |
| MI300 | 4 nodes × 16 GPUs | Main production training |

The two clusters are used independently; we do not train across clusters.

## Layout

| Path | Contents |
|---|---|
| `src/` | Training code (model / data / train + launch scripts). See `src/README.md`. |
| `docs/` | Research notes and design decisions |

## Quick start

```bash
cd src
bash download_data.sh   # download data
bash run_smoke.sh       # single-node 4×A100 smoke test (tiny model + TinyStories)
```

For multi-node training, path conventions, and dependencies, see [`src/README.md`](src/README.md).

## Requirements

- Python, torch >= 2.4 (FSDP2 `fully_shard`)
- transformers, datasets, numpy
