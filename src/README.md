# pretrain/src — FSDP2 预训练代码

从 0 预训练 7B–10B dense 模型的训练代码。核心：用 FSDP2 (`fully_shard`) 跑通多节点训练。

## 文件

| 文件 | 作用 |
|---|---|
| `model.py` | Llama 结构模型：GQA + RoPE + RMSNorm + SwiGLU，用 SDPA(flash) |
| `configs.py` | 模型预设（tiny/1b/8b）+ 训练超参 |
| `data.py` | tokenize 成 packed `.bin` + memmap 数据加载 |
| `tokenize_data.py` | 一次性预处理入口 |
| `train.py` | FSDP2 训练循环（`fully_shard`）+ cosine LR + checkpoint |
| `download_data.sh` | 下载 TinyStories + FineWeb-10BT 到数据目录 |
| `run_smoke.sh` | 单节点冒烟测试：tiny 模型 + TinyStories |
| `run_multinode.sh` | 单节点执行脚本（多机训练，需 NODE_RANK/MASTER_ADDR） |
| `launch_multinode.sh` | 从 node-0 分发到各节点启动多机训练 |

## 路径约定

用环境变量 `$WORKDIR` 指向工作目录。约定：
- **数据**（下载 + tokenize）放共享存储（跨节点可见），大文件顺序读快。
- **代码**放各节点本地盘（强一致，避免网络文件系统缓存问题）。
- **checkpoint** 写共享存储。

详见仓库根 `docs/progress-2026-07.md` 的存储分层与 code-sync 工作流。

## 环境

集群上以节点间 SSH 互信用户身份、在含 torch 的 conda 环境里运行。额外依赖：
```
pip install -U datasets huggingface_hub hf_transfer transformers
```

## 流程

```bash
# 1. 下载数据
bash download_data.sh

# 2. 冒烟测试（单节点，tiny 模型 + TinyStories）
bash run_smoke.sh

# 3. 1B 多机训练（多节点 + FineWeb-10BT）
#    先 tokenize：
python tokenize_data.py --dataset HuggingFaceFW/fineweb --hf_config sample-10BT \
    --split train --out "$WORKDIR/data/fineweb_tok"
#    再从 node-0 启动：
bash launch_multinode.sh
```

## 要求

- torch >= 2.4（FSDP2 `fully_shard` API）
- 依赖：transformers, datasets, numpy
