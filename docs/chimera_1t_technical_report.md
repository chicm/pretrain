# Chimera-8B 1T-Token 从零预训练技术报告

**项目**：Chimera 稠密大语言模型从零预训练
**规模**：7.602B 参数 dense decoder-only，998B（≈1T）token 训练预算
**硬件**：AMD MI300X（192GB HBM3），最终 15 节点 × 8 = 120 GPU
**框架**：PyTorch FSDP2（fully_shard）
**日期**：2026-07

> 本报告面向"从零预训练一个 7–10B 稠密模型"的完整工程实践，讲清楚我们的关键技术选择、实现细节、**踩过的所有坑**、有效与无效的优化，以及评测结果。目标不是写一份通用 guide，而是把这个真实项目做过、验证过、失败过的东西如实记录下来。

---

## 目录

1. 概览与关键结论
2. 模型结构与关键设计
3. 数据集设计（100BT → 1T 语料扩展）
4. 训练框架与并行策略
5. 数值精度：BF16 基线与 FP8 突破
6. 性能优化设计与结果（逐项技术细节，含无效项）
7. 训练稳定性：NaN 事故与修复
8. 检查点、断点续训与弹性扩缩容
9. 跨集群迁移（64 → 120 GPU）
10. 评测方法与结果（含 MMLU 涌现曲线）
11. 关键原理详解（面向首次做预训练的读者）：混合精度分层 / Flash Attention / 数据混合逻辑 / QK-Norm / Fused CE / NaN 根治
12. 工程踩坑总录
13. 经验总结

---

## 1. 概览与关键结论

| 维度 | 结论 |
|---|---|
| 框架 | 7–10B 稠密模型首选 **FSDP2**（原生 PyTorch、改架构方便、无 pipeline bubble）；MoE/30B+ 才考虑 Megatron |
| 精度 | BF16 做稳定基线；**FP8（仅大型 Linear）在 ROCm7.1 + torch.compile 下 8B 端到端 +25% 吞吐、省 28G 显存** |
| 最大单笔吞吐优化 | **Fused Cross-Entropy**（借鉴 OLMo），省 14GB logits 显存 → 打开更大 micro-batch，+33% over 原始基线 |
| FSDP2 通信优化 | 梯度累积期间**只在最后一个 micro 同步梯度 + 保留未分片参数**（A1+A2），+6.2% |
| 有效吞吐 | 64 GPU ~637K tok/s；迁移到 120 GPU 后 **~1.12M tok/s**（~9.3K tok/s/GPU） |
| 稳定性 | 深度缩放残差 init + non-finite 梯度跳步守卫，13000 步 8B 训练 0 真 NaN |
| 训练进度 | 1T 训练进行中，已过 ~350B token（35%）；MMLU 出现**涌现拐点**，从随机 0.25 升至 0.40 |

**一句话核心经验**：*8B 模型在 MI300X 上通常不是"参数装不下"，而是激活显存、FSDP 参数 all-gather 峰值、通信效率和 kernel 利用率成为瓶颈。优化要围绕"用富余 HBM 换通信"和"结构性解除显存+带宽双约束"来做。*

---

## 2. 模型结构与关键设计

### 2.1 骨干：Qwen3 风格稠密 decoder-only

Chimera 是一个 Qwen3-backbone 的稠密 decoder-only Transformer，配置见 `src/configs.py`（tiny / 1b / 8b）。核心设计：

| 设计点 | 选择 | 理由 |
|---|---|---|
| 归一化 | **RMSNorm，pre-norm** | 训练稳定、算子简单、可被 compile 融合 |
| 位置编码 | **RoPE**（split-half NeoX 约定） | 外推友好，与 Qwen3 对齐便于导出 |
| 注意力 | **GQA**（8B：32 Q-head / 8 KV-head） | 大幅降低 KV cache 和 KV 投影带宽 |
| **QK-Norm** | **ON** | Qwen3 关键稳定性设计，对大 LR 训练尤其重要 |
| FFN | **SwiGLU** | 标准现代选择 |
| bias | **全部无 bias** | 简化、稳定 |
| 词嵌入 | **tie_emb（输入输出共享）** | 省参数、省显存；8B vocab=151936 巨大，tie 收益明显 |
| 上下文 | 可配 hybrid sliding/full，**预训练全 full** seq=4096 | 简化 |

### 2.2 8B 具体规格（实际 7.602B）

```
dim         = 4096
n_layers    = 32
n_heads     = 32  (Q)
n_kv_heads  = 8   (GQA)
ffn hidden  = 14336 (SwiGLU)
seq_len     = 4096
vocab       = 151936 (padded, Qwen3)
tie_emb     = True
```

1B 版本（实际 1.444B，用于快速验证）：dim2048 / 24层 / 16Q-8KV / SwiGLU5632 / seq2048。

### 2.3 分词器

采用 **Qwen3 分词器**（`Qwen/Qwen3-8B`，vocab 151936 padded，eot id 151643）。仓库中只放 ~16MB 分词器文件，不放权重。选 Qwen3 而非自训 BPE 的原因：成熟、多语言/代码覆盖好、且便于把 checkpoint 导出成 HF Qwen3 格式复用生态。

### 2.4 关键设计：与 HF Qwen3 完全对齐，实现无损导出

我们刻意让 Chimera 的每个约定都与 HF Qwen3 对齐（QK-Norm、no-bias、tie-emb、RoPE split-half NeoX），这样 `export_hf.py` 可以做**纯映射、无需 permute**：

```
tok_emb        → embed_tokens
attn.wq/wk/wv/wo → self_attn.{q,k,v,o}_proj
attn.q_norm/k_norm → self_attn.{q,k}_norm
ffn.w1/w3/w2   → mlp.{gate,up,down}_proj
attn_norm/ffn_norm → input_layernorm/post_attention_layernorm
```

**验证结果**：导出后 logits 校验 argmax 一致率 **100%**，max|diff| 3.45e-2（bf16 噪声级）。一举两得——解锁 `lm-eval` 快速评测（原生 KV-cache，比朴素 generate 快 ~50×），并为未来 serving/发布铺路。

---

## 3. 数据集设计（100BT → 1T 语料扩展）

### 3.1 数据管线：预分词 + memmap

- `src/tokenize_data.py`：HuggingFace streaming → Qwen3 分词 → uint32 二进制。
- 存储：`train.bin`（memmap 预分配）+ `val.bin` + `meta.json`。
- 100BT 阶段实战：140 parquet（329GB）→ 9727 万文档 → 96 进程 tokenize（~70min）→ 写 398GB train.bin（99.66B token）。
- **memmap 顺序读不是瓶颈**：即便在 ROCm7.1 上用同步 DataLoader（`data_workers=0`），mmap 吞吐仍高于旧集群，读盘从来不是限制项。

### 3.2 从 100BT 到 1T：8 源混合语料

1T 训练需要远超 100BT 的语料。最终构建了 **8 源、约 938B 唯一 token** 的混合：

| 源 | token 量 | 混合占比 (mix_1t) |
|---|---:|---:|
| DCLM | 308B | 35.4% |
| FinePhrase | 192B | 20% |
| FineWeb-Edu | 174B | 18% |
| Code (StarCoder) | 134B | 15% |
| FinePDFs | 62B | 5% |
| Math (FineMath) | 33.2B | 3.12% |
| InfiMath | 21.6B | 2.16% |
| OpenWebMath (OWM) | 13.2B | 1.32% |

设计原则：以高质量网页（DCLM + FineWeb-Edu + FinePhrase = 73%）为主干，掺入 15% 代码、~8.6% 数学，兼顾通识、推理与代码能力。DCLM 以约 1.15 epoch 使用，其余源在 1T 预算内基本单 epoch。

### 3.3 EpochMixtureDataset

实现 `src/data_mix.py` 的 `EpochMixtureDataset` + `WeightedMultiSource`：按权重从各源采样，跨源做 epoch 级混合。踩坑（见 §11）：数据边界导致的短/空尾片必须显式跳过，否则 `torch.stack` 在某个 rank 上崩溃。

### 3.4 一个真实数据坑：`du` 的 awk %d 是 32 位

统计语料总量时，`awk '{printf %d}'` 在 >2^31 字节时**静默截断**，导致早期把语料量算错。正确做法用 `du -bc` 原始字节除以 4（uint32）。记录在案，提醒后人。

---

## 4. 训练框架与并行策略

### 4.1 为什么是 FSDP2

对 8B 稠密 + MI300X（192GB）这个组合，我们评估了三条路：

| 框架 | 结论 |
|---|---|
| **FSDP2** | ✅ 选定。原生 PyTorch、代码清晰、改架构方便、参数/梯度/optimizer state 全分片、与 DTensor/DeviceMesh 集成、适合模型研发 |
| ROCm Megatron-LM | 追求极限吞吐时值得对照，但改架构成本高，本项目未采用 |
| DeepSpeed | 8B + 192GB 场景下 CPU/NVMe offload 只会把瓶颈搬到 PCIe，不推荐 |

**核心判断**：8B 参数状态本身不是问题（BF16 完整副本仅 15.2GB），瓶颈在激活、all-gather 峰值、通信效率、kernel 利用率。FSDP2 的 1D full shard 足够，**不需要 TP/PP/CP**（避免 TP 高频通信、pipeline bubble）。

### 4.2 并行配置

- **1D FSDP full shard**，world size 64（旧）→ 120（新）。
- 逐层 `fully_shard(layer)` + 顶层 `fully_shard(model)`。
- **不启用** Tensor/Pipeline/Context Parallel。
- mixed precision policy：BF16 参数/激活/梯度通信，FP32 optimizer state。

### 4.3 全局 batch 与学习率

| 阶段 | 配置 | global batch |
|---|---|---|
| 8B 100BT run | mbsz2 / ga8 / world32 | 2.10M tok/step |
| 1T run（64 GPU） | mbsz4 / ga4 / world64 | 4.19M tok/step |
| 1T run（120 GPU） | mbsz4 / ga2 / world120 | 3.93M tok/step |

学习率 **2.8e-4**（min 2.8e-5，warmup 1500 步），相对 8B run 做了 √2 提升以匹配 2× batch。迁移到 120 GPU 后改用**基于 token 的 LR 调度**（而非基于 step），使 world size 变化下 LR 曲线保持一致（见 §9）。

---

## 5. 数值精度：BF16 基线与 FP8 突破

### 5.1 BF16 稳定基线

BF16 是 MI300X 上的首选基础精度：参数/激活/梯度通信 BF16，Adam 一二阶矩 FP32，敏感算子（softmax/norm 统计）FP32。BF16 指数范围接近 FP32，比 FP16 稳定且无需 loss scaling。**不使用 FP16。**

### 5.2 FP8 的关键突破：旧"FP8 无收益"是方法漏洞，不是硬件极限

这是本项目最有价值的精度发现之一。早期在 ROCm6.4.3 + torchao 0.13 上测 FP8 拿到 1.00×（无收益），当时归因于"ROCm6 软件栈不成熟"。**但那次测试有两个漏洞，正确组合从没测过：**

1. **fnuz dtype**：MI300 hipBLASLt 只支持 `float8_e4m3fnuz`/`e5m2fnuz`；OCP 的 `e4m3fn` 一律返回 `HIPBLAS_STATUS_NOT_SUPPORTED`（ROCm6/7 都一样，**不是版本差异**）。torchao 靠 `is_MI300()=True` 自动选 fnuz。
2. **torch.compile 强制**：eager 模式下 fp8 比 bf16 **慢 2×**（量化/scale 未融合）；旧测试是 eager 跑的 → 踩坑。

**在 ROCm7.1 + torch.compile 下重测的微基准（SwiGLU d4096 h14336, M=8×4096）：**

| 实现 | 时延 | 相对 bf16 |
|---|---:|---:|
| bf16 | 37.7ms | 1.00× |
| fp8 tensorwise **eager** | 83.7ms | 0.45×（慢！） |
| **fp8 tensorwise + compile** | **24.2ms** | **1.56× ✅** |
| fp8 rowwise + compile | 27.2ms | 1.39× |

**8B 端到端 A/B（mbsz=4，真实数据，--fused_ce）：**

- bf16：67.5K tok/s / 118.0G
- **fp8：84.4K tok/s / 90.0G → +25% 吞吐，−28G 显存**，loss 逐步重合无 NaN，224/225 Linear 转换。

### 5.3 FP8 实现细节

`src/fp8_utils.py`：`convert_model_to_fp8(model, recipe='tensorwise')`：

- 在 `fully_shard`/`compile` **之前**调用。
- 只转 attention/MLP 的大型 Linear（Q/K/V/O、gate/up/down），**跳过 lm_head**。
- in/out 维度需 16 对齐。
- RMSNorm/Softmax/RoPE/loss/optimizer 保持 BF16/FP32。
- `--fp8` 强制启用 compile（否则 eager 会慢 2×，直接 SystemExit 拒绝）。

**关键坑**：正确性验证必须比对 train loss / val loss / grad norm / NaN 检查 / checkpoint 续训连续性，确认无数值偏移后再进正式长训。FP8 理论 GEMM 吞吐高，但端到端收益受 attention/通信/数据/optimizer 环节稀释，实际收益必须实测。

---

## 6. 性能优化设计与结果

本节是全报告的技术核心。方法论贯穿始终：**先微基准探路，确认收益和显存后再接入训练代码；每次只改一个变量；以稳态端到端吞吐为准（排除首次 compile/checkpoint/warmup）。**

### 6.1 【最大单笔收益】Fused Cross-Entropy（+33% over 原始基线）

**动机**：8B vocab=151936 极大，vanilla `F.cross_entropy(logits.float())` 会物化 `[M×151936]` 的 FP32 logits（~7.5GB）+ softmax，成为显存和带宽双瓶颈。借鉴 OLMo 的 flash-attn Triton CE kernel。

**微基准 `_ce_probe.py`（单卡）：**

| 实现 | mbsz3 时延/显存 | mbsz4 |
|---|---|---|
| vanilla F.cross_entropy | 115ms / 22.1GB | 137ms / 29.1GB |
| **flash-triton CE** | **94ms(−18%) / 8.2GB(省13.9G)** | 107ms / 10.6GB |

flash-attn Triton CE 在 ROCm/MI300 上**能用**（不像早期 FP8 撞 hipBLASLt）！省 ~14GB + 快 18%。

**关键坑**：`torch.compile` 会破坏 flash-attn 手写 Triton CE kernel → **loss=210（垃圾值）**！修复：用 `@torch._dynamo.disable` 包住 `_fused_ce_loss`，让 compile 只编译 transformer 主体、绕过 CE。

**A/B 结果（80 步 smoke）：**

| 配置 | tok/s | vs 189K 基线 | mem |
|---|---|---|---|
| 原始基线 mbsz=2 | 189K | — | 89G |
| 旧最优 mbsz=3+compile | 202K | +6.9% | 131G |
| **fused_ce mbsz=4 +compile(修复后)** | **250.8K** | **+33%** | 106.5G |

**为何比之前七个 1–7% 的优化都大**：fused CE **同时解除了显存约束（可开 mbsz=4 大 GEMM）和带宽约束（不物化 fp32 logits）**，是结构性突破。三重协同：CE 省显存 + mbsz4 大 GEMM + compile 可安全叠加。

### 6.2 【FSDP2 通信】梯度累积期只在最后 micro 同步（A1+A2，+6.2%）

**问题**：原训练循环每个 micro-batch 都触发跨 rank 梯度同步，而真正需要同步的只有最后一个 micro。

**A1 — 只在最后 micro 同步梯度**（`model.set_requires_gradient_sync(is_last_micro)`）：
- 结果 623.25K tok/s，**+3.88%**，HBM 106.6G，loss 匹配。

**A2 — 累积早期 micro 后不 reshard 参数**（`model.set_reshard_after_backward(is_last_micro)`，保留 unsharded 参数，下次 forward 不必重新 all-gather）：
- 结果 637.15K tok/s，**+6.19%**（在 A1 基础上再 +2.3%），HBM 106.5G，loss/gnorm 健康。

**A1+A2 = 生产采纳配置**（`--fsdp_sync_last_micro --fsdp_reshard_last_micro`）。这正是"用富余 HBM 换通信"原则的直接兑现——每卡 115GB 余量足以保留完整参数副本。

### 6.3 无效 / 拒绝的优化（同样重要）

**这些是我们真金白银试过、但在本栈上无收益或不可用的，列出来避免后人重走：**

| 优化 | 结果 | 结论 |
|---|---|---|
| **A3 `reshard_after_forward=8`** | step0 后 HBM 飙到近 200GB/卡 → RCCL accept 失败/double-free | ❌ 拒绝。`False` 更激进直接跳过 |
| **FP8 FSDP all-gather** | 574K tok/s（HBM 112G），明显**回退** | ❌ 拒绝 |
| **BF16 gradient reduce** | 640.4K，仅比 A2 高 +0.51%，低于 1% 噪声阈值，且数值略有差异 | ⚠️ 不采纳，保留 default off |
| **hipBLASLt / TunableOp 离线 GEMM 调优** | torch2.8 ROCm 上 env 文件名保持空/懒初始化，record 从没落盘；在线调优 v8–v10 单个 op 卡 20–25min 不出步 | ❌ 本栈操作上不可用 |
| **AITER FlashAttention/RMSNorm/RoPE/SwiGLU** | AITER 0.1.3 JIT 报 build 成功但 import 失败（`No module named module_mha_fwd`） | ❌ 环境阻塞 |
| **flash-attn 2.8.3 native-GQA 后端** | 试跑 20+min 卡在 step0 前，GPU 活跃但不出步 | ❌ 不可用 |
| **`torch.compile` max-autotune** | 记录 103 次 scaled-mm autotune，到 step0 后持续 autotune 20+min 无 step1 | ❌ 操作上不适用 |

**教训**：现有 RMSNorm/RoPE/SwiGLU 表达式已在 `torch.compile` 融合内，重复手写 kernel 收益有限且环境风险高。ROCm 上很多"理论最优"的 kernel/调优路径在实际软件栈上并不成熟，必须逐一实测。

### 6.4 确定性 GC（借鉴 OLMo，降 straggler 抖动）

循环前 `gc.disable()`，每 `gc_collect_interval`（默认 1000）步显式 `gc.collect(1)`。消除随机 full-GC 造成的单卡卡顿 → 全体 all-gather straggler。world=32 时对此敏感，收益体现在吞吐方差降低。

### 6.5 micro-batch 探边

8B：mbsz=2 峰值 ~89–97G（甜点）；mbsz=4 配 fused_ce 后峰值仅 106.5G（fused CE 省出来的空间）；mbsz=6 时 bf16 峰值 160.2G 逼近上限且 compile 病态卡死（非 fp8 特有），不实用。**结论：fused_ce + mbsz=4 是最佳工作点。**

---

## 7. 训练稳定性：NaN 事故与修复

### 7.1 8B 训练 step 110 NaN 事故

8B run 在 **step 110**（warmup 中 lr≈1.66e-4）出现 NaN，0–100 步健康（loss 12.73→6.30）。

**根因 = 模型 init 缺残差深度缩放**：output projection（`wo`/`w2`）未按深度缩放，32 层残差方差累积爆炸。

**三个共同修复（缺一不可）：**

1. **深度缩放残差 init（根因）**：对 `_residual_proj` 标记的 Linear 施加 `std = 0.02/√(2·n_layers)`。
2. **non-finite 梯度跳步守卫（决定性）**：`torch.isfinite(grad_norm)` 否则 `zero_grad` + `skipped_steps++`。关键洞察：grad_clip 只处理 inf（1/inf=0），**不处理"有限但巨大"的毒梯度**——这类梯度能通过 clip 却毒害 optimizer。
3. **保守 lr/warmup**：lr 3e-4→2e-4，warmup 200→500，降低尖峰频率。

**修复后**：loss 12.7→2.35，grad_norm 0.04–0.06，13000 步全跑完 **0 真 NaN**，累计 skip 455/13000 步（守卫兜底）。事故报告：`docs/incident_8b_loss_nan.md`。

### 7.2 1T 训练的稳定性

1T run 继承全部修复。skip rate ~1%（20 非有限/2120 步），warmup→peak LR 后**未恶化**，故 LR 2.8e-4 保持（阈值是：skip >5% 才降到 2.4e-4）。gnorm 长期 0.05–0.25，非常稳定。

---

## 8. 检查点、断点续训与弹性扩缩容

### 8.1 检查点方案

- 每 2000 步保存，`.pt` ~87–93GB/个，`keep_last 3` 轮换 + 保留最早一个。
- **原子写 + latest 指针**：写临时文件 → 原子 rename → 更新 `latest` 指针，避免半写崩溃。
- `--resume latest` 自动续训。

### 8.2 数据快进续训

续训不只恢复模型/optimizer，还要恢复**数据位置**。实现 `resume_skip`：按 consumed_tokens 快进 DataLoader，保证续训不重复/不跳过数据。

### 8.3 弹性 world-size 续训（关键设计）

为支持 64→120 GPU 迁移，实现了 **world-size 无关的续训深度** + **基于 token 的 LR 调度**：

- checkpoint 存 `consumed_tokens` 元数据，而非 step 数。
- LR 由 consumed_tokens 计算，world size / global batch 变化时曲线连续。
- `resume_skip_for_rank` 按新 world size 重新分配数据跳过量。

`consumed_tokens(step)` 迁移后公式：`75,501,666,304 + (step−18000)×3,932,160`（18000 前用旧的 4,194,304/step）。

---

## 9. 跨集群迁移（64 → 120 GPU）

1T 训练从 `chec-mi300-3`（8 节点 64 GPU，ROCm6.4.3，conda）迁移到 `chec-mi300-4`（15 节点 120 GPU，ROCm7.1，`/opt/venv`）。从 ckpt_18000 续训。

**迁移中暴露并修复的 3 个真实 bug：**

1. **NCCL init 前必须先绑定本地 GPU**（`set_device`）——否则 120-GPU NCCL init 挂死。这是 64 GPU 时侥幸没触发、规模上来才暴露的经典坑。
2. **`import gc` 提到模块顶层**——原来在函数内 import，某路径下 NameError。
3. **ROCm7.1 DataLoader fork 死锁**：`num_workers>0` 在 ROCm7.1 上 fork 死锁 → 改 `--data_workers 0` 同步 DataLoader。前面说过，同步 mmap 读不是瓶颈，吞吐反而更高。

**迁移结果**：120 GPU 稳态 **~1.12M tok/s**（~9.3–9.4K tok/s/GPU，120/120 ~100% util，mem 106.1G），loss/gnorm 连续，无 NaN/OOM。5 次 smoke 端到端验证通过后启动正式续训（rendezvous `chimera1t-resume120-ec03e70`）。

---

## 10. 评测方法与结果

### 10.1 评测基础设施

- 环境：`chec-test-env` 单 MI300X，`/opt/venv/bin/python`（ROCm torch）+ overlay，**绝不新建 pip venv**（会拉 CUDA torch 退化成 CPU-only）。
- 工具：`lm-eval` zero-shot，利用 HF Qwen3 导出 + 原生 KV-cache（比朴素 generate 快 ~50×：HumanEval 164 题 11 分钟 vs 9.7 小时）。
- **结构化输出布局**（避免早期 `_rocm7`/`_parallel`/`_v6` 目录混乱）：
  - 一个 run = 一棵树（run_id 顶层）。
  - step 用 8 位零填充（字典序=数字序）。
  - 元数据放 `config.json` 而非目录名。
  - `summary.csv` 长格式 append（新增 benchmark 不改表头）。
  - 每次评测后重新生成 `summary.md`。
- **自动评测**：schedule 定时任务每 30 分钟检查新 checkpoint，只评最新未评的，去重、算 consumed_tokens、验证 GPU、结构化写入。

### 10.2 下游能力进展（zero-shot，节选节点）

| Benchmark | 2000 (8.4B) | 14000 (58.7B) | 34000 (141B) | 66000 (264B) | 82000 (327B) | 90000 (350B) |
|---|---:|---:|---:|---:|---:|---:|
| HellaSwag acc_norm | 0.293 | 0.615 | 0.673 | 0.711 | 0.712 | 0.712 |
| LAMBADA acc | 0.213 | 0.574 | 0.613 | 0.641 | 0.667 | 0.650 |
| ARC-Easy acc | 0.386 | 0.721 | 0.770 | 0.770 | 0.785 | 0.775 |
| ARC-Challenge acc_norm | 0.209 | 0.392 | 0.433 | 0.476 | 0.493 | 0.488 |
| PIQA acc_norm | — | 0.748 | 0.769 | 0.781 | 0.774 | 0.783 |
| Winogrande acc | — | 0.598 | 0.635 | 0.652 | 0.675 | 0.673 |
| MMLU acc | — | 0.260 | 0.268 | 0.305 | 0.384 | 0.391 |

### 10.3 关键发现：MMLU 涌现拐点

MMLU 在 141B token 前一直贴随机（0.25–0.27），随后**急速涌现**：

```
34000 (141B): 0.268  贴随机
60000 (241B): 0.300  首次脱离随机
66000 (264B): 0.305  拐点启动
70000 (280B): 0.358  突破 0.35
82000 (327B): 0.384
88000 (342B): 0.405  首次突破 0.40
90000 (350B): 0.391  单点噪声回落
```

这是知识密集型任务的典型涌现曲线——**一个仅训练 36%、7.6B 的 dense 模型 MMLU 达 ~0.40 是扎实的里程碑**。其余能力（HellaSwag 0.71、PIQA 0.78、ARC-Challenge 0.49、Winogrande 0.67）在各自高位稳定。

### 10.4 评测方法论经验

- **下游 benchmark 比训练 loss 更敏感、更可信**：多域 batch loss 噪声大（σ≈0.22），下游指标单调性更清晰。
- MMLU 近随机（0.25–0.27）在数百 B token 前是**预期的**，不是问题信号。
- 单点噪声（ARC-Easy σ≈±0.9%）会造成个别指标"抖动"，看趋势不看单点。

---

## 11. 关键原理详解（面向首次做预训练的读者）

> 假设你熟悉 Transformer、做过 finetune，但没做过 from-scratch 预训练。本节把几个最容易混淆、也最影响成败的机制讲透。

### 11.1 混合精度：哪些层 FP8、哪些 BF16、哪些 FP32，以及为什么

预训练不是"全模型一个精度"，而是**分算子按数值敏感度选精度**。Chimera 的精度分布如下：

| 组件 | 计算精度 | 存储精度 | 为什么 |
|---|---|---|---|
| **大型 Linear 的 GEMM**（Q/K/V/O、gate/up/down） | **FP8 (e4m3fnuz)** | 权重主副本仍 BF16 | 这些是**算力大头**（占绝大多数 FLOPs）。矩阵乘对量化相对鲁棒；MI300 FP8 GEMM 吞吐~2×。输入在 kernel 内动态量化到 FP8，累加仍在高精度 |
| **lm_head / 词嵌入** | **BF16（不转 FP8）** | BF16 | 词表 151936 极大，输出层直接决定 loss；量化误差在这里会放大成系统性偏差。宁可不省这一层 |
| **注意力 softmax、RMSNorm 统计、RoPE** | **FP32（内部累加）** | — | 这些是**归约/非线性**，对精度极敏感：softmax 的 exp、norm 的方差、RoPE 的三角函数，FP8/BF16 会累积误差导致训练发散 |
| **激活、残差流、all-gather 通信** | **BF16** | BF16 | BF16 指数范围≈FP32，不需 loss scaling；带宽减半 |
| **loss 归约（cross-entropy）** | **FP32** | — | loss 求和跨越 151936 词表 × 数百万 token，必须 FP32 累加否则精度丢失 |
| **优化器状态**（Adam 一/二阶矩） | **FP32** | FP32 | 这是**训练稳定性的地基**。二阶矩 `v` 长期累积，BF16 会让小梯度被吃掉，等效学习率漂移。**这一条几乎没有例外** |
| **梯度通信（reduce-scatter）** | **BF16**（默认） | — | 我们测过 FP32 reduce 收益 <0.51%，不值得，保持 BF16 |

**设计心法**：*算力大头用最激进精度（FP8），归约/非线性/优化器状态用最保守精度（FP32），其余走 BF16。* 这不是拍脑袋，而是"哪里的误差会被放大就在哪里保精度"——softmax/norm/loss/优化器矩是误差放大器，GEMM 是误差稀释器。

一个反直觉但重要的点：**FP8 只作用于前向/反向的矩阵乘 kernel 内部**，权重的"主副本"（master weight）始终是 BF16，优化器更新的也是 BF16 权重。FP8 只是"这次 GEMM 临时用低精度算得更快"，不是"把模型变成 FP8 模型"。所以数值稳定性由 BF16 主权重 + FP32 优化器矩共同保证。

### 11.2 Flash Attention：用了吗？它到底怎么起作用

**用了，但用的是 PyTorch 原生 SDPA（`scaled_dot_product_attention`）的 flash 后端**，不是独立的 `flash-attn` 包（后者在本 ROCm 栈上 JIT 不可用，见 §6.3）。

**Flash Attention 解决的核心问题**：朴素注意力要物化 `S = QKᵀ`，形状 `[batch, heads, seq, seq]`。seq=4096 时，单头单样本的注意力矩阵就是 4096×4096≈1600 万个数，全部头全部样本会占**巨量显存**，而且要反复读写这个大矩阵（HBM 带宽瓶颈）。

Flash Attention 的做法是**分块（tiling）+ 在线 softmax**：

1. 把 Q、K、V 切成小块，逐块计算，**永不物化完整的 `[seq, seq]` 注意力矩阵**。
2. 用"在线 softmax"技巧（维护 running max 和 running sum）增量地累积输出，数学上等价于完整 softmax。
3. 整个注意力在 GPU 的 SRAM/寄存器里流式完成，只把最终输出写回 HBM。

**收益**：显存从 O(seq²) 降到 O(seq)，且大幅减少 HBM 读写 → 又快又省。对 seq=4096 的预训练这是**必备项**，没有它长上下文根本训不动。反向传播同理，用重计算避免存储中间注意力矩阵。GQA（8 KV-head）进一步减少 K/V 的读取量。

### 11.3 数据加载与混合：一条一条，还是按 block？

这是预训练特有、finetune 很少遇到的问题。答案：**在"打包成定长序列"这一层是按 token 流拼接的，在"多源混合"这一层是按样本（序列）加权采样的。**

分两步理解：

**第一步：单源内——token 流打包（packing）。**
预训练数据是海量文档，长短不一。我们不做 padding（浪费算力），而是把所有文档**分词后首尾相接成一条超长 token 流**，再每 `seq_len+1`（4097）切一刀，得到一个训练样本（input 4096 + target 4096，错一位）。所以**一条训练序列里可能包含多个文档的片段**，文档间用 eot token（151643）分隔。这叫 sequence packing，是预训练吞吐的关键——GPU 永远在算真实 token，没有 padding 浪费。

**第二步：多源之间——按权重采样序列（EpochMixtureDataset）。**
8 个源（DCLM 35.4%、FinePhrase 20%…）的混合发生在**序列级别**：`WeightedMultiSource` 按配比权重决定"下一条序列从哪个源取"。所以：

- **不是**把不同源的 token 在一条序列里逐 token 混合。
- **而是**一条序列完整来自某一个源（内部是该源多个文档 packing 的结果），一个 batch 里的不同序列可能来自不同源，长期看各源的 token 占比收敛到设定权重。

用图示表达一个 global batch（比如 960 条序列）：

```
seq 0:   [DCLM 文档片段 | eot | DCLM 文档 | eot | ...]  ← 整条来自 DCLM
seq 1:   [Code 文件 | eot | Code 文件 | ...]            ← 整条来自 StarCoder
seq 2:   [FineWeb-Edu ... ]
seq 3:   [Math ... ]
...
→ 全 batch 里约 35% 的序列来自 DCLM，15% 来自 Code，以此类推
```

**为什么这样设计**：序列级混合保持了每条序列内部的语义连贯性（不会把代码和网页硬塞进同一条 4096），同时通过大量序列的加权采样在统计上精确控制各源占比。跨源 epoch 混合还保证了数据顺序的充分打散。

**踩过的坑**（§11 会再提）：某个源的 memmap 尾部索引长度 > 实际写入长度时，最后一片会返回短/空数组，`torch.stack` 因形状不一崩溃 → yield 前必须 skip `shape[0]!=bs` 的片。

### 11.4 QK-Norm：原理与为什么它对大 LR 训练关键

**QK-Norm = 在计算注意力分数之前，对 Query 和 Key 向量各做一次 RMSNorm（按 head 维度归一化）。**

标准注意力：`score = QKᵀ / √d`。问题在于：训练中 Q、K 的**幅值（norm）会不受控地增长**，导致 `QKᵀ` 的数值越来越大，softmax 进入饱和区（一个位置接近 1、其余接近 0），梯度消失、注意力"锁死"，这是大规模、大学习率训练发散的常见诱因（尤其在 warmup 后 LR 冲到峰值时）。

QK-Norm 的做法：

```
q = rms_norm(q)   # 沿 head_dim 归一化，可学习缩放
k = rms_norm(k)
score = (q · kᵀ) * scale
```

归一化后 Q、K 的幅值被钉在可控范围，`QKᵀ` 不会爆炸，softmax 稳定工作。**代价极小**（两个 RMSNorm），**收益是显著提升训练稳定性上限**，让我们能用 2.8e-4 这种较激进的 LR。这是 Qwen3 相对早期架构的关键改进之一，我们完整继承。可以把它理解成"给注意力打分器做了个自动增益控制"。

### 11.5 Fused Cross-Entropy：原理与为什么省这么多

先看 **vanilla cross-entropy 在大词表下为什么是灾难**。语言模型最后一步：

```
logits = hidden @ lm_head.T   # [M, vocab] = [M, 151936]
loss = F.cross_entropy(logits.float(), targets)
```

`M` 是一个 batch 的总 token 数（micro-batch × 4096，轻松几万）。`logits` 是 `[M, 151936]`，转 FP32 后是 **~7.5GB**，softmax 又要一份同样大的中间张量。这块显存**纯粹是为了算一个标量 loss**，用完就扔，却挤占了本可用于更大 batch 的空间，还要在 HBM 里来回读写（带宽瓶颈）。

**Fused CE 的原理**：把"logits 计算 + log-softmax + gather 目标位置 + 求 loss"**融合进一个 Triton kernel**，**永不物化完整的 `[M, vocab]` FP32 logits**。它分块计算，在 kernel 内部流式地：

1. 逐块算出 logits 的一小片；
2. 在线累积 log-sum-exp（类似 flash attention 的在线 softmax）；
3. 只取出目标 token 对应的 logit；
4. 直接输出 per-token loss。

中间的巨大 logits 矩阵**从不落 HBM**，反向也在 kernel 内直接算梯度。

**实测收益**（§6.1）：省 ~14GB 显存 + 快 18%。而省下的 14GB 让我们把 micro-batch 从 2 开到 4（更大的 GEMM 更高效），三重协同最终 **+33%** 吞吐。**这就是为什么 fused CE 是全项目最大单笔优化**——它同时解除了显存和带宽两个约束。

**关键坑**：`torch.compile` 会破坏这个手写 Triton kernel（loss 变成 210 垃圾值），必须用 `@torch._dynamo.disable` 把 fused CE 单独隔离出编译图。

### 11.6 NaN / Inf loss：为什么会发生，怎么根治

这是预训练新手最容易被"劝退"的一关。finetune 很少遇到，因为预训练用大 LR、从随机初始化开始、跑几十万步，任何微小的数值不稳都会被放大成 NaN。

**NaN 的产生链条**（我们真实踩过的 step 110 事故）：

1. **根因：初始化没做残差深度缩放。** 每层的残差连接 `x = x + sublayer(x)` 会让激活方差**逐层累积**。32 层叠加后，如果输出投影（`wo`、`w2`）用标准 init，深层的激活/梯度方差会指数放大。
2. warmup 中 LR 爬升到 ~1.66e-4 时，某一步产生一个**巨大但仍有限**的梯度。
3. 这个梯度更新权重后，下一步前向就溢出成 inf，再一步 loss=NaN，之后全崩。

**为什么 gradient clipping 挡不住**：clip 只处理 `inf`（`1/inf=0` 自然被裁掉），但对"**有限但巨大**"的毒梯度无能为力——它能通过 clip 的范数检查，却足以毒害 Adam 的二阶矩，污染后续所有更新。

**三个共同修复（缺一不可）：**

1. **深度缩放残差 init（治本）**：对残差路径的输出投影用 `std = 0.02/√(2·n_layers)` 初始化。这样无论多少层，残差流的方差都保持稳定，从源头杜绝方差爆炸。
2. **non-finite 梯度跳步守卫（兜底）**：每步算完 `grad_norm` 后 `if not torch.isfinite(grad_norm): zero_grad(); skipped_steps++; continue`。这一步是**决定性的安全网**——即使偶发毒梯度，直接丢弃这一步而非让它进优化器。这也补上了 grad_clip 挡不住"有限巨大梯度"的漏洞。
3. **保守 LR/warmup**：适度降峰值 LR、拉长 warmup，降低尖峰频率。

**效果**：修复后 8B 13000 步全跑完 **0 真 NaN**，累计 skip 仅 455 步（守卫兜底，占 3.5%）。1T run 继承全部修复，skip rate ~1%，warmup→peak 后未恶化，故 LR 保持 2.8e-4。

**给新手的经验**：预训练一定要**默认装好 non-finite 守卫**，把它当成安全带而不是可选项；同时**从第一天就把 init 做对**（残差缩放、QK-Norm），治本比事后调 LR 有效得多。

---

## 12. 工程踩坑总录

汇总散落在各阶段的高价值坑，多数已固化到记忆/脚本约定：

### 分布式 / 框架
- **NCCL init 前必须 `set_device`**，否则大规模 init 挂死。
- **梯度累积默认每 micro 都同步**，需显式 `set_requires_gradient_sync(False)`。
- **`reshard_after_forward=8`** 在本模型上显存爆炸 → RCCL 崩溃。
- **ROCm7.1 DataLoader `num_workers>0` fork 死锁** → 用 `data_workers=0`。

### 精度 / kernel
- **MI300 FP8 只支持 fnuz dtype**，OCP e4m3fn 一律 hipBLASLt NOT_SUPPORTED。
- **FP8 必须配 torch.compile**，eager 慢 2×。
- **torch.compile 破坏 flash-attn Triton CE** → loss=210 → 用 `@torch._dynamo.disable`。
- **AITER/TunableOp/max-autotune/flash-attn 后端**在本 ROCm 栈上均不可用（见 §6.3）。

### 数据
- **短/空尾片**导致 `torch.stack` 崩溃 → yield 前 skip `shape[0]!=bs`。
- **collate resize-storage crash** → 用模块级 `_stack_collate`（`.copy()` 不够）。
- **`awk %d` 32 位截断**大文件字节数 → 用 `du -bc`。

### 稳定性
- **深度缩放残差 init** 是防 NaN 根因修复。
- **non-finite 梯度跳步守卫**处理 grad_clip 漏掉的"有限但巨大"毒梯度。

### 运维（本地 Windows + 远程）
- **CRLF 危害**：push 的 `.sh` 带 `\r` → `bash: cd $'/scratch\r'`。**销毁式排序 bug**：`open(p,'wb').write(open(p,'rb').read())` 会先截零文件——必须先读完再写。
- **嵌套引号地狱**：ssh 上 Python `-c` 带 import/`;`/heredoc、`for`/`$()` 循环都失败 → 写零嵌套引号的参数化脚本文件再 push+run。
- **run_remote.py ~30s 超时**：长任务用 `setsid bash ... >log 2>&1 </dev/null &` 自脱离。
- **日志缓冲假"卡死"**：blobfuse2 + Python 全缓冲 → 日志久久 0 字节，别误判卡死；用 `PYTHONUNBUFFERED=1` 直写 logfile。
- **本地读远程日志 UnicodeDecodeError**（tqdm 进度条字节）→ `tr -cd '[:print:]\n'` 过滤。
- **绝不用 GPU keepalive**；启动后立即用 rocm-smi util+VRAM 验证 GPU 真在用。

---

## 13. 经验总结

1. **框架选择**：7–10B 稠密 + 大显存 GPU，FSDP2 是甜点。不要过早引入 TP/PP/CP，1D full shard 足矣。

2. **优化方法论**：**先微基准探路，再接入训练**。`_ce_probe.py` 预判省 14G 能开 mbsz4，`_ce_sig.py` 验证 API 语义，及时发现 compile×Triton 不兼容——这套流程反复奏效。每次只改一个变量，以稳态端到端吞吐为准。

3. **最大收益来自结构性突破**：Fused CE 之所以 +33%（远超之前七个 1–7% 的优化），是因为它**同时解除显存和带宽两个约束**。找瓶颈时要问"这个改动解除了几个约束"。

4. **用富余 HBM 换通信**：MI300X 192GB 是资源，A1+A2 保留未分片参数换 all-gather，是这个思路的直接兑现。

5. **无效结果同样是产出**：ROCm 栈上大量"理论最优"路径（AITER/TunableOp/max-autotune/FP8 all-gather）实测不可用或回退。如实记录，避免团队重走。

6. **稳定性靠 init + 守卫双保险**：深度缩放 init 治本，non-finite 跳步守卫兜底。grad_clip 挡不住有限毒梯度。

7. **评测比 loss 更能反映进展**：下游 benchmark 单调性清晰，是比噪声训练 loss 更可信的进度指标。MMLU 的涌现拐点是本次训练最有说服力的成果信号。

8. **可复现的续训是长训生命线**：原子 checkpoint + latest 指针 + 数据快进 + world-size 无关续训，让 64→120 GPU 迁移得以无缝进行。

---

*本报告基于 Chimera 项目 2026-07 的真实训练、A/B 实验与评测数据整理。相关文档：`docs/data_scaling_1T_design.md`、`docs/chimera_1t_training_speedup_plan.md`、`docs/fp8_results.md`、`docs/incident_8b_loss_nan.md`、`docs/8b_training_report.md`、`docs/mi300x_8b_pretraining_optimization.md`。*
