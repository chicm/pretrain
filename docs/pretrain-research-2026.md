# 从0预训练 7B–10B 大模型：技术调研与实施建议

> 调研日期：2026-07-10（框架选型/近期进展更新）
> 适用场景：从零开始预训练一个 7B–10B 规模的 dense 语言模型
> 可用算力：集群1 = 8 节点 × 4×A100（32 卡，单机可用，无 IB）；集群2 = 4 节点 × 8×MI300X（32 卡，主力，有 IB）
>
> **当前决策（2026-07 更新）**：用 **FSDP2 + FineWeb sample-10BT** 在 **MI300 集群**（主力）上跑通并做真实数据训练；A100 集群因 NC SKU 无 InfiniBand，仅单机可用（数据流水线 / 1B 消融 / 评测）。多机训练全部在 MI300。

---

## 0. 一页速览（TL;DR）

- **算力够用**：MI300 集群（32 卡 MI300X，192GB HBM3/卡）是主力，训练 8B / 2–3T tokens 约需 **3–4 个月**；A100 集群（32 卡，无 IB 仅单机）定位为**数据流水线 + 小模型消融 + 评测/后训练**。
- **不要跨集群合并训练**：A100 与 MI300 指令集/通信库不同，且两集群间无 InfiniBand 互联，异构并行会被拖死。
- **训练框架**：大厂（DeepSeek/Qwen/GLM）底座都是 **Megatron 系**，但那是为百 B MoE 服务的。7–10B dense 用 **FSDP2** 更简单够用。**本项目起步：FSDP2**（TorchTitan 或纯 PyTorch），日后要训 MoE/30B+ 才迁 Megatron。
- **模型结构**：选 **dense decoder-only（命名 Chimera）**。对齐 2026 主流：GQA + RoPE + RMSNorm(Pre-Norm) + SwiGLU + **QK-Norm**，可选借鉴 Gemma 的「滑动窗口 + 全局注意力交替」（预训练阶段关闭，全 full attention）。不上 soft-capping。
- **最该抄的配方**：**OLMo 3-7B**（全流程开源，含数据+代码+recipe，官方称对标 Qwen3）+ **SmolLM3**（三阶段数据配比全公开）。Qwen/Gemma/GLM 只放权重不放配方，参考价值不如前两者。
- **数据**：FineWeb-Edu / Nemotron-CC（网页）+ The Stack v2（代码）+ Dolma（书籍/论文/百科）+ 数学。退火阶段务必混入**合成/改写数据**（2026 头号质量变量）。
- **Token 预算**：8B 训 **2–3T tokens 起步**（约 250–375 tokens/参数），预算充足可到 8T。避免极端过训练（>2000 tokens/参数会损害后续微调）。

---

## 1. 算力评估：够不够用

### 1.1 计算公式与硬件

预训练计算量经验公式：`FLOPs ≈ 6 × 参数量 × token 数`

| 集群 | 配置 | 卡数 | 单卡 BF16 峰值 | 备注 |
|---|---|---|---|---|
| 集群1 | 8 节点 × 4×A100 | 32 | ~312 TFLOPS | A100-80GB（无 IB，仅单机） |
| 集群2 | 4 节点 × 8×MI300X | 32 | ~1.3 PFLOPS | MI300X，约为 A100 的 ~4×（主力，有 IB） |

按实际 MFU（有效算力利用率）35–40% 估算，训练 **8B 模型** 的墙钟时间：

| Token 预算 | 集群2（32×MI300X）单独 | 集群1（32×A100）单独 |
|---|---|---|
| 1T tokens | ~30–36 天 | ~4 个月 |
| 3T tokens（推荐起步） | ~3–3.5 个月 | 不现实（~1 年） |
| 8T tokens（追 SOTA） | ~8–10 个月 | 不现实 |
| 15T tokens（Llama3 级） | 不现实 | 不现实 |

> 注：以上为**数量级估算**。实际会因 MI300 具体型号（X/325/355）、互联带宽、并行策略而变化，正式立项前应以集群实测吞吐为准。

### 1.2 结论

- **MI300 集群是绝对主力**，32 卡跑 8B、2–3T tokens 可行（约 3–4 个月）。
- **A100 集群单独跑全量预训练太慢**，最佳定位：
  - 数据清洗/去重/tokenize 流水线
  - 1B proxy 模型的架构与数据配比消融实验
  - 最终评测 + SFT/对齐后训练
- **不要跨集群合并训练**：异构 GPU + 无跨集群高速互联 = 效率被最慢一方拖死。

---

## 2. 训练框架（2026 现状）

### 2.0 大厂实际用什么？—— Megatron 是事实标准底座

| 大厂 | 预训练框架 | 备注 |
|---|---|---|
| **DeepSeek** | 自研 **HAI-LLM**（闭源），技术全公开 | NVIDIA 已把 DeepSeek-V3 的 MLA / MTP / DeepSeek-MoE 移植进 Megatron-Core / Megatron-Bridge，复现走 Megatron。 |
| **Qwen（阿里）** | **Megatron-LM/Core** 系 | 魔搭 Megatron-SWIFT 明确支持 Qwen3/Qwen3-MoE 预训练。 |
| **GLM（智谱）** | **Megatron-LM** 训练 + SGLang 推理 | GLM-5.2 后训练框架 `slime` = Megatron 训 + SGLang rollout。 |
| **Llama（Meta）** | 自研内部栈 | 社区复现全在 Megatron / TorchTitan。 |

> 结论：**Megatron-LM（= Megatron-Core + 训练脚本）是行业事实标准的底层预训练框架。** DeepSeek/Qwen/GLM 本质都是「Megatron 系 + 各自自研优化」。它们用 Megatron 全套多维并行，是因为它们是几百 B 的大 MoE，单层都放不下一张卡——不是因为「只有 Megatron 才专业」。

### 2.1 FSDP2 vs Megatron —— 本项目的核心选型

| 维度 | **FSDP2** | **Megatron(-Core)** |
|---|---|---|
| 并行思路 | 沿数据/参数切分（ZeRO-3），每步 all-gather 权重 | 沿模型结构切分：TP 切矩阵、PP 切层、EP 切专家 |
| 通信 | all-gather + reduce-scatter | TP 需层内高频 all-reduce（要 NVLink/xGMI 级带宽）、PP 点对点 |
| 代码侵入性 | **低**，`fully_shard(model)` 包一下，模型定义基本不动 | **高**，需用 ColumnParallelLinear/RowParallelLinear 重写模型 |
| 适合规模 | 单层能放进一张卡（≤~13B dense 很舒服） | 单层放不下 / 超大 MoE / 需要极致 MFU |
| 上手难度 | 简单（PyTorch 原生，TorchTitan 底座） | 陡峭，配置项多 |
| 大规模 MFU | 略低（通信更重） | 更高（尤其加 TP 后） |

**决策：7–10B dense 用 FSDP2 就够，且更省心。** 8B 的单层完全放得进一张 A100/MI300，不需要 TP 拆层。只有当哪天要训 MoE 或 30B+ 时，才真正需要迁到 Megatron 全套。

现代做法其实是**组合**：`TP/PP(Megatron) + FSDP/ZeRO(数据维度)`。真实光谱：

```
纯 FSDP2  ──►  FSDP2 + TP(少量)  ──►  Megatron 全套 TP+PP+EP
 (本项目起步)     (10B+ 可选)         (百B/MoE 才需要)
```

注意：MI300 走 ROCm，**FSDP2 是 PyTorch 原生、ROCm 支持好**；Megatron 在 ROCm 上 AMD 有官方移植，但 TP 强依赖节点内 xGMI 带宽，正式上 TP 前先实测节点内 all-reduce 带宽。

### 2.2 各框架定位

| 框架 | 适用集群 | 说明 |
|---|---|---|
| **FSDP2（纯 PyTorch / TorchTitan）** ⭐ 起步首选 | A100（跑通）→ MI300 | 本项目第一步。原生、简单、ROCm 友好，7–10B dense 足够。 |
| **AMD Primus + Megatron-Core** | MI300（若日后需 MoE/更大规模） | AMD 官方统一训练框架，为 Instinct 优化，PyTorch+Megatron docker 开箱即用。 |
| **Megatron-LM / NVIDIA NeMo** | A100（工业级） | 最成熟 3D 并行基准；7–10B dense 其实用不到复杂并行。 |

**当前推荐**：起步阶段 A100 上用 **FSDP2（TorchTitan 或纯 PyTorch）** 跑通；若日后训 MoE/30B+ 再迁 Megatron。

**参考链接**
- FSDP2 文档：https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- TorchTitan 论文/仓库：https://arxiv.org/html/2410.06511v3 、https://github.com/pytorch/torchtitan
- Primus 深度解析：https://rocm.blogs.amd.com/software-tools-optimization/primus-deep-dive/README.html
- ROCm Megatron 预训练教程：https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html
- AMD 训练指南 2026：https://www.spheron.network/blog/train-llm-amd-gpu-rocm-mi300x-mi355x-zaya1-guide/

---

## 3. 模型结构（2026 最佳实践）

7–10B 规模推荐 **dense decoder-only**，不上 MoE（MoE 在小规模收益低、系统复杂度高，你的卡数也不需要）。对齐 Qwen3 / Llama3.x / Gemma4 这一代成熟配置：

| 组件 | 选择 | 说明 |
|---|---|---|
| Attention | **GQA**（分组查询注意力） | 省 KV cache，已成标配 |
| 长上下文注意力 | **滑动窗口 + 全局注意力交替** | Gemma4 主流做法，省长上下文显存/算力 |
| 位置编码 | **RoPE**（可配 NTK/YaRN） | 长上下文扩展 |
| 归一化 | **RMSNorm + Pre-Norm** | 部分新模型加 QK-Norm 稳训练 |
| 激活 | **SwiGLU** | — |
| bias | 无 bias | — |
| 上下文长度 | 预训练 4K–8K，后期扩 32K+ | 分阶段扩展 |
| 词表 | ~12.8 万 | 建议复用 Qwen3/Gemma tokenizer，别从头训 |

**参考 8B 配置**：hidden ~4096、layers ~32、heads 32、KV heads 8、SwiGLU intermediate ~14336。

**新兴可选（先只在 proxy 消融，别用于正式跑）**：KV sharing、hyper-connections 等 2026 上半年架构改进，收益仍在验证。

### 3.1 本项目最终架构决策：**Chimera**

模型代号 **Chimera**（`Chimera-tiny` / `Chimera-1B` / `Chimera-8B`）。取"奇美拉/混合体"之意——Qwen3 底座 + Gemma 长上下文能力的混血，且不蹭 Qwen/Gemma 商标。定型如下：

| 决策 | 结论 | 说明 |
|---|---|---|
| **底座** | Qwen3-8B dense | dim 4096 / 32 层 / 32 Q-head / 8 KV-head（GQA 4:1）/ SwiGLU 14336 / RoPE / RMSNorm(Pre-Norm) / 无 bias |
| **tokenizer** | Qwen3（vocab_size **151936**） | 151669 向上取整到 128 倍数；中英文+代码友好 |
| **QK-Norm** | ✅ **默认开** | 对 per-head Q/K 做 RMSNorm，稳训练、抑制 loss spike；成本极低。Qwen3 本身即用，非"抄 Gemma" |
| **混合滑窗注意力** | 🔧 **架构预留，预训练默认关闭** | 实现为可配置 `layer_types`（full/sliding）+ `sliding_window`。预训练 4K–8K 全用 full attention（贴近 Qwen3 已验证配置，降低翻车风险）；等长上下文扩展（32K+）阶段再开 Gemma 式 **5:1 local:global 交错 + 末层强制 global**（`make_gemma_layer_types()`）。窗口小于等于 0 时强制全 full，安全兜底 |
| **GQA 比例** | 统一用 Qwen3 原生比例 | **不**搞局部层/全局层异构 GQA（那是未经大规模验证的"发挥"，KISS 原则） |
| **Logit/attn soft-capping** | ❌ **不上** | Gemma3 起已被 QK-Norm 替代，且与 FlashAttention/SDPA 兼容差（fallback 慢路径）。既已上 QK-Norm，soft-capping 纯属累赘 |

**核心哲学**：预训练（最烧钱、最不容错）阶段尽量贴近 Qwen3 已验证配置；把 Gemma 最值钱的一点（长上下文效率）做成**可插拔架构预留**，等 base 训好、进入长上下文扩展阶段再启用。既拿到收益，又不在预训练引入未验证改动。

---

## 4. 数据集（决定质量的头号因素）

### 4.0 跑通代码用的小数据集（几十 GB 内，快速下载）⭐ 当前阶段

按体量从小到大，先小后大跑通流程：

| 数据集 | 体量 | Token 量 | 用途 | HF 名称 |
|---|---|---|---|---|
| **TinyStories** | ~1–2 GB | ~0.5B | 秒级跑通/debug 首选，能真训出会说话的小模型 | `roneneldan/TinyStories` |
| **MiniPile** | ~6 GB | ~1.5B | The Pile 精简版，多样性好 | `JeanKaddour/minipile` |
| **FineWeb sample-10BT** ⭐ | ~28 GB | 10B (GPT2) | 正经跑通全流程首选，质量高 | `HuggingFaceFW/fineweb`, `name="sample-10BT"` |
| **FineWeb-Edu sample-10BT** ⭐ | ~28 GB | 10B | 教育精选版，小模型 loss 更好看 | `HuggingFaceFW/fineweb-edu`, `name="sample-10BT"` |
| Cosmopedia v2 (SmolLM-Corpus) | 稍大 | ~28B | 合成教科书数据，想试合成数据可选 | `HuggingFaceTB/smollm-corpus` |

> 参考：FineWeb `sample-100BT` = 277 GB（偏大）；`sample-10BT` 的 28 GB 是甜点区。

**推荐两段式**：
1. **TinyStories (~1GB) 秒级跑通** — 验证分布式启动、FSDP2 分片、loss 下降、checkpoint 存取。
2. **FineWeb sample-10BT (~28GB)** — 换真实网页数据，~1B 模型跑几百到几千步，确认真实数据上 loss 曲线健康。正式 scale 前的最后验证。

下载示例：
```python
from datasets import load_dataset
# TinyStories（最小，先跑这个）
ds = load_dataset("roneneldan/TinyStories")
# FineWeb 10B 子集（跑通正式流程）
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")
# 流式：不下全量即可开跑
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
```
> 国内下载慢：`export HF_ENDPOINT=https://hf-mirror.com` 走镜像。

### 4.1 主流开源预训练数据

| 数据集 | 规模 | 说明 |
|---|---|---|
| **FineWeb / FineWeb-Edu**（HF） | 15T+ / 1.3T | 英文网页质量标杆，Edu 版精选教育内容 |
| **Nemotron-CC**（NVIDIA） | 6.3T | 高质量 Common Crawl，unique token 约为 DCLM 的 4×，长周期训练强 |
| **DCLM-Baseline** | ~4T | DataComp-LM 精选，质量基准 |
| **The Stack v2**（BigCode） | 代码 | 代码能力必备 |
| **Dolma**（AllenAI） | 3T | 含书籍/论文/百科，来源合规 |
| **FineWeb2** | 多语言 | 需要多语言时补充 |

### 4.2 配比建议（起步 recipe）

网页(FineWeb-Edu/Nemotron-CC) **70%** + 代码(Stack v2) **15%** + 书籍/论文/百科(Dolma) **10%** + 数学 **5%**。

### 4.3 合成数据（2026 头号变量）⭐

- SmolLM3 后训练几乎全部使用 DeepSeek-R1 等模型生成的数据。
- FineWeb 团队做了 333 组改写实验（FinePhrase）寻找最佳合成数据配方。
- **建议在退火（decay）阶段务必混入合成/改写数据**，这是拉开质量差距的关键。

### 4.4 Token 预算

- Chinchilla 的 20 tokens/参数已过时。2026 SLM 普遍严重「过训练」以换取推理期性价比。
- **建议：8B 训 2–3T tokens 起步**（约 250–375 tokens/参数），预算充足可到 8T。
- **警告**：极端过训练（>2000 tokens/参数）会损害后续微调（见 "Overtrained LMs Are Harder to Fine-Tune"），别无脑堆到 15T。

---

## 5. 2026 最新模型格局（参考谁、抄谁）

### 5.1 可「全流程照抄」的开源配方 ⭐（对从0开始最有价值）

| 模型 | 规模 | 为什么值得抄 |
|---|---|---|
| **OLMo 3**（AllenAI, 2026） | 7B / 32B | **pretrain + mid-training + post-training 全流程开源**，含数据配方 + `Olmo-Core` 训练代码，checkpoint 可精确复现；官方称 7B recipe 对标 Qwen3。**这是你的教科书。** |
| **SmolLM3**（HuggingFace, 3B） | 3B | 公开**三阶段预训练精确数据配比** + 架构细节 + decay 阶段数据实验；数据集全部公开在 HF。适合先在 A100 上小规模复现验证流程。 |

链接：
- OLMo 3：https://allenai.org/blog/olmo3 ，解读 https://www.interconnects.ai/p/olmo-3-americas-truly-open-reasoning
- SmolLM3：https://huggingface.co/blog/smollm3 ，数据集 https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets

### 5.2 强但「只放权重不放配方」的模型（看趋势，别硬抄）

| 模型 | 类型 | 要点 |
|---|---|---|
| **Gemma 4**（Google, 2026-04） | dense + MoE | 家族：E2B/E4B/12B/31B dense + 26B-A4B MoE（128 专家选 8）；架构亮点=**滑动窗口+全局注意力交替**、原生多模态；31B 跑分 MMLU-Pro 85.2 / AIME2026 89.2；Reddit「gemma-4 is killing it」。技术报告 https://arxiv.org/html/2607.02770v1 |
| **GLM-4.5/4.6 → GLM-5/5.1/5.2**（智谱） | 大 MoE | 200K 上下文、agentic coding 极强，与 DeepSeek V4 / Kimi K2.6 平起平坐。但全是几百 B MoE，**路线不适合你复制**。 |
| **Qwen3 / Qwen3.5** | dense + MoE | Qwen3 预训练三阶段、30T+ tokens；架构（GQA/RoPE/RMSNorm/SwiGLU）是 dense 参考基线。 |
| **DeepSeek V4 / Kimi K2.6** | 大 MoE | 2026 开源前沿，与闭源模型交手；同样是大集群玩法。 |

> 关键结论：Gemma4/GLM/DeepSeek 证明的趋势是「MoE + 多阶段训练 + 合成数据」，但它们不放数据配方（X 上戏称「预训练配方是新的可口可乐配方」）。**真正能照抄的是 OLMo3 / SmolLM3。**

### 5.3 榜单参考

- LMArena（arena.ai/leaderboard/text）：开源权重实时 Elo 排名
- HuggingFace Open LLM Leaderboard：https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- Artificial Analysis：https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard
- llm-stats 开源榜：https://llm-stats.com/leaderboards/open-llm-leaderboard

### 5.4 架构验证利器

- **modded-nanogpt**（Keller Jordan speedrun）：https://github.com/kellerjordan/modded-nanogpt
  社区把它优化到比原版快 30 倍，是在小规模快速验证架构点子的最佳试验田。正式跑前用它验证你的架构改动。

---

## 6. 实施路线（落地步骤）

> 进度标记：✅ 已完成 · 🔄 进行中 · ⬜ 待办。下方阶段划分保留原规划骨架，并对齐实际进展。

### 阶段 0：环境 + 跑通代码 ✅ 已完成
> 关键变更：原计划在 A100 集群做多机验证，但实测 **A100 (NC A100 v4 SKU) 无 InfiniBand**（仅以太网加速网络，无 `/dev/infiniband`），多机 FSDP 受 TCP 延迟限制在 step 0 后即挂死——**A100 多机是死胡同**。因此多机验证全部转到 **MI300 集群**（有 8×IB HCA、192GB HBM3/卡、8 GPU/节点），MI300 成为主力平台。
- ✅ 装好 PyTorch + FSDP2（`torch.distributed.fsdp.fully_shard`）；A100 用 CUDA，MI300 用 ROCm 6.4 + RCCL。
- ✅ 小模型（~1B，**Chimera** = Qwen3 结构 GQA+RoPE+RMSNorm+SwiGLU+QK-Norm）实现完成。1b preset 实际 ~1.444B 参数（含 embedding）。
- ✅ **TinyStories** tokenize（473,992,236 tokens）→ 冒烟测试通过：A100 单机 loss 收敛；MI300 单机 ~1.88M tok/s（≈4× A100）。
- ✅ **MI300 多机 1B 训练跑通（核心里程碑）**：4 节点 × 8 GPU = 32 卡，IB/RCCL，~11s/step、~928K tok/s，loss 平滑下降，checkpoint 中途+结尾均正确保存（ROCm 上集合通信保存逻辑验证无误），干净退出。
- ✅ checkpoint 存/恢复验证（`get_model_state_dict(full_state_dict=True)`，全 rank 参与、master 落盘）。

### 阶段 1：数据流水线 🔄 进行中
> 变更：数据格式改用**自研 packed `.bin` + memmap**（配合 FSDP2 训练循环），不用 Megatron mmap 格式。
- ✅ TinyStories 全流程打通。
- ✅ 存储分层确定：**代码** → 各节点本地盘（git 同步，强一致，避免网络文件系统缓存）；**数据/checkpoint** → 共享存储（大文件顺序读写快）；**HF 下载缓存** → 本地盘（切忌写共享盘，小文件极慢）。
- ✅ 优化 `data.py` 逐 token 填充循环（已向量化：memmap + 批量 `np.concatenate`）。
- ✅ tokenize **FineWeb `sample-10BT`**（HF `HuggingFaceFW/fineweb`）→ ~10.2B tokens（uint32）→ 真实数据多机训练已启动。
- ⬜ 扩充数据：FineWeb-Edu + Nemotron-CC + The Stack v2 + Dolma；去重、质量过滤。
- ✅ 确定 tokenizer：**Qwen3**（`Qwen/Qwen3-8B`，vocab padded 151936，eot `<|endoftext|>` id 151643）。

### 阶段 2：小模型消融 ⬜ 待办（MI300 集群）
- 用 **1B proxy 模型**，在 ~50–100B token 上跑几组数据配比 / 学习率消融。
- 确认 loss 曲线健康、无 spike。可先复现 SmolLM3/OLMo3 的 1B 设置。
- 这一步能在正式跑前避免几十天的浪费。

### 阶段 3：正式预训练 ⬜ 待办（MI300 集群，主力平台）
- 8B dense，2–3T tokens。框架：**FSDP2**（若 MFU 不足或需扩上下文，再加 TP=2）。192GB HBM3/卡显存充裕，8B 有余量。
- 架构对齐 Qwen3 / Gemma4。
- BF16 + FSDP2（8B 规模纯 FSDP/ZeRO 即可，不需复杂 3D 并行）。
- WSD 或 cosine 学习率，全局 batch ~4M tokens，勤存 checkpoint，监控 grad norm / loss spike。
- **结尾退火（decay）阶段**：混入高质量 + 合成/改写数据、降 LR 收尾提质。

### 阶段 4：评测与后训练 ⬜ 待办
- 评测：MMLU / MMLU-Pro / GSM8K / HumanEval / 中文评测；对标 OLMo3-7B、Qwen3。
- 后训练：SFT + 偏好对齐（DPO/GRPO 等）。

---

## 7. 关键决策清单（Checklist）

- [x] 主力模型 = 8B dense decoder-only（Chimera，Qwen3 结构，不上 MoE）
- [x] 起步框架 = FSDP2（MI300 多机跑通）；MoE/30B+ 才迁 Megatron
- [x] 第一步 = TinyStories(~1GB)秒级跑通 → FineWeb sample-10BT(~10.2B tokens) + 1B 模型，MI300 上用 FSDP2 跑通
- [x] 主训练集群 = MI300（FSDP2，必要时加 TP=2）
- [x] A100 集群 = 单机数据流水线 + 1B 消融 + 评测/后训练（无 IB，多机不可用）
- [x] tokenizer = Qwen3（vocab 151936）
- [ ] 配方模板 = OLMo 3-7B（主）+ SmolLM3（辅）
- [ ] Token 预算 = 2–3T 起步（避免 >2000 tokens/param 极端过训练）
- [ ] 退火阶段混入合成/改写数据
- [ ] 正式跑前用 1B proxy 验证架构（已用 FineWeb 1B 验证观测/收敛）
- [x] 不跨集群合并训练

---

## 附录：主要信息来源

**框架**：AMD ROCm blogs (Primus)、TorchTitan (arXiv 2410.06511, ICLR 2025)、NVIDIA NeMo/Megatron docs、spheron.network 2026 AMD 指南
**架构**：Gemma4 技术报告 (arXiv 2607.02770)、Sebastian Raschka "A Dream of Spring for Open-Weight LLMs"、Qwen3 技术报告、Maarten Grootendorst "Visual Guide to Gemma 4"
**数据**：FineWeb/FinePhrase (HuggingFace)、Nemotron-CC (NVIDIA, ACL 2025)、Dolma/DCLM、SmolLM3 数据集合集
**配方**：OLMo 3 (allenai.org/blog/olmo3, Olmo-Core repo, Interconnects/Cameron Wolfe 解读)、SmolLM3 (huggingface.co/blog/smollm3)
**Token 预算**：Databricks "How Long Should You Train"、"Overtrained LMs Are Harder to Fine-Tune" (OpenReview)、Chinchilla scaling laws
**榜单/社区**：LMArena、HF Open LLM Leaderboard、llm-stats、r/LocalLLaMA、modded-nanogpt (Keller Jordan)、METR nanoGPT 进展报告

> 免责声明：算力/时长为数量级估算，正式立项前请以集群实测吞吐为准。部分 2026 模型版本号（GLM-5.2、Qwen3.5、DeepSeek V4 等）来自搜索快照，使用前请核对官方最新发布。
