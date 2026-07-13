# LM 预训练数据集详细调研报告（2026-07）

> 来源：arXiv / HuggingFace / NVIDIA / AI2 / DatologyAI / Reddit r/LocalLLaMA / X(Twitter) / mid-training survey。
> 对象：为 Chimera-8B 预训练做数据选型、配比、分阶段策略参考。
> 当前基线：`HuggingFaceFW/fineweb-edu` config `sample-100BT`。

---

## 0. 一页速览（截至 2026-07 的关键格局）

1. **网页质量榜（2025→2026 演进）**：`FinePDFs-EDU / Nemotron-CC-v2(.1) > Nemotron-CC-HQ ≈ DCLM-baseline > FineWeb-Edu`。
2. **合成数据已成"核心组件"**：BeyondWeb/Nemotron-Synth/Cosmopedia 证明 rephrasing 可 2.7–7.7× 提速；
   到 2026，前沿模型合成占比可达 ~33%（Nemotron 3 post-training 3.5T/10.6T 为合成）。
3. **三阶段训练已是标准**：pretraining（大而杂）→ **mid-training/annealing**（小而精，学习率衰减期切高质量 math/code/instruct）→ context extension（长上下文）。
4. **PDF 成新富矿**：FinePDFs 从 PDF 解放 3T token，不混合就超过 Nemotron-CC-v2。
5. **多语言/全许可**：FineWeb-2（1000+ 语言，HF 下载量第一）、**Common Pile ~8T（全版权清洁）** 兴起。

---

## 0.5 2026 H1 最新动态（今天是 2026-07，这些是过去半年的新变化）

- **NVIDIA Nemotron 3 家族发布（GTC，2026-03-11）**：
  - 同时开源 **~10T 预训练 token**（含 Common Crawl 代码抽取 + 2.5T 新英文 web token）。
  - 数据集升级为 **Nemotron-CC-v2.1 / Nemotron-Pre-Training-Dataset-v2.1**，服务 Nemotron 3 系列。
  - **Nemotron 3 Ultra**：550B 总 / 55B active 的 **MoE + Mamba-Transformer 混合**；**Nemotron 3 Super** 亦已发布。
  - post-training 数据集 10.6T token 中 **约 33%（3.5T）为合成** —— 印证"合成占比大幅上升"趋势。
- **Common Pile（AI2 + EleutherAI + 社区）**：~8T token，**明确版权清洁 / 全许可**，成为"最干净可商用"的大池首选。
- **Zyda 2（Zyphra）~5T、HPLT 2.0（CC-0 多语言）** 等新宽松许可大池进入视野。
- **DCLM 稳定在 ~3.8T，采用 CC-BY-4.0**，授权友好度确认（商用省心）。
- **FineWeb-2 成为 HF 下载量第一** 的开放预训练集（~10T 多语言）。
- 合成数据研究进入"Langlais 2026 框架"：Memory(rephrasing) / Logic(reasoning primitives) / …，即按用途细分合成类型。

---

## 1. 网页类数据集（预训练主体，大 token 池）

| 数据集 | 规模 | 出品 | 质量/特点 | 授权 |
|---|---|---|---|---|
| **FineWeb** | 15T | HF | 干净去重通用 CC，96 快照，基线 | ODC-By（宽松） |
| **FineWeb-Edu** | 1.3T（另有 5.4T score-2 版） | HF | 分类器筛教育性，reasoning 强，**我们在用** | ODC-By |
| **FineWeb-2** | ~3T words / 8TB | HF | **1000+ 语言**多语言版，全开放 | ODC-By |
| **DCLM-baseline** | ~4T | Apple/UW/mlfoundations | model-based 过滤强，llm.c 复现优于 FineWeb-Edu | 宽松 |
| **Nemotron-CC** | 6.3T | NVIDIA | CC 精炼 + 合成改写，长 horizon SOTA | NVIDIA（较严，见下） |
| **Nemotron-CC-v2 / v2.1** | v1 collection 6.6T | NVIDIA | +8 个 2024–25 快照，Qwen3-30B 改写，多语 QA×15 | ⚠️ NVIDIA 许可较严，商用需查 |
| **FinePDFs** | **3T（PDF 提取）** | HF | 从 PDF 解放；FinePDFs-EDU **不混合即超 Nemotron-CC-v2** | ODC-By |
| **Common Pile** | ~8T | AI2+EleutherAI+社区 | **版权清洁/全许可**，2025–26 最干净可商用大池 | 宽松（curated） |
| **Zyda 2** | ~5T | Zyphra | 精炼混合，宽松许可 | 宽松 |
| **HPLT 2.0** | 多语言 | HPLT 联盟 | CC-0，多语言 | CC-0 |
| **RedPajama-V2** | 30T | Together | 超大原始 CC + 质量信号，需自己过滤 | 宽松 |
| **SlimPajama** | 627B | Cerebras | RedPajama 去重精简版 | Apache-2.0 |
| **C4 / The Pile** | 0.75T / 0.8T | Google / EleutherAI | 经典老基线，现多作对照 | 宽松 |

> ⚠️ **授权（2026 确认）**：**DCLM = CC-BY-4.0、FineWeb/FinePDFs = ODC-By、Common Pile = 全许可 → 商用省心**。
> **Nemotron-CC 系为 multi-license，商用前务必核对**；X 上（xlr8harder）曾指出 FinePDFs 的 benchmark 对比可能触及 Nemotron-CC-v2 条款。

---

## 2. 代码数据集

| 数据集 | 规模 | 出品 | 特点 |
|---|---|---|---|
| **The Stack v2** | ~900B（67TB 源） | BigCode | StarCoder2 训练集，600+ 语言，license 过滤 |
| **StarCoderData** | ~250B | BigCode | The Stack v1 精炼，经典代码源 |
| **Nemotron-Pretraining-Code-v1** | 大规模 | NVIDIA | GitHub + OpenCoder 启发式过滤 + 11 语言 code QA 合成 |
| **OpenCoder pretrain** | 数百 B | OpenCoder | 强质量过滤管线，附完整 recipe |

配比经验：现代通用模型 **code 占 5–20%**；code 专用模型（Qwen2.5-Coder）里还会掺 math/science。

---

## 3. 数学 / 科学数据集

| 数据集 | 规模 | 出品 | 特点 |
|---|---|---|---|
| **Nemotron-CC-Math-v1** | 133B | NVIDIA | Lynx+LLM 保留公式/代码，标准化 LaTeX，宣称超越以往所有数学集 |
| **FineMath** | ~50–100B | HF | 从 CC 提数学，FineWeb 团队出品 |
| **OpenWebMath (OWM)** | ~15B | — | 经典数学网页，SmolLM2/OLMo 都用 |
| **Proof-Pile-2** | ~55B | EleutherAI | arXiv + 数学论坛 + 代码，Llemma 用 |
| **DeepSeekMath corpus** | 120B | DeepSeek | 数学专用 CC（非全开放） |

---

## 4. 合成数据集（2025 的重头戏）

**核心思想**：用 LLM 把原始网页 **rephrase/改写** 成信息密度更高、更干净、更多样的版本，突破"数据墙"。

| 数据集/方法 | 出品 | 关键结论 |
|---|---|---|
| **Cosmopedia (v2)** | HF | 25B+ 合成教科书/故事，复刻 Phi 路线；开源合成数据先驱 |
| **Nemotron-Synth（Nemotron-CC HQ 合成子集）** | NVIDIA | Nemotron-CC 内的高质量合成子集，5 种 prompt 改写 |
| **BeyondWeb** | DatologyAI | **targeted rephrasing**；8B 上超 Cosmopedia +5.1pp、超 Nemotron 合成 +2.6pp；**训练提速 2.7–7.7×**；3B@180B 超 8B@180B(Cosmopedia) |
| **phi 系列 "textbooks"** | Microsoft | 高质量合成教科书，小模型高分的起点（非全开放） |

### 2026 新增合成数据集（今年上半年）
| 数据集/方法 | 出品 | 关键结论 |
|---|---|---|
| **FinePhrase** ★ | HuggingFace | **486B token 开源改写数据**（`HuggingFaceFW/finephrase`）；90 组实验、生成 >1T token 调出最优 recipe。**结论爆点见下** |
| **Nemotron-Synth** | NVIDIA | Nemotron-CC-v2.1 内的高质量合成子集，classifier 引导质量过滤 |
| **Code Concepts / synthetic-code-concepts** | NVIDIA | 预训练级合成代码，1500 万 Python 编程题（针对 code 能力） |
| **Nemotron-Pretraining-Code-v2** | NVIDIA | v2.1 刷新，更高质量 + 更多语言的合成 code QA |

**★ FinePhrase 论文（arXiv 2604.13977，Niklaus/Penedo/Wolf 等 HF 团队）——2026 最实用的合成数据指南：**
1. **结构化输出格式最强**：把网页改写成 **表格 / 数学题 / FAQ / 教程** 等结构化形式，
   一致地超过 curated web 基线和以往所有合成方法。
2. **generator 模型 >1B 无额外收益** —— 用 ≤1B 小模型当 rephraser 就够（大幅省成本，呼应 BeyondWeb 经验5）。
3. **混合用的原始数据选择影响很大**（跟 DCLM/FineWeb-Edu 质量分挂钩）。
4. FinePhrase **超过所有现有合成基线，同时生成成本降低最多 30×**。全开源：数据 + 所有 prompt + 生成框架。
5. 但仍需保留原始 web：**合成 token 给逻辑深度，原始 web 保 commonsense 和语言多样性**（2604.13977 明确结论）。

**BeyondWeb 的 7 条经验（很有指导价值）**：
1. 合成 ≠ 单纯蒸馏；简单 summarization-rephrase 就能接近 Cosmopedia，但精心设计能更好。
2. 单纯用 LLM"扩写"网页几乎无增益，等于重复数据，突破不了数据墙——**必须有意设计**。
3. 高质量 seed 数据有帮助但边际递减；有限的高质量数据更该当"种子"去合成。
4. 合成数据带来的多样性 > 单纯质量。
5. 小模型（≤3B）当 rephraser 就够划算，不必用大模型生成。
6. 合成数据在 pretraining 中后期收益最大。
7. 学界警告（UC Berkeley 讲义 / Kang 2025）：**rephrased 数据无退化，但纯 textbook-style 高占比会退化** → 合成别喂太多、别单一风格。

### ⭐ 可直接下载的"改写好"成品合成数据（拿来即用，无需自建生成管线）
| 数据集 | HF 路径 | 规模 | 类型 | 授权 |
|---|---|---|---|---|
| **FinePhrase** ★首选 | `HuggingFaceFW/finephrase` | **486B** | 网页改写成表格/数学题/FAQ/教程 | ODC-By（宽松） |
| **Cosmopedia v2** | `HuggingFaceTB/cosmopedia` | ~25B+ | 合成教科书/故事/博客 | Apache-2.0 |
| **Nemotron-CC**（含合成子集） | `nvidia/Nemotron-CC-v2` | 含 rephrase 子集 | CC 网页改写 | ⚠️ Nemotron 多许可 |
| **Nemotron-Pretraining-Code-v2** | `nvidia/Nemotron-Pretraining-Code-v2` | 大规模 | 合成 code QA | ⚠️ Nemotron 许可 |
| **DCLM baseline** | `mlfoundations/dclm-baseline-1.0` | ~3.8T | model-filtered web（部分改写） | CC-BY-4.0 |

- **⚠️ BeyondWeb 无公开数据集**：DatologyAI 只发方法（blog/论文），数据商用不可下载 → 其开放替代品 = **FinePhrase**。
- **拿来即用首选 = FinePhrase**：唯一同时满足「已改写好 + 大规模(486B) + 授权宽松 + SOTA 质量」。
  下载：`huggingface-cli download HuggingFaceFW/finephrase --repo-type dataset`，再走现有 Qwen3 tokenize → uint32 .bin。
- **务实起手混合**（照抄社区经验，防退化）：
  `FineWeb-Edu(原始 web) 60–70%` + `FinePhrase(合成改写) 20–30%` + `Cosmopedia(合成教科书) 5–10%`。

---

## 5. 已知模型的预训练数据配比（可直接抄）

### SmolLM2（HF, 1.7B, 11T token）★最清晰的实战配比
- 英文网页：**FineWeb-Edu : DCLM = 40 : 60**
- **~10% 数学**（OpenWebMath + 英文文本部分）
- **代码**：The Stack
- 多阶段训练，末期切高质量数据。
> 对我们最有参考价值：直接给了 FineWeb-Edu 和 DCLM 的黄金配比。

### OLMo 3 / Dolma 3（AI2, 7B/32B）★最透明的全开放 recipe
- **Stage 1 预训练**：Dolma 3 Mix，**6T token**（Web + code + math 混合，code/math 占比高于前代）。
- **Stage 2 mid-training（Dolmino）**：**100B token**，从 2.2T 高质量池采样（math/science/code/instruction）。
- **Stage 3**：长上下文扩展。

### NVIDIA Nemotron Nano 2（9B/12B）
- Nemotron-Pre-Training-Dataset-v1 **6.6T token**：精炼 CC 网页 + 数学(133B) + 代码 + 合成 SFT-style。
- 一个 8B 训 15T token，其中 7.2T 来自 Nemotron-CC。

### Qwen2.5-Coder（配比示例）
- 80% general + 20% reasoning；reasoning 内部 = 17% code / 56% math / 27% science。

### 经验性通用配比（综合上述）
```
高质量 Web        60–75%   (DCLM / Nemotron-CC / FinePDFs / FineWeb-Edu)
Code             10–20%   (The Stack v2 / Nemotron-Code)
Math/Science      5–10%   (FineMath / OWM / Nemotron-CC-Math)
合成/改写 QA       5–15%   (Cosmopedia / 自建 rephrase)
```

---

## 6. 分阶段训练用哪些数据（mid-training / annealing 已成标准）

来源：*Mid-Training of LLMs: A Survey* (arXiv 2510.06826 / 2510.23081)、OLMo 3、Cameron Wolfe。

| 阶段 | 数据特征 | 学习率 | 典型 token 量 | 数据例 |
|---|---|---|---|---|
| **① Pretraining（主体）** | 大而杂，覆盖广 | 恒定/缓降 | 数 T–15T | FineWeb-Edu/DCLM/Nemotron-CC + code + math 大混合 |
| **② Mid-training / Annealing** | 小而精，高质量、高密度 | **快速衰减** | 50–300B | Dolmino 类：精选 math/science/code/instruction + 合成 QA |
| **③ Context extension** | 长文档 | 低 | 数十 B | 长 code、书、arXiv、拼接长序列 |

**关键洞见**：
- **Annealing 阶段（学习率快速 decay）是"点睛"窗口**——此时喂高质量 math/code/instruction 收益最大（多篇论文一致，有工作用 RL 引导 annealing 数据选择）。
- Mid-training 保留部分旧数据防遗忘 + 注入新的高质量专门数据（"bridge"）。
- 合成数据在中后期收益最大（BeyondWeb 经验 6）。

---

## 7. 对 Chimera-8B 的落地建议

**现在（验证期）**：`fineweb-edu sample-100BT` 保持不动，先把 8B loss/吞吐跑稳。

**下一步升级（按性价比 + 授权友好度排序）**：
1. **Web 源升级**：混入 **DCLM-baseline** 或 **FinePDFs-EDU**（授权省心，质量≥Nemotron-CC-v2）。
   照抄 SmolLM2：`FineWeb-Edu : DCLM = 40 : 60`。
2. **掺 code 10–15%**（The Stack v2 / StarCoderData）——直击我们 **HumanEval baseline 0%** 的痛点。
3. **掺 math 5–10%**（FineMath / OpenWebMath）。
4. **加一个 mid-training 阶段**：主训练后用 ~5–10% token 预算、学习率快速 decay，喂精选 math/code/instruct（抄 Dolmino）。
5. **合成数据（进阶）**：先用最省的 summarization-rephrase 试点（BeyondWeb 经验：简单方法就能接近 Cosmopedia），别一次喂太多、别单一 textbook 风格。

**授权提醒**：优先 FineWeb 系 / FinePDFs / DCLM / The Stack；Nemotron 系商用前核对 license。

---

## 参考来源
- HF：FineWeb / FineWeb-Edu / FineWeb-2 / FinePDFs(Blog) / Cosmopedia(blog) / SmolLM2-1.7B card
- NVIDIA：Nemotron-CC blog + arXiv 2412.02595；HF nvidia/Nemotron-CC-v2(.1) / Nemotron-CC-Math-v1 / Nemotron-Pretraining-Code-v1
- AI2：allenai/dolma3；allenai.org/blog/olmo3；olmo-3.pdf；Cameron Wolfe "Olmo 3"
- DatologyAI BeyondWeb（blog + arXiv 2508.10975）
- Mid-training survey：arXiv 2510.06826、2510.23081；OpenReview RL-guided annealing
- SmolLM2：arXiv 2502.02737
- Reddit r/LocalLLaMA（数据集选型讨论）；X @xlr8harder（Nemotron license 争议）
- RUCAIBox/awesome-llm-pretraining
