# Chimera-8B 预训练数据扩充至 1T token —— 设计文档（2026-07）

> 依据：`docs/pretraining-datasets-survey-2026.md`（数据集调研）+ SmolLM2/OLMo3 实战配比。
> 目标：从当前「纯 FineWeb-Edu 100BT（99.66B token）」扩充到 **1T token 多源混合**预训练语料。
> 约束：授权友好（公开 GitHub repo）、复用现有 Qwen3 tokenizer / uint32 .bin 管线、
> FSDP2 训练不变。全程自主实施，配比与规模已由用户拍板。

---

## 1. 目标与预算

| 项 | 值 |
|---|---|
| 目标 token | **1,000B（1T）** |
| 模型 | Chimera-8B（7.602B 参数）→ 125 tok/param（合理 over-train 区间）|
| .bin 体积 | 1T × uint32 = **~4TB**（共享盘 /scratch 可用 23T，充足）|
| 训练步数 | 1T / (global batch 4.2M) ≈ **238K steps** |
| 训练耗时 | 按 251K tok/s ≈ **~46 GPU-天**（32×MI300 上约 46 天 wall，或按可用时段分段）|

---

## 2. 数据配比（最终，全非门控可访问源）

> **重要现实修正**：集群无 HF token，`bigcode/the-stack-v2` / `starcoderdata` /
> `the-stack-dedup` 全部 **HF-gated 无法访问**；`proof-pile-2` / `github-code` 为
> 已废弃的 script-based dataset（datasets 库不再支持）。经 `_probe_sources` 系列
> 实测，改用以下**全部非门控 + 授权友好**的确认可访问源。若日后提供 HF token，
> 可换回 the-stack-v2 并提高 code 占比。

| 类别 | 占比 | token | 数据源（HF dataset，已实测 streaming OK） | 授权 | 本地目录 |
|---|---|---|---|---|---|
| **高质量 Web** | 42% | 420B | `mlfoundations/dclm-baseline-1.0-parquet` | CC-BY-4.0 | `dclm_tok/` |
| **教育 Web** | 24% | 240B | `HuggingFaceFW/fineweb-edu` (sample-350BT) | ODC-By | `fineweb_edu_240bt_tok/` |
| **PDF 源** | 8% | 80B | `HuggingFaceFW/finepdfs-edu` | ODC-By | `finepdfs_edu_tok/` |
| **Math** | 12% | 120B | `HuggingFaceTB/finemath` (finemath-3plus) | ODC-By | `math_tok/` |
| **Code** | 4% | 40B | `codeparrot/codeparrot-clean` (Python, 非门控) | 宽松 | `code_tok/` |
| **合成改写** | 10% | 100B | `HuggingFaceFW/finephrase` (all) | ODC-By | `finephrase_tok/` |
| **合计** | 100% | **1000B** | | | |

> Code 占比从 15% 降到 4% 系门控所迫（仅 Python 单语言可用，多喂会过拟合单语言风格）；
> 腾出的份额补到 DCLM/FineMath/FinePDFs（质量更高的可访问源）。HumanEval 提升目标改由
> 后续 mid-training 阶段（可届时申请 token 引入 the-stack-v2）承接。


### 依据（均来自调研文档）
- **DCLM 为主 web**：质量 > FineWeb-Edu（质量榜 §0.1），SmolLM2 实证 FineWeb-Edu:DCLM=40:60。
- **Code 15%**：直击当前 HumanEval pass@1 = 0% 痛点（§7.2）。
- **Math 10%**：现代通用模型标配 5–10%（§5 经验配比）。
- **FinePhrase 合成 10%**：2026 唯一「已改写好 + 大规模(486B) + ODC-By + SOTA 质量」拿来即用合成数据
  （§4.129）；合成数据在预训练中后期收益最大（BeyondWeb 经验 6）。保留原始 web 以保 commonsense/多样性。

### mid-training 池（单独，不计入 1T 主训练）
预留 **~50–100B** 高质量池，供 annealing 阶段（学习率快速 decay）喂精选 math/code/instruct
（OLMo3 Dolmino 路线，§6）。本文档主体聚焦 1T 主训练；mid-training 作为后续阶段。

---

## 3. 工程架构（核心设计决策）

### 决策：每源独立 tokenize + 训练时多源加权采样
**不**把所有源拼成单一巨型 shuffle bin。理由：
- 4TB 数据，若拼成一个 bin，每次调配比都要重新 tokenize，成本不可接受。
- 多源加权让「配比 = 一个配置参数」，调比例零成本，天然支持 mid-training 换配比。

### 数据布局
```
$SHARED/data/
  dclm_tok/           shard_0000.bin ... shard_NNNN.bin  + index.json
  fineweb_edu_tok/    (复用现有 fineweb_edu_100bt_tok + 增量补充)
  stack_v2_tok/       shard_*.bin + index.json
  math_tok/           shard_*.bin + index.json
  finephrase_tok/     shard_*.bin + index.json
```
- 每源切成 ~10–20GB 的 shard（uint32），避免单文件过大、便于断点续传。
- 每目录 `index.json` 记录：shard 文件列表、各 shard token 数、总 token 数、tokenizer 指纹、eot id。

### 数据加载器改造（`src/data.py`）
新增 **多源加权 IterableDataset**：
- 输入：`{源目录: 权重}` 映射（如 `{dclm:0.39, fineweb_edu:0.26, stack_v2:0.15, math:0.10, finephrase:0.10}`）。
- 每个 rank 按权重随机选源 → 从该源随机选 shard → memmap 连续读一个 block（seq_len+1）。
- 权重归一化；支持 `seed` 保证可复现；DDP-aware（每 rank 不同偏移）。
- 保持与现有单源 memmap 相同的 `(x, y)` 张量接口，训练循环零改动。
- 向后兼容：单源路径仍可用（退化为权重={单源:1.0}）。

### 配比配置
在 `configs.py` 或独立 `data_mix.py` 定义命名配比：
```python
DATA_MIX_1T = {
    "dclm":        0.39,
    "fineweb_edu": 0.26,
    "stack_v2":    0.15,
    "math":        0.10,
    "finephrase":  0.10,
}
```
train.py 加 `--data_mix` 参数（值为 mix 名或 JSON），默认单源保持现状。

---

## 4. Tokenize 管线（复用现有，逐源跑）

- 复用 `tokenize_data.py`：Qwen3 tokenizer（vocab 151936）、uint32 .bin、EOT=`<|endoftext|>`(151643)。
- **HF 下载铁律**（记忆）：`HF_HOME=/scratch/hf_local`（本地盘，绝不用 blobfuse 共享盘）+
  `HF_HUB_DISABLE_XET=1`。
- 改造 tokenize 脚本支持 **分片输出**（每满 N token 写一个 shard + 更新 index.json），支持断点续传
  （已完成的 shard 跳过）。
- **顺序**（大的先跑、边跑边可 smoke）：
  1. DCLM-baseline（390B，最大，parquet TB 级，网络瓶颈，后台分批）
  2. The Stack v2 / StarCoder（150B code）
  3. Math（FineMath + OWM + Proof-Pile-2，100B）
  4. FinePhrase（100B 合成）
  5. FineWeb-Edu 增量补到 260B（已有 100B）
- 每源 tokenize 完即可加入混合 smoke，不必等全部完成。

---

## 5. 落地步骤（自主实施顺序）

- **[Step 1] data loader 多源加权改造**（地基，风险最高，先做先验证）
  - 实现 `WeightedMultiSourceDataset`；单元测试（用现有 fineweb_edu + tinystories 两源验证加权采样）。
  - commit + push dev-chicm。
- **[Step 2] tokenize 脚本分片化改造** + 逐源下载 tokenize
  - 改 `tokenize_data.py` 支持 shard 输出 + index.json + 断点续传。
  - 后台跑 DCLM → code → math → finephrase → fineweb_edu 增量。
- **[Step 3] 混合配比 smoke**（8B, ~200 步）
  - 验证 loss 正常下降、各源都在被采样、tok/s 不退化。
- **[Step 4] 起 1T 正式训练**
  - `MAX_STEPS≈238000`，`--data_mix DATA_MIX_1T`，其余用已优化的最优配置
    （fused_ce + mbsz=4 + compile + GC）。
  - 断点续训（checkpoint 每 2000 步）、observability/TensorBoard。

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| DCLM 下载 TB 级、网络慢 | 分批 streaming tokenize，边下边 tokenize，不落原始 parquet；后台 + 轮询 |
| 磁盘打满（4TB .bin + HF 缓存）| 每源 tokenize 后清 HF 缓存；监控 df；shard 化便于分批 |
| 多源采样权重错误导致某源饿死 | smoke 阶段打印每源实际采样计数校验 |
| 合成数据过量退化（Kang 2025 警告）| FinePhrase 限 10%、保留原始 web，不单一 textbook 风格 |
| tokenizer 不一致 | 所有源统一 Qwen3 + 同 EOT；index.json 记指纹校验 |
| 训练中断 | checkpoint 续训；shard index 幂等 |

---

## 7. 验收标准
- data loader：多源加权单测通过，采样比例误差 <2%。
- 各源 bin：token 数达标 ±5%，index.json 完整，抽样 decode 可读。
- 混合 smoke：loss 正常（~ln(V) 起步、单调下降）、gnorm 正常、各源均被采样、tok/s ≈ 250K 不退化。
- 1T 训练：稳定推进、checkpoint 可续、无 NaN。

---

## 8. 授权合规
- DCLM = CC-BY-4.0（署名）、FineWeb-Edu/FinePDFs/FinePhrase = ODC-By、Cosmopedia = Apache-2.0、
  The Stack v2 = license 过滤宽松。**全部商用/公开友好**。
- 明确**排除** Nemotron 系（商用许可争议）。
- README/数据卡注明各源授权与署名。

---

## 9. 执行状态（并行 tokenize，实时更新 2026-07）

> 集群从 `chec-mi300-2`（4 节点）迁移到 **`chec-mi300-3`（8 节点，node-0..7，96 核/节点）**。
> 共享盘路径不变，旧数据可见。keepalive 仅需 node-0。

### 9.1 最终配比落地（plan B，已提交 `src/data_mix.py` @c954573）

| 源 | 权重 | 目标 | HF 数据集 | 目录 | 节点 |
|---|---|---|---|---|---|
| dclm | 32% | 320B | mlfoundations/dclm-baseline-1.0-parquet | dclm_tok | 0 |
| fineweb_edu | 18% | 180B | HuggingFaceFW/fineweb-edu (sample-350BT) | fineweb_edu_240bt_tok | 1 |
| finephrase | 20% | 200B | HuggingFaceFW/finephrase (all) | finephrase_tok | 2 |
| code | 15% | 150B | **bigcode/starcoderdata**（HF-gated，已授权 token，`content` 字段）| starcoder_tok | 6 |
| math | 10% | 100B | HuggingFaceTB/finemath (finemath-3plus) | math_tok | 4 |
| finepdfs | 5% | 50B | HuggingFaceFW/finepdfs-edu | finepdfs_edu_tok | 5 |

> **变更**：code 源最终确定为 `bigcode/starcoderdata`（用户已接受门控、获得 HF token），
> 取代早期"全非门控"方案。starcoderdata 已 PII 清洗/去重/质量过滤，`content` 内联，
> 流式友好；较 the-stack-v2（内容不在 parquet、需二次抓 Software Heritage）性价比更高。

### 9.2 并行架构：文件分片（file-sharding）

- 每源的 parquet 文件按 `--file_shards N --file_shard_id K` 划分给 N 个 worker，
  各写 `part_K/`，训练时 `read_index()` 自动合并 `part_*/` 子目录。
- **每源 6 分片、独占一个节点**（`RAYON_NUM_THREADS=14`，6×14=84<96 核，避免过度订阅）。
- 聚合吞吐 **~50M tok/s**，整个 1T 语料预计 **~4 小时**完成（对比单 worker/源 ~2 天）。

### 9.3 踩坑与根治

1. **48 路并发（8 节点×6 源）触发 HF API 限流（429，1000 请求/5 分钟）**——启动瞬间
   每 worker 分页 `list_repo_files` ×48 = 瞬时上千请求。根治：改**每源 1 worker、错峰
   60–90s 启动、分散节点**；稳态顺序流式请求率极低。（扩容到 6 分片时仍需分批错峰。）
2. **HF 缓存目录权限**——`/scratch/hf_local` 被 root 以 700 占用，worker 以 aiscuser
   身份写不进。根治：改用 aiscuser 自建的 `/scratch/hf_aiscuser`（各节点本地盘）。
3. **进程脱离方式**——直接经 run_remote ssh 启的 setsid 子进程会随 session 断开被杀；
   必须用「node-0 setsid 包住 `su aiscuser -c "ssh node-K 'setsid nohup ... </dev/null &'"`」
   的模式，单引号内的远程 setsid 才能存活。
4. **历史残留 worker 重复 tokenize**——早期 48-way 尝试的 `file_shards 8` 与 dispatch2
   的 `file_shards 1` 单 worker 会残留并写入重叠数据。扩容前后需清理，判据：
   `pgrep -af tokenize_data.py | grep -oE 'file_shards [0-9]+ --file_shard_id [0-9]+' | sort | uniq -c`
   应恰好为 6 行 `file_shards 6 id 0..5`；并删除残留的 `part_6/part_7` 孤儿目录。
- 补充：`index.json` 仅在每个 2B token 满 shard 时刷新，早期显示 0B 属正常，靠
  日志 `[rate]` 行判活；finepdfs 偶在分片启动瞬间再撞 429，`_par_retry.sh` 自愈重试。

### 9.4 状态快照（扩容后）

各源 6 分片就位。ETA：dclm ~3.6h、fineweb ~1.7–3.2h、finephrase ~3.1h、code ~1h、
math ~2h、finepdfs ~2h。全部 1T 约 4h 内完成。

### 9.5 后续

1. 周期性 `_status6.sh` 监控至各源达标，确认无残留分片。
2. 混合 smoke：8B ~200 步 `--data_root $S/data --data_mix mix_1t`，校验加权采样比例。
3. 启动 1T 训练：MAX_STEPS ~238000，配置 fused_ce + mbsz4 + grad_accum4 + compile
   + fp8（详见 §10；实际吞吐 ~610K tok/s / 64 GPU）。

---

## 10. 实际训练落地（2026-07，正式 1T 已启动）

> 本节记录从 tokenize 完成到 1T 正式训练启动之间的所有工程决策与踩坑根治，
> 是 §1–§9 计划的**实际执行结果**。8B/100BT 基线训练已完成，吞吐优化已完成，
> 现进入 1T 多源正式训练。

### 10.1 语料最终落地（8 源，~938B unique）

各源实际 token（`du -bc`/4 精确统计，awk %d 32-bit 会截断需避免）：

| 源 | 实际 token | 目标 | 备注 |
|---|---|---|---|
| dclm | 308B | 320B | 用户接受不补顶，训练中 ~1.15 epoch |
| fineweb_edu | 174B | 180B | |
| finephrase | 192B | 200B | 合成改写 |
| code (starcoder) | 134B | 150B | HF-gated 已授权 |
| finepdfs | 62B | 50B | |
| math (finemath) | 33.2B | 100B | |
| infimath | 21.6B | — | 追加数学源 |
| owm (open-web-math) | 13.2B | — | 追加数学源 |
| **合计** | **~938B** | 1000B | 权重采样补足到 1T |

**最终 1T mix（plan B，`src/data_mix.py`）**：dclm 35.4%、fineweb_edu 18%、
finephrase 20%、code 15%、math 3.12%、infimath 2.16%、owm 1.32%、finepdfs 5%。
dclm 354B 目标但仅 308B unique → ~1.15 epoch（可接受）。

### 10.2 训练侧重大变更

1. **fp8 默认 ON**（commit `3798ed6`）：MI300 tensorwise recipe，224/225 层转换，
   torchao 0.13，实测 **+24.5% 吞吐 / -28GB 显存**。`--fp8`/`--no-fp8`，`--fused_ce`/
   `--no-fused_ce`、compile 全部**代码内默认 ON**，启动脚本 EXTRA_ARGS 仅放业务 flag。
2. **OLMo 式 `EpochMixtureDataset`**（`76e579a`）替代 WeightedMultiSource 作为 mix 主力：
   无限 per-replica 确定性流、上来即把非本源权重清零、懒 memmap（修 64-rank 冷 blobfuse
   初始化 hang）。
3. **扩到 8 节点 / 64 GPU，plan A 保持全局 batch**（`151781b`）：grad_accum 8→4 抵消
   world 32→64。但相对**实际** 8B run（mbsz2/ga8/world32=2.10M）全局 batch **翻倍**到
   mbsz4×ga4×world64×seq4096 = **4.19M tok/step**。
4. **LR 随 batch √2 上调**（`3f08484`）：`_MODEL_OPT["8b"]` lr 2e-4→**2.8e-4**、
   min_lr 2e-5→**2.8e-5**、warmup 500→**1500**。
5. **fused CE 默认 ON**（`a14ac87`）：flash-attn Triton CE + 可用性探测 + graceful
   fallback 到 `F.cross_entropy`。
6. **checkpoint / 续训大改**（`87f73a5`）：`save_ckpt` 原子 tmp+`os.replace` + `latest`
   指针 + `_prune_ckpts`（`keep_last_ckpts` 默认 3）；`resolve_resume_path` 支持
   `latest`/裸文件名/显式路径 + graceful fresh-start；**数据快进** `resume_skip` 重放
   同一确定性选择流跳过已消费样本（per-worker = `resume_skip//num_workers`）。
7. **`--tb_dir` flag**（`23ab530`）：TensorBoard log 目录可覆盖，指向系统 TB 目录
   （`/scratch/azureml/cr/j/<id>/exe/wd`），平台自带 tensorboard 直接读取，无需自起 server。

### 10.3 数据加载踩坑与根治

1. **collate resize-storage 崩溃**：`RuntimeError: Trying to resize storage that is
   not resizable` @ `collate_tensor_fn`。`.copy()`（`b4e5603`）**无效**；真正修复是
   模块级 **`_stack_collate(batch)`**（`torch.stack`，绕过共享内存 `out=` 路径），
   接到全部 3 个 DataLoader（commit `d46a14d`）。
2. **短/空尾切片崩溃**：修完 collate 后暴露真根因
   `stack expects each tensor to be equal size, got [4096] and [0]` @ rank19。
   某 shard 的 index `ntok` **大于真实 memmap 长度**（dclm 被中断的尾 shard：index 记
   320B 但只写 308B），末尾切片越界返回短/空数组。**根治**：yield 前校验
   `chunk.shape[0] != bs1` 则 `continue`，加到 EpochMixture + WeightedMultiSource
   （commit `6609a71`）。此前误判为「node-2 硬件/代码陈旧」，实为纯数据边界 bug，
   任何 rank 都可能命中。

### 10.4 部署踩坑

- **node-0 与 node-2 `/scratch/code` 属 root**：launcher 的 aiscuser ssh 无法 git-pull
  → 停在陈旧代码（「one node runs old code」反复根因）。root/`run_remote.py`（落地为
  uid=0）可更新。**每次正式启动前必须先 root-update node-0 到目标 HEAD**。
  （node-2 实测 `.git` 属 aiscuser、可自更新，非此列。）
- `mi300_mn.sh` git-sync 改为 best-effort 非致命（`d8b4676`），避免只读 `.git` 中断
  torchrun。
- blobfuse per-node 覆盖缓存：push `_*.sh` 用新文件名（`_v1`,`_v2`…）避免读到陈旧内容。

### 10.5 Smoke 验收（200 步，8 节点 × 8 GPU）

全绿通过并正常退出：8/8 节点起 41 procs；fp8 224/225；模型 7.602B / world=64 /
FSDP2 1D；flash-attn CE ON；loss 12.77→6.06 单调下降；吞吐稳定 **~601–607K tok/s**；
显存 76.7G；`ckpt_200.pt` 已存。200 步内 non-finite grad skip 3 次（1.5%，均在 warmup
早期 LR<3e-5 段，被自动跳过保护兜住，loss 未受影响）→ 正式训练前 2000 步需盯 skip 率。

### 10.6 1T 正式训练启动状态（LOCKED）

- **启动脚本**：`_launch_1t_v1.sh`（node-0 setsid fan-out node-0..7，nohup torchrun
  --nnodes=8）。
- **配置锁定**：model 8b，seq 4096，**mbsz 4 / grad_accum 4 / world 64**，global batch
  **4.19M tok/step**，**max_steps 238000**，LR 2.8e-4 / min_lr 2.8e-5 / warmup 1500，
  fp8 + fused_ce + compile 默认 ON，`--data_mix mix_1t --data_root $S/data`，
  `--resume latest --keep_last_ckpts 3`，`--tb_dir /scratch/azureml/cr/j/<id>/exe/wd`。
- **启动确认**：8 节点全起，loss 12.77→9.12，吞吐 **~607–612K tok/s**，skip=0，无
  traceback，TB log 已写入系统目录。**ETA ~456h ≈ ~19 天** wall-clock（可断可续）。
- **监控计划**：前 ~2000 步（覆盖 warmup 到 peak LR 2.8e-4）盯 skip 率；若 >5% 则停下
  把 peak LR 降到 ~2.4e-4 再 `--resume latest` 续训。

### 10.7 与原计划的差异小结

| 项 | 原计划（§1–§9） | 实际落地（§10） |
|---|---|---|
| 平台 | 32×MI300（4 节点） | **64×MI300（8 节点）** |
| 全局 batch | 4.2M（mbsz4/ga8/world32 计划） | 4.19M（mbsz4/**ga4**/world64） |
| LR | 未定 | **2.8e-4**（√2 上调）/ warmup 1500 |
| 吞吐 | 250.8K tok/s | **~610K tok/s**（fp8 + 64 GPU） |
| 精度 | bf16 | **fp8**（默认 ON）+ bf16 param |
| 源数 | 5–6 源 | **8 源** ~938B unique |
| loader | WeightedMultiSource | **EpochMixtureDataset**（OLMo 式） |
| 续训 | checkpoint 每 2000 步 | + latest 指针 + 数据快进 + ckpt 轮转 |
| ETA | ~46 GPU-天 | ~19 天 wall（64 GPU 并行） |
