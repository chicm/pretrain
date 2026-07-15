# FP8 训练优化结果（MI300X, ROCm 7.1）

分支：`dev-fp8`。集群：`chec-test-env`（单节点 8× MI300X VF, ROCm 7.1.0, 显存 190G/卡）。
栈：torch 2.10-dev+rocm7.1 / **torchao 0.15.0** / flash_attn 2.8.3 / triton 3.4（`/opt/venv`，系统 python 无 torch，务必用 `/opt/venv/bin/python`）。
模型：Chimera 8B（也用 1B 做正确性验证）。数据：真实 FineWeb `fineweb_tok`（10.2B tok, uint32, vocab 151643, EOT 151643）。

---

## TL;DR

- **FP8 在 8B 上是净收益**：真实数据端到端 **+25% 吞吐、−28G 显存**，loss 曲线与 bf16 重合、无 NaN。
- **旧「FP8 无收益」结论（ROCm 6.4.3, torchao 0.13）是测试方法漏洞**，不能当成硬件/kernel 极限。
  旧测试同时踩了两个坑（下节），正确组合从没测过。
- 生效的两个关键修复：**(1) fnuz dtype** + **(2) torch.compile 强制**。缺一不可。
- **甜点配置 = 8B, mbsz=4, tensorwise, --fused_ce, torch.compile**（mbsz=6 触发 compile 病态，不实用）。

---

## 背景：旧「FP8 无收益」为什么不算数

早前在主力 MI300 集群（ROCm 6.4.3 + torchao 0.13，见 memory `8b-throughput-optim-2`）测 fp8 拿到
**1.00×**（零收益），当时归因「ROCm6 软件栈不成熟」。但复盘发现那次测试**同时**有两个漏洞，
导致 fp8 gemm 实际上根本没走上高速路径 —— 所以它证明不了「硬件不行」，只证明了「那样测不行」：

### 漏洞 1：dtype 用了 OCP e4m3fn，MI300 不支持
FP8 有两套编码：
- **OCP 标准**：`float8_e4m3fn` / `float8_e5m2`（NVIDIA H100 用这套）
- **fnuz 变体**：`float8_e4m3fnuz` / `float8_e5m2fnuz`（"fnuz" = finite, no -0；AMD CDNA3 用这套）

MI300 的 hipBLASLt **只支持 fnuz**。喂 OCP `e4m3fn` 会直接 `HIPBLAS_STATUS_NOT_SUPPORTED`，
落回慢路径或报错。**这一点 ROCm 6 和 7 都一样**，不是版本差异。
新 torchao（0.15）靠 `torchao.utils.is_MI300()==True` 自动把 dtype 切成 fnuz；旧 torchao 0.13
是否也会自动切、当时是否生效，**从未验证**。

### 漏洞 2：跑的是 eager，没开 torch.compile
FP8 的 gemm 本身快，但**量化/反量化（quant + scale）**是一堆小算子。eager 模式下这些开销散落、
不融合，整体反而**比 bf16 慢 ~2×**。必须靠 `torch.compile`（Inductor）把 quant/scale 融进
gemm 前后的 epilogue/prologue，才能把 fp8 gemm 的速度真正兑现出来。旧测试是 eager 跑的 → 踩坑。

**结论**：旧的 1.00× 是「eager + e4m3fn」的产物，是方法漏洞，不是硬件极限。是否真是「ROCm6 版本问题」
需要在主力集群单独用 fnuz+compile 重测才能一锤定音（见文末待办）。

---

## 微基准（node-0 单卡, HIP_VISIBLE_DEVICES=0）

### 裸 `torch._scaled_mm`（`_probe_fnuz.py`）
直接测底层 fp8 gemm 原语，绕过所有框架层：

| dtype | 结果 |
|---|---|
| e4m3fn（OCP）| ❌ `HIPBLAS_STATUS_NOT_SUPPORTED` |
| **e4m3fnuz** | ✅ ~1.1 PFLOP/s，真实 MLP shape 上 **1.7–1.9× over bf16** |

说明 MI300 的 fp8 gemm 硬件本身很快，前提是喂对 dtype。

### torchao Float8Linear（SwiGLU MLP, d=4096 h=14336, M=8×4096 ≈ 8B 单层形状）
测「一层完整 fp8 Linear」（含 quant/scale/gemm 全链路），这才是训练里实际跑的东西：

| 配置 | ms/it | vs bf16 | 备注 |
|---|---|---|---|
| bf16 | 37.7 | 1.00× | 基线 |
| bf16 + compile | ~37 | ~1.0× | bf16 本身 compile 几乎无收益 |
| fp8 tensorwise **eager** | 83.7 | **0.45×** | ← **eager 陷阱**，旧测试踩的就是这里（比 bf16 还慢一半） |
| **fp8 tensorwise + compile** | **24.2** | **1.56×** ✅ | **最快**，推荐配置 |
| fp8 rowwise + compile | 27.2 | 1.39× | rowwise scale 精度更高但略慢 |

**关键对比**：同样是 fp8 tensorwise，eager 0.45× vs compile 1.56× —— 差 3.5 倍，完全由 compile 决定。
这就是为什么 `--fp8` 在代码里**强制**要求 compile（不开直接 SystemExit）。

### recipe 选择：tensorwise vs rowwise
- **tensorwise**：整个 tensor 一个 scale。最快（1.56×），精度略低，8B 实测 loss 无异常 → **默认用它**。
- **rowwise**：每行一个 scale。精度更高、更稳，略慢（1.39×）。若将来长跑发现精度问题再切。

---

## 端到端验证（8×MI300X 单节点）

### 1B — 正确性验证（不看吞吐）
- fp8 loss 与 bf16 **逐位一致**（step0 12.348 vs 12.349，差异在数值噪声内）
- 无 NaN，全程稳定
- 显存更低（17.8 vs 20.2G）
- **168/169 Linear 转换**，lm_head 正确跳过
- 1B 的 gemm 太小，量化开销占比高 → **吞吐无增益（符合预期）**。1B 只用来证明「转换正确、数值无损」。

### 8B @ mbsz=4 — 权威吞吐数字 ✅
真实 FineWeb 数据、`--fused_ce`、torch.compile、tensorwise fp8：

| 配置 | tok/s | 峰值显存/卡 | loss / NaN |
|---|---|---|---|
| bf16 | 67.5K | 118.0 G | 基线 |
| **fp8** | **84.4K** | **90.0 G** | 曲线与 bf16 重合, 无 NaN |
| **收益** | **+25% 吞吐** | **−28 G 显存** | — |

- **224/225 Linear 转换**（lm_head 跳过）
- 8B 才体现规模效应：gemm 足够大，fp8 gemm 加速盖过量化开销（微基准就是 8B 单层 MLP shape）
- **这是本次工作的权威结论数字**

### 8B @ mbsz=6 — 只拿到显存数据，吞吐未取到
尝试加大 batch 看 fp8 省显存能否换更大 batch：

| 配置 | step-0 峰值显存/卡 | 状态 |
|---|---|---|
| bf16 | **160.2 G** | 逼近 190G 上限，勉强放下 |
| **fp8** | **118.3 G** | 省 **~42 G**，仍有大余量 |

- **稳态 tok/s 未取到**：bf16 与 fp8 **都卡死在 step-10 的 torch.compile recompile**
  —— 出了 step 0 后 ~30 分钟零推进，CPU ~159%（编译 worker 在忙 autotune），GPU 空转。
- **两组症状完全一致** → 是 **mbsz=6 + 8B 的 compile 路径病态**（某个守卫在 step 10 失效触发全量重编，
  8B 大 gemm 的 Triton autotune 单次就要十几分钟），**与 fp8 无关**。
- 结论：**mbsz=6 这条配置 compile 不实用**，权威数字用 mbsz=4。显存数据仍有价值：证明 fp8 省 ~42G，
  理论上可支撑更大 batch（若解决 recompile 问题）。

> 关于 recompile：`torch.compile` 缓存每份编译产物时带一组「守卫（guards）」。当某步输入 shape /
> dtype / 控制流违反守卫，Dynamo 判缓存失效 → Inductor 从头重编（含极慢的 Triton autotune）。
> 若每隔几步就触发，训练时间几乎全耗在编译上，表现为「卡死」。诊断可挂 `TORCH_LOGS=recompiles`
> 看是哪个守卫失效；根治靠固定所有输入 shape（定长 seq、drop_last、固定 micro_bsz）或 `dynamic=` 显式设定。

---

## 代码（dev-fp8 分支）

- **`src/fp8_utils.py`**：`convert_model_to_fp8(model, recipe='tensorwise'|'rowwise')`
  - 必须在 `fully_shard` / `torch.compile` **之前**调用
  - 只转 attn/MLP 的大 Linear；跳过 lm_head（词表维度大且对精度敏感）
  - 要求 in/out 特征数 16 对齐（fp8 gemm 约束），不满足的层跳过
  - 转换后打印：`[fp8] recipe=tensorwise converted N/M Linear layers (is_MI300=True; dtype=fnuz). torch.compile REQUIRED for speedup.`
- **`src/train.py`**：新增 `--fp8` / `--fp8_recipe {tensorwise,rowwise}`
  - `--fp8` 会**强制要求** torch.compile 开启，否则直接 SystemExit（避免 eager 陷阱）

**Commits**（dev-fp8）：`4c70517`（fp8 代码） → `893e9ac`（docs 初版） → `06d46c8`（8B mbsz4/6 结果）。
不影响 `dev-chicm` / `main`。

---

## 操作经验（chec-test-env，供复现）

- **SSH tunnel（run_remote.py）频繁在建连时 drop**：命令通常仍执行，但返回丢失、后台任务常来不及 detach 就被杀。
  最稳启动法：`nohup /启动器.sh & exit 0`，启动器内部再 `setsid bash 真job >log 2>&1 </dev/null &; exit 0`
  —— 父 shell 秒退，SSH drop 杀不到孙进程。（容器里没有 `at` / `systemd-run`。）
- **CRLF**：从 Windows push 上去的 `.sh` 带 `\r` → `bash: cd $'/scratch\r'` 报错。启动前必
  `sed -i 's/\r//g' 脚本 && bash -n 脚本` 验证语法。
- **输出缓冲**：`torchrun ... | grep | tail` 会 buffer 到进程退出才 flush（看不到进度）→ 改
  `PYTHONUNBUFFERED=1` 直接写 logfile，别用中间管道。
- **run_remote 30s 超时 + 长 compile**：后台跑 + 分次 poll；远端命令末尾加 `| tr -cd '[:print:]\n'` 滤掉进度条特殊字节（否则本地 cp1252 解码报错）。

---

## 待办

- [x] 微基准（fnuz `_scaled_mm` + torchao Float8Linear）→ compile 后 1.56×。
- [x] 1B 正确性（loss 逐位一致、无 NaN、转换正确）。
- [x] 8B @ mbsz=4 端到端 A/B（真实数据）→ **+25% 吞吐, −28G 显存**。
- [x] 8B @ mbsz=6 显存对比（bf16 160G vs fp8 118G）；吞吐因 compile recompile 未取到。
- [ ] **主力 MI300 集群（ROCm 6.4.3）单卡重跑 `_probe_fnuz.py`** —— 一锤定音「版本问题 vs 方法问题」：
  若 ROCm6 上 fnuz `_scaled_mm` 也快 → 生产集群现在就能用 fp8，不必等升级 ROCm7。
  （等主力集群数据下载完成；用户指示只跑这个微基准、不做别的；job 名待用户提供。）
- [ ] fp8 长跑 loss 曲线 vs bf16（收敛性确认后再进 `mi300_mn.sh` 默认配置）。
- [ ] （可选）诊断 mbsz=6 的 recompile 触发点（挂 `TORCH_LOGS=recompiles`），若能固定 shape 消除重编，
  fp8 的 −42G 显存可换更大 batch，进一步提吞吐。
