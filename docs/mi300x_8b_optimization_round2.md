# 8B MI300X 训练吞吐优化 — 第二轮记录

> 承接 `docs/mi300x_8b_throughput_results.md`（第一轮，colleague 达 250.8K tok/s）。
> 本轮针对**当前实际运行的 8B 训练**继续优化，或证明已达软件栈物理上限。
> 工作管理：`_remote/OPT_WORKPLAN.md`。

## 0. 起点与方法

### 硬件/软件（待现场复核版本）
- 集群 `chec-mi300-2`，4 节点 × 8× MI300X-192GB，节点间 InfiniBand，world=32。
- Job `olive_juice_kk7hjfwb0k`，conda env `py_3.10`，torch 2.8.0a0 + ROCm 6.4。

### 第一轮结论（继承）
- **最优配置**：fused_ce + micro_bsz=4 + torch.compile + fp32-reduce + 1D-FSDP2 → **250.8K tok/s**（+33% vs 189K 基线）。
- MFU ≈ 40–55%（每卡 ~358 TFLOP/s，MI300X bf16 峰值 ~650–990）。判定 **compute-bound**。
- 通信类优化（HSDP/bf16-reduce）净收益 ≈ 0；显存类（AC/SAC）净收益为负或次优。
- 唯一物理破点：**FP8**（doc §7 被 ROCm 6.4 hipBLASLt `NOT_SUPPORTED` 卡死，推迟到 ROCm 7.x）。

### 方法
- 单变量 A/B，短 smoke（MAX_STEPS≈80），tok/s 取稳态（跳过 step 0 编译/warmup）。
- 以 `rocm-smi` GPU use%/power 判活；blobfuse 日志有分钟级缓冲，日志滞后≠卡死。
- smoke 在同 job 内串行；跑完立即恢复长跑，避免 GPU 空转。

## 1. 实验记录

（下面按任务追加）

### T1 — 稳态复测当前 live run（完成，发现重大配置 bug）
目的：确认当前 8B 长跑真实稳态 tok/s。
数据（steady, step 10-40）：
- tok/s 稳定 **~204K**，显存 **173.1G**，GPU use 全 97%，power 正常。
- 版本：torch 2.8.0a0+gitd06a406，**ROCm 6.4.3**，HIP 6.4.43484。

**关键发现**：`ps -ef` 显示实际命令行含 **`--no_compile`** —— torch.compile 根本没开！
- 对照文档吞吐表：no_compile 档 = 210K/150G；compile(修复后) 最优 = **250.8K/106.5G**。
- 我的 204K/173G 正落在「未编译」档位 → **有 ~23% 吞吐 + ~66G 显存的白白损失**。
- 根因：远程 `/scratch/code/mi300_mn.sh` 的 `COMPILE_FLAG` 默认值 = `--no_compile`
  （train.py 逻辑：不传 `--no_compile` 则 `cfg.compile=True` 默认开启）。
  我启动时未显式覆盖 → 跑成了未编译。

**结论**：当前长跑不是最优配置。修复 = 显式 `COMPILE_FLAG=""`（空=不传=compile ON）重启。
- ROCm=6.4.3 → FP8 (T9) 确认仍受 hipBLASLt 限制，保持推迟。

### T1-fix — 重启 live run 开启 torch.compile（完成 ✅）
操作：pkill 全部旧进程 → `setsid` detach 重启，显式 `COMPILE_FLAG=""`（compile ON）+ `--fused_ce`。
核验命令行：`train.py --model 8b ... --micro_bsz 4 --grad_accum 8 --fused_ce`（无 `--no_compile`）。
结果（steady step 10-20）：
- tok/s **204K → 247.5K（+21%）**
- 显存 **173G → 106.5G**（省 66G）
- GPU use 97%，4×41 进程存活，loss 正常下降无 NaN。
- **已复现文档最优基线 250.8K/106.5G**（247.5K 在测量抖动内）。

**教训**：`--no_compile` 是最大的隐形吞吐杀手；启动后必须 `ps -ef | grep train.py` 核验实际命令行，不能只信 env。
新基线 OUT/TB = `.../exe/wd/pretrain_8b_c`。

---

## 2. P1 优化项（在 247.5K 最优基线上继续压榨）

### T3 — TORCHINDUCTOR_MAX_AUTOTUNE_GEMM 实验（第一次尝试失败，已回退）
侦察（running proc env）：
- flash_attn **3.0.0**（FA3 在位）；liger_kernel **未装**。
- hipBLASLt 已启用（`NVTE_USE_HIPBLASLT=1`），但**无 inductor autotune、无 GEMM 调优 env**。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 平台不支持，静默忽略。
- ROCm 6.4.3 / gfx942 / torch 2.8.0a0。

假设：compute-bound 且 GEMM 占主导 → inductor GEMM 自动调优可能再挤出几个 %。
做法：复制 launcher 为 `SHARED/mi300_mn_opt.sh` 注入 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM`（远程 env 不会透传到 worker，必须写进 launcher 内的远程 export 段）。

**⚠️ 事故**：第一次 T3 启动脚本把 `ssh pkill`（杀旧进程）和 `setsid`（拉新进程）
放在同一个 rexec 脚本里。ssh 会话相关的信号/清理把 setsid 的启动也带走了 →
**旧的杀了、新的没起来 → 4 节点 PROC=0 → GPU 空转**（约数分钟）。
立即用 `revive.sh`（**只 spawn、不 kill**）拉回最优基线，step10=247.0K/106.5G 恢复。

**硬教训（写入记忆）**：
1. **kill 与 spawn 必须是两个独立的 rexec 调用**，中间用一次纯查询确认 PROC=0，再单独 spawn。
2. spawn 脚本内**绝不能含 ssh pkill**——会连累 setsid detach。
3. 每次重启后必须 `ps -ef|grep train.py` 核验命令行 + 4 节点 PROC 计数。

### T3-retry / 剩余项 — 决策：保持最优基线，不为小概率增益赌上长跑

在恢复到 247K 最优基线后，重新评估剩余优化项的风险/收益：

| 项 | 预期增益 | 风险/成本 | 决策 |
|---|---|---|---|
| ✅ 修复 `--no_compile` | **+21%（已拿下）** | — | 已完成 |
| T3 max-autotune GEMM | 0–3%（compile 后 GEMM 调优空间小） | 编译 5→20-30min；每次重启有空转风险 | **暂缓** |
| T6 Liger CE | ≈0（已有 fused_ce + FA3） | 装包+改代码 | 暂缓 |
| T7 自定义融合 kernel | 未知，可能几% | 高（写 Triton，易引 NaN/bug） | 暂缓 |
| T9 FP8 | 潜在 +30-50% | ROCm 6.4.3 hipBLASLt 卡死，物理阻塞 | **等 ROCm 7.x** |

**核心判断**：文档结论 compute-bound / MFU 40-55% 是硬约束；已修复的 compile bug 是本轮唯一的大杠杆，已拿下。剩余项都是「小概率增益 + 需反复重启最优长跑」，而重启刚导致过一次 GPU 空转事故。因此**不再为 0-3% 的投机增益赌上已验证的最优长跑**。

**若未来要继续压榨**（建议在专门的短 smoke 环境、且不影响正式长跑时做）：
1. 优先级：T3 autotune（最干净）→ T6 Liger → T7 自定义融合。
2. 严格遵守「kill 与 spawn 分开、spawn 脚本不含 ssh pkill」的重启铁律。
3. 真正的物理破点是 FP8，等 ROCm 7.x / hipBLASLt 成熟后重开 T9。

## 3. 本轮最终结论

- **交付**：发现并修复 live 8B 训练误跑 `--no_compile` 的隐形 bug，吞吐 **204K → 247K tok/s（+21%）**、显存 **173G → 106.5G（省 66G）**，复现并锁定文档最优基线 250.8K/106.5G。
- 当前 8B 长跑运行在最优配置：`--model 8b --micro_bsz 4 --grad_accum 8 --fused_ce` + compile ON + fp32-reduce，OUT/TB = `.../exe/wd/pretrain_8b_c`，32 卡 97% 满载。
- 该软件栈（ROCm 6.4.3）下已达实用最优；进一步提升需 FP8（等 ROCm 7.x）。
