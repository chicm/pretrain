# 8B MI300X 吞吐优化 —— A/B 实测结果详细报告

本文记录对 `docs/mi300x_8b_pretraining_optimization.md` 优化指南中每一个候选优化项，在
**8B Chimera / MI300X ×32（4 节点 × 8 GPU，Infiniband 互联）** 上逐项 A/B 实测的完整过程、原始
数据与分析结论。

---

## 0. 实验环境与方法

### 硬件 / 软件栈
- **集群**：`chec-mi300-2`，VC `webxt-webdata-mi200`，eastus2，SKU `Singularity.ND96isr_MI300X_v5`。
- **拓扑**：4 节点（node-0..3），每节点 8× MI300X（192GB HBM3），节点间 Infiniband。world_size=32。
- **软件**：torch `2.8.0a0+gitd06a406`，ROCm 6.4.3，RCCL 2.22.3，torchao `0.13.0+git262b180ce`，
  conda env `py_3.10`。

### 模型（被测负载）
- Chimera 8B = 7.602B 参数：dim 4096 / 32 层 / 32 Q-head / 8 KV-head（GQA）/ SwiGLU 中间维 14336 /
  seq_len 4096。Tokenizer Qwen3（vocab 151936 padded）。dropout=0。
- 模型状态（fp32 master + bf16 params + Adam m/v）≈ 106GB，1D-FSDP2 下每卡持 106/32 ≈ 3.3GB。

### 基线配置（189K tok/s）
`no-AC, micro_bsz=2, grad_accum=8, fp32-reduce, 1D-FSDP2, no torch.compile`。
- global batch = 2 × 8 × 32 × 4096 ≈ 2.1M tokens/step。
- 稳态显存 89.3GB/卡（192GB 上限，余量充足）。

### A/B 方法
- 每次 A/B 用 `MAX_STEPS=80` 短 smoke，单变量对照，其余参数与基线一致。
- **tok/s 读数取稳态段**（跳过 step 0 编译/warmup），多次读数确认一致后记录。
- **判活以 `rocm-smi` GPU use%/power 为准**，不看 log tail —— blobfuse 对训练 stdout 有分钟级缓冲，
  日志滞后不代表进程卡死。
- OOM 判定：ROCm 上单卡 OOM 会导致该 rank 退出、其余 rank 在 all-gather 上挂起，表面报
  `Watchdog collective timeout` / `SeqNum=... _ALLGATHER_BASE` —— 实为**伪装的 OOM**，需用单节点
  `torchrun --standalone` 隔离复现真实 `OutOfMemoryError`。

---

## 1. micro_bsz（最大单项收益，+6.9%）✅ 采纳

### 动机
计算瓶颈下，增大 per-GPU micro-batch 会把每个 GEMM 的 M 维（token 数）做大，提升算术强度
（arithmetic intensity）与 GPU 利用效率；同时 grad_accum 步数减少 → 梯度累积期间的通信被摊薄。

### 原始数据
| micro_bsz | tok/s | vs 基线 | 稳态显存 | 结果 |
|---|---|---|---|---|
| 2（基线） | 189K | — | 89.3G | ✅ |
| **3** | **~202K** | **+6.9%** | 131.2G | ✅ **最优** |
| 4 | — | — | ~180G → **OOM** | ❌ |

### 分析
- mbsz=2→3 单步就 +6.9%，是所有优化里**最大的单项收益**，远超算子融合类。印证「计算瓶颈下算术
  强度是第一杠杆」。
- mbsz=3 显存 131.2G，距 192G 上限还有 ~60G 余量，安全。
- mbsz=4 显存冲到 ~180G 后 OOM（激活量随 mbsz 线性增长）。故 **mbsz=3 是不开 AC 时的显存天花板内
  最优点**。
- **副作用**：global batch 从 2.1M 变为 3.15M tokens/step（1.5×）。若要严格保持 token/step，需把
  grad_accum 从 8 调到 ~5（3×5×32×4096 ≈ 2.0M）。本项目最终选择保留 grad_accum=8（用户决定，接受
  更大 global batch）。

---

## 2. torch.compile（零代价正收益，+1.2%）✅ 采纳

### 动机
算子审计发现：最重的两类 FLOPs 已是 fused 实现，但若干中小算子为手写、未融合，torch.compile 可
自动 fuse 掉这些逐元素 kernel、减少 kernel launch 与 HBM 往返。

### 算子融合审计表
| 指南要求 | 现状 | 说明 |
|---|---|---|
| FlashAttention-2 / fused SDPA | ✅ 已有 | `F.scaled_dot_product_attention(is_causal=True)`，ROCm 自动派发 AOTriton FlashAttn-2 |
| Fused AdamW | ✅ 已有 | `torch.optim.AdamW(fused=True)` |
| Fused RMSNorm | ⚠️ 半 | 手写 `x.float()*rsqrt(mean(x²)+eps)`，多 kernel 未融合 |
| Fused RoPE | ⚠️ 半 | 手写 split-half（NeoX）cos/sin，多 kernel |
| Fused SwiGLU | ⚠️ 半 | `w2(silu(w1(x))*w3(x))`，silu 与逐元素乘分开 kernel |
→ 最重的 SDPA / AdamW 已 fused（占绝大多数 FLOPs），剩下 RMSNorm/RoPE/SwiGLU 是 torch.compile 的
自动融合目标。

### 实现
`mi300_mn.sh` 原硬编码 `--no_compile`；改为 `COMPILE_FLAG=${COMPILE_FLAG:---no_compile}` 可配，空值
即启用 compile。train.py 在 `fully_shard` 之后调 `torch.compile(model)`。commit `5c133bd`。

### 原始数据（no-AC, mbsz=2 对照）
| 指标 | no_compile | torch.compile ON |
|---|---|---|
| tok/s | 189K | **191.2K（+1.2%）** |
| 稳态显存 | 89.3G | 89.3G（持平） |
| 精度（gnorm/NaN） | 正常 | 正常，0 NaN |

### 分析
- +1.2%，收益不大但**零代价**：显存持平、无精度风险，仅首步一次性编译开销（后续步 warm）。
- 是唯一「零代价正收益」的算子融合路径，采纳并设为默认 ON。
- **注意**：AC + torch.compile 组合在 checkpoint 边界可能产生 graph break，本轮 AC 测试均在
  `--no_compile` 下单独进行以隔离变量。

---

## 3. Selective Activation Checkpointing (SAC，+4%，仍次于 mbsz=3）❌ 留 flag（最佳 AC 变体）

### 动机
Full AC 无脑重算整个 Block（含昂贵的 attention），重算成本过高。SAC 按算子成本分级：**保存**昂贵的
matmul/SDPA 输出（不重算），只**重算**便宜的逐元素/norm/RoPE，理论上以少得多的重算换取显存下降，
从而能开更大 micro_bsz。

### 实现（commit `c99c645`）
```python
_save_ops = {aten.mm, aten.addmm, aten.bmm,
             aten._scaled_dot_product_flash_attention,
             aten._scaled_dot_product_efficient_attention}
def _sac_policy(ctx, op, *a, **k):
    return MUST_SAVE if op in _save_ops else PREFER_RECOMPUTE
checkpoint_wrapper(layer, checkpoint_impl=NO_REENTRANT,
                   preserve_rng_state=False,
                   context_fn=create_selective_checkpoint_contexts(_sac_policy))
```
- `preserve_rng_state=False`：dropout=0，无需保存/恢复 RNG 状态，省开销。
- configs.py 加 `selective_ac` 字段；`--selective_ac` 会同时置 `activation_checkpoint=True`。
- 保留 full-AC 路径（`--activation_checkpoint` 不带 selective）。

### 原始数据
| 配置 | tok/s | vs 基线 | 稳态显存 | 结果 |
|---|---|---|---|---|
| no-AC mbsz=3（最优参照） | 202K | +6.9% | 131G | ✅ |
| **SAC mbsz=6** | **196.4 / 196.2K** | **+4%** | 139G | ✅ |
| SAC mbsz=12 | — | — | 168G 已占 + 要 27.8G → **OOM** | ❌ |
| full-AC mbsz=12（对照，见 §5） | 175K | −7% | 124.6G | ✅ |

启动日志确认：`[ac] SELECTIVE activation checkpointing ON for 32 blocks (save matmul/SDPA,
recompute elementwise)`。

### 分析
- **SAC 显著优于 full-AC**：SAC mbsz=6 (196.4K) 比 full-AC mbsz=12 (175K) 快 **+12%** —— 有力验证
  指南「只重算便宜算子」的核心洞察。full-AC 把 attention 也重算掉是巨大浪费。
- **但 SAC 显存节省有限**：因为保存了所有 matmul/SDPA 输出，激活占用仍大。SAC 在 mbsz=6 时显存已
  139G，**mbsz=12 直接 OOM**（要再 27.8G 而卡上已占 168G）。故 SAC 可用上限约 **mbsz=6**，无法像
  full-AC 那样把 mbsz 推到很高。
- **仍打不过 no-AC mbsz=3**：SAC mbsz=6 (196.4K) 比 no-AC mbsz=3 (202K) 慢 **2.8%**。根因还是计算
  瓶颈——SAC 即便只重算便宜算子，那部分重算 FLOPs 仍是**净增计算量**，而 no-AC 零重算。当显存够用
  能容纳 mbsz=3 时，任何重算都是亏的。
- ⚠️ **观察到的风险点**：SAC 运行时**早期 gnorm 持续为 inf**（step 10/20 均 inf，被 NaN grad-skip
  guard 跳过），而 no-AC 训练早期为有限值。此间 loss 仍正常下降（12.7→10.1→9.5），说明训练在推进，
  但这是 SAC 重算路径可能引入的数值行为差异。**若将来真用 SAC 实训，须先确认该 inf 会随 warmup 消失、
  不影响最终收敛**，否则有隐患。
- 结论：SAC 是**所有 AC 变体里最好的**，作为 `--selective_ac` flag 保留，供未来 10B+/显存真正吃紧
  （无法容纳理想 mbsz）时使用。

---

## 4. bf16 gradient reduce（+1%，精度险）❌ 留 flag

### 动机
FSDP2 默认梯度 all-reduce 用 fp32。改用 bf16-reduce 可将梯度通信量减半，理论上加速通信。

### 实现
`--reduce_bf16` flag（commit `dffe5f9`）：仅改 `MixedPrecisionPolicy(reduce_dtype=bf16)`，即只影响
FSDP2 的梯度 all-reduce；Adam m/v、fp32 master、loss reduction、grad accum 全部仍为 fp32。

### 原始数据
| 指标 | fp32-reduce（默认） | bf16-reduce |
|---|---|---|
| tok/s | 189K | **191K（+1%）** |
| 稳态显存 | 89.3G | 89.3G |
| gnorm / NaN | 正常 | 正常，0 NaN |

### 分析
- 仅 +1%。因为 8B 是计算瓶颈，通信本已被计算掩盖，减半通信量几乎不反映在端到端吞吐上。
- **代价是精度风险**：bf16 做 32-way（world=32）的梯度求和，尾数只有 8 位，训练后期梯度极小时求和
  误差累积可能触发不稳定。考虑到本项目 8B 有 NaN 历史（step 110 曾 NaN，靠 3 项修复才稳），这 +1%
  不值得冒 32-way bf16 求和的精度风险。
- 结论：**默认保持 fp32-reduce**，flag 保留供未来通信瓶颈（更大规模/弱互联）场景。

---

## 5. Full Activation Checkpointing（−7%）❌ 不用

### 动机
指南列为标准显存优化：每个 Block 非重入 AC，激活显存大降，理论上可换更大 micro_bsz。

### 实现（commit `ef452fd`）
每个 Block `checkpoint_wrapper(layer, NO_REENTRANT)`，在 `fully_shard` **之前** wrap。

### 原始数据
| 配置 | tok/s | vs 基线 | 显存 | 结果 |
|---|---|---|---|---|
| no-AC mbsz=2（基线） | 189K | — | 89.3G | ✅ |
| AC mbsz=12 | 175K | **−7%** | 124.6G | ✅ |
| AC mbsz=16 | — | — | ~210G → **OOM** | ❌ |

### 分析
- 即便 AC 把 mbsz 推到 12，吞吐反而 **−7%**：backward 全 Block 重算（含 attention）的计算开销 >
  大 batch 带来的算术强度收益。在计算瓶颈下，重算是纯粹的净增 FLOPs。
- 额外副作用：`checkpoint_wrapper` 给 state_dict 加 `_checkpoint_wrapped_module.` 前缀，**破坏与
  AC-off checkpoint 的兼容性**（污染 ckpt 键名）。
- **旁证**：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在 ROCm/HIP 上**不被支持**（静默忽略，
  `HIPAllocatorConfig.h:36`），无法用它缓解碎片。commit `9d67e6a` 保留该 env 但对 ROCm 无效。
- 结论：8B **不用 full AC**（更慢 + ckpt 前缀污染）。SAC（§3）是其严格更优的替代。

---

## 6. HSDP（Hybrid Sharded Data Parallel，0%）❌ 留 flag

### 动机
把 1D 全分片改为 2D mesh（节点内 shard、节点间 replicate），减少跨节点 all-gather，理论上利于通信
瓶颈场景。

### 实现
`--hsdp_shard N`（commit `17bec2b`）：
`init_device_mesh("cuda", (replicate, N), mesh_dim_names=("replicate","shard"))` 传入 `fully_shard`。

### 原始数据
| 配置 | tok/s | vs 基线 | 显存 |
|---|---|---|---|
| 1D-FSDP2（默认） | 189K | — | 89.3G |
| HSDP shard=8 × replicate=4 | 189.8K | **0%** | 100.7G（+11.4G/卡） |

### 分析
- 吞吐完全无变化（0%），但**每卡显存 +11.4G**（89.3→100.7G），与预测精确吻合：模型状态从 /32
  变为 /8（106GB 模型状态 /8 而非 /32，每卡多存 ~10G）。
- 无收益的原因：8B 计算瓶颈，prefetch 已把 all-gather 与计算重叠，减少跨节点通信不反映在吞吐上；
  反而白付显存代价。
- 结论：1D-FSDP2 保持默认，HSDP flag 保留供 30B+/通信瓶颈未来场景。

---

## 7. FP8（torchao Float8Linear，零加速）❌ 推迟

### 动机
FP8 是**唯一直接攻击 GEMM 计算瓶颈**的方向（MI300X 硬件支持 FP8，理论峰值 ~2× bf16）。分层策略：
仅对 7 个大 Linear（Q/K/V/O + gate/up/down）用 FP8，RMSNorm/Softmax/RoPE/loss/optimizer 保 bf16/fp32。

### 方法：微基准探路（不碰训练代码）
在单 GPU 上用 `_fp8_probe.py` / `_fp8_probe2.py` 直接测 GEMM 加速比，规避盲改训练代码的风险。

### 原始数据
| 探测 | 结果 |
|---|---|
| fp8 dtype 可用性 | ✅ `torch.float8_e4m3fn` 等可创建 |
| `torch._scaled_mm` 直调 | ❌ **`HIPBLAS_STATUS_NOT_SUPPORTED`** |
| torchao `convert_to_float8_training` import | ✅ 可用 |
| Float8Linear qkv/o（4096→4096, M=8192） | **0.98×**（略慢） |
| Float8Linear gate/up（4096→14336） | **1.00×** |
| Float8Linear down（14336→4096） | **1.00×** |
（eager 与 compiled 两种模式一致）

### 分析
- **FP8 加速比 = 0.98–1.00×，即零加速甚至略慢**。根因不在算法而在软件栈：torch 2.8 / ROCm 6.4.3 的
  hipBLASLt FP8 kernel 尚未成熟，`_scaled_mm` 路径直接 `NOT_SUPPORTED`，torchao 走通了但底层 GEMM
  没有真正跑在高效 FP8 kernel 上。
- **分层策略本身正确**，方向也对（唯一攻计算瓶颈的路径），但被软件栈卡死。
- 结论：**推迟到 ROCm 7.x / 更新 torch 且 hipBLASLt FP8 kernel 成熟后，重跑 `_fp8_probe*.py`
  微基准，加速比 >1.3× 再考虑接入训练。**

---

## 8. 总结

### 七方向 A/B 结果总表
| 方向 | tok/s vs 189K | 显存 | 主要代价 | 采纳 |
|---|---|---|---|---|
| **micro_bsz=3** | **202K (+6.9%)** | 131G | global batch 1.5× | ✅ **最大收益** |
| **torch.compile** | **191.2K (+1.2%)** | 89.3G | 首步编译 | ✅ 采纳 |
| Selective AC, mbsz=6 | 196.4K (+4%) | 139G | 重算；仍<mbsz3；gnorm inf 风险 | ❌ 留 flag（最佳 AC）|
| bf16-reduce | 191K (+1%) | 89.3G | 32-way bf16 求和精度险 | ❌ 留 flag |
| HSDP shard=8×repl=4 | 189.8K (0%) | +11.4G/卡 | 无收益纯付显存 | ❌ 留 flag |
| Full AC, mbsz=12 | 175K (−7%) | 124.6G | 重算 attention + ckpt 前缀污染 | ❌ |
| FP8 (torchao) | 1.00× (零加速) | — | 软件栈未成熟 | ❌ 推迟 |

### 最优配置
**no-AC + micro_bsz=3 + grad_accum=8 + fp32-reduce + 1D-FSDP2 + torch.compile ON → ~204K tok/s
（较基线 189K +8%）。** 已固化为 `mi300_mn.sh` 默认（commit `1f773b1`）。

### 根因：8B 在 MI300×32（有 IB）是计算瓶颈
GPU 满载 100%，主 FLOPs（matmul）跑在高效 rocBLAS/hipBLASLt 上，导致：
- **通信类优化（HSDP、bf16-reduce）净收益 ≈ 0** —— 通信已被计算掩盖、prefetch 已重叠 all-gather。
- **显存类优化（full-AC / SAC）净收益为负或次优** —— 重算是净增 FLOPs，显存够用时任何重算都亏。
- **只有提升算术强度（更大 micro_bsz → GEMM M 维）和算子融合（torch.compile）真正有效。**
这是一个健康的「系统已接近计算最优」信号。

### 保留的可选 flag（供未来 10B+/通信瓶颈/软件栈升级场景）
`--activation_checkpoint`（full AC）、`--selective_ac`（SAC）、`--reduce_bf16`、`--hsdp_shard N`、
`COMPILE_FLAG`（默认 ON）。

### 方法论要点
1. **微基准探路**（FP8）：在碰训练代码前先单 GPU 验证收益，避免盲改引入「零加速 + NaN 风险」的浪费。
2. **计算瓶颈判定**（GPU 100% + matmul 主导）：能事先预判哪些方向注定无收益，把精力集中在算术强度
   与算子融合这两条真正有效的路径上。
3. **单变量 A/B + 稳态读数 + rocm-smi 判活**：blobfuse 日志缓冲下不可靠，须以 GPU 利用率判断进程存活。
4. **ROCm 上 OOM 会伪装成 collective timeout**：单节点 `--standalone` 隔离才能复现真实 OOM。
