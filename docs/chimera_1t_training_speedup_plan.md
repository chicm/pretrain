# Chimera-8B 1T 预训练吞吐优化计划（MI300X / FSDP2）

**日期**：2026-07  
**状态**：调研与实验计划；尚未应用到正在运行的正式任务  
**适用范围**：Chimera-8B（7.602B dense）、PyTorch FSDP2、64×MI300X、FP8 Linear、sequence length 4096

> 本文目标是在**不改变模型架构、数据配比、全局 batch 和学习率计划**的前提下，利用当前富余 HBM，减少 FSDP2 通信和参数重建开销，并进一步优化 MI300X 上的 GEMM/Attention kernel。所有改动必须先经过短程 A/B，再从已有 checkpoint 恢复正式训练；禁止直接在运行中的 1T 任务上试验。

---

## 1. 当前基线

| 项目 | 当前值 |
|---|---:|
| 模型 | Chimera-8B，7.602B 参数 |
| GPU | 8 节点 × 8 MI300X = 64 GPU |
| sequence length | 4096 |
| micro batch / GPU | 4 |
| gradient accumulation | 4 |
| global tokens / step | 4,194,304 |
| 精度 | BF16 parameters / FP32 gradient reduce / FP8 大型 Linear |
| 并行 | FSDP2，1D full shard，world size 64 |
| 编译 | `torch.compile` ON |
| fused kernel | fused cross entropy、fused AdamW、SDPA |
| activation checkpoint | OFF |
| 实际吞吐 | 通常约 600K–612K tok/s |
| 显存峰值 | 约 76.7GB / 192GB 每卡 |
| 剩余 HBM | 约 115GB / GPU |

训练已经稳定运行并能定期保存、轮换及恢复 checkpoint。当前 loss、gradient norm 和 non-finite skip rate 均健康，因此本计划不调整 LR，也不通过增大全局 batch 换取表面吞吐。

### 1.1 已知现象

提高 micro-batch 并未继续提升 tokens/s，说明当前已不是单纯的 batch launch overhead 或显存容量瓶颈。主要优化空间更可能来自：

1. gradient accumulation 中不必要的重复梯度同步；
2. FSDP2 forward/backward 之间的参数 reshard/all-gather；
3. BF16 parameter all-gather 的通信量；
4. 固定 Transformer GEMM shape 未选到最优 hipBLASLt kernel；
5. SDPA、RMSNorm、RoPE、SwiGLU 等 kernel 的 MI300X 效率。

---

## 2. 总体原则

1. **正式任务优先**：不干扰正在运行的 1T 任务，只在 checkpoint 边界切换。
2. **一次只改一个变量**：每项实验均以同一 checkpoint、同一数据状态、同一训练配置对比。
3. **先通信、后 kernel**：先修复明确的 FSDP2 通信冗余，再投入高改造成本的 AITER/Liger 类 kernel。
4. **用 HBM 换吞吐**：当前每卡有约 115GB 余量，允许保留 unsharded parameters/gradients。
5. **保持训练语义**：第一阶段不改变 global batch、LR、optimizer、数据 mix、FP8 recipe。
6. **以稳态吞吐为准**：排除 FP8/`torch.compile` 首次编译、checkpoint 保存和前若干 warmup step。
7. **收益不可直接相加**：多个优化可能作用于同一通信或计算区间，组合收益需实测。

---

## 3. 优先级与预期

| 优先级 | 优化项 | 预期收益（需实测） | 显存代价 | 数值风险 | 工程成本 |
|---:|---|---:|---:|---:|---:|
| P0 | gradient accumulation 仅最后一个 micro 同步梯度 | +3%～20% | 中 | 低 | 低 |
| P0 | accumulation 中关闭 backward reshard | +1%～8% | 中 | 低 | 低 |
| P1 | `reshard_after_forward=8/False` | +3%～12% | 中/高 | 低 | 低 |
| P1 | FP8 FSDP all-gather | +2%～8% | 低 | 中低 | 中 |
| P1 | hipBLASLt / TunableOp 离线 GEMM 调优 | +2%～10% | 低 | 低 | 中 |
| P2 | AITER FlashAttention / fused kernels | 单项 +1%～5%，组合可能更高 | 低 | 中 | 高 |
| P2 | BF16 gradient reduce | 0%～数个百分点 | 低 | 中 | 低 |
| P3 | `torch.compile` max-autotune | 不确定 | 低 | 低 | 中 |

以上区间仅用于安排实验优先级，不是承诺值。

---

## 4. P0：修正 gradient accumulation 通信方式

### 4.1 当前问题

当前训练循环在每个 micro-batch 上直接执行：

```python
for micro in range(cfg.grad_accum):
    _, loss = model(x, y)
    (loss / cfg.grad_accum).backward()
```

没有显式调用 FSDP2 的 `set_requires_gradient_sync()`。在 `grad_accum=4` 时，默认行为会让每个 micro-batch 都进行 gradient synchronization，而真正需要跨 rank 同步的只有最后一个 micro-batch。

PyTorch FSDP2 将：

```python
model.set_requires_gradient_sync(False)
```

定义为 FSDP1 `no_sync()` 的等价接口，可实现 gradient accumulation without communication。

### 4.2 计划改动

```python
for micro in range(cfg.grad_accum):
    is_last_micro = micro == cfg.grad_accum - 1

    model.set_requires_gradient_sync(is_last_micro)
    model.set_reshard_after_backward(is_last_micro)

    try:
        x, y = next(data_iter)
    except StopIteration:
        ...

    x = x.to("cuda", non_blocking=True)
    y = y.to("cuda", non_blocking=True)
    _, loss = model(x, y)
    loss = loss / cfg.grad_accum
    loss.backward()
```

含义：

- micro 0–2：不 reduce-scatter/all-reduce 梯度；
- micro 0–2：backward 后不立即 reshard，降低下一次 forward 前的参数通信；
- micro 3：恢复梯度同步并在 backward 后恢复正常分片状态；
- optimizer step、gradient clipping、non-finite 检查保持不变。

`recurse=True` 是 FSDP2 API 默认值，因此在 root FSDP module 上调用会递归应用到各层。

### 4.3 正确性测试

增加单元/集成测试：

1. 单 GPU 或两 GPU 小模型，比较：
   - 基线：每个 micro 同步；
   - 优化：前三个 micro 不同步；
   - 相同初始权重和输入下，最终 accumulated gradients/updated weights 接近。
2. 测试 `grad_accum=1`：行为必须与当前代码完全一致。
3. 测试 non-finite skip：最后 micro 出现非有限梯度时仍跳过整个 optimizer step。
4. 测试 checkpoint save/resume：FSDP 状态在最后 micro 后已经恢复到可保存状态。

### 4.4 验收标准

- 100 个稳态 step 平均吞吐至少提高 **3%**；
- loss 曲线与基线一致，无系统性偏移；
- gradient norm 处于同一数量级；
- 无新 OOM、hang、RCCL timeout；
- checkpoint 可保存并从其恢复至少 20 step。

若收益低于 3%，仍可保留代码开关，但默认不切换正式任务。

---

## 5. P1：用富余 HBM 减少 parameter all-gather

### 5.1 `set_reshard_after_backward(False)`

此项与 P0 配套。PyTorch 官方说明它可在 gradient accumulation 时，以更高显存换取更少通信：backward 后保留 unsharded parameters，使下一个 forward 不必重新 all-gather。

先分别测：

- A1：仅 gradient no-sync；
- A2：gradient no-sync + accumulation 中 no-reshard-after-backward。

这样能确认收益来自减少 gradient reduce 还是 parameter all-gather。

### 5.2 静态 `reshard_after_forward`

当前：

```python
for layer in model.layers:
    fully_shard(layer, mp_policy=mp, **fsdp_kw)
fully_shard(model, mp_policy=mp, **fsdp_kw)
```

候选方案：

```python
fully_shard(
    layer,
    mp_policy=mp,
    reshard_after_forward=reshard_mode,
    **fsdp_kw,
)
```

依次测试：

1. `True`：当前基线；
2. `8`：forward 后分片到较小 world size，目标是贴合单节点 8 GPU 拓扑；
3. `False`：forward 后保留完整参数，消除 backward 前再次 all-gather。

### 5.3 显存判断

7.602B BF16 参数完整副本约 15.2GB。即使考虑 FP8 转换状态、梯度、临时 buffer 和 compile workspace，当前约 115GB/GPU 的余量也足以测试更激进的 no-reshard 策略。

但实际峰值必须以 `torch.cuda.max_memory_allocated()` 和 ROCm 侧 reserved memory 为准；不能只用静态参数量估算。

### 5.4 验收标准

- 在 A2 最优配置上额外提高至少 **2%**；
- 峰值显存建议保持 `< 150GB/GPU`，至少保留 20% 安全余量；
- 无 backward hook、checkpoint 或 compile 兼容问题。

若 `False` 收益不明显或引入显存波动，优先采用 `8` 或恢复 `True`。

---

## 6. P1：FP8 parameter all-gather

### 6.1 当前状态

当前 `src/fp8_utils.py` 对 tensorwise recipe 使用：

```python
cfg = Float8LinearConfig()
```

大型 attention/MLP Linear 已转换为 TorchAO `Float8Linear`，但没有显式启用：

```python
enable_fsdp_float8_all_gather=True
```

因此需要确认当前 TorchAO/FSDP2 组合的 all-gather 是否仍以 BF16 传输。

### 6.2 计划

1. 在实际运行环境中检查 `Float8LinearConfig` signature；
2. 确认已安装版本是否支持 `enable_fsdp_float8_all_gather`；
3. 确认它与当前 MI300 FNUZ dtype、tensorwise recipe、FSDP2 mixed precision 的兼容性；
4. 若支持，增加显式 CLI/config 开关，默认先 OFF；
5. 单节点和多节点分别验证。

候选代码：

```python
cfg = Float8LinearConfig(
    enable_fsdp_float8_all_gather=True,
)
```

对于 rowwise recipe，应基于 `from_recipe_name()` 返回配置后再安全地设置，避免覆盖 recipe 的其它字段。

### 6.3 风险

- 本项目实际 TorchAO 版本可能早于当前在线文档；不能直接按新版本 API 修改；
- FP8 communication scale/metadata 可能影响 FSDP grouping 或 compile graph；
- 需要确认 checkpoint 中保存的是可恢复的原始/高精度状态，而不是不可移植的临时 FP8 通信表示。

### 6.4 验收标准

- 多节点吞吐额外提高至少 **2%**；
- 200-step loss/gnorm 与 FP8 基线一致；
- checkpoint round-trip 通过；
- FP8 converted layer 数量不减少，compile 时间没有异常增长。

---

## 7. P1：hipBLASLt / PyTorch TunableOp 离线 GEMM 调优

### 7.1 原因

Chimera-8B 的 GEMM shape 高度固定：

- Q/K/V/O projections；
- SwiGLU gate/up/down projections；
- micro-batch 4 × sequence 4096；
- 相同 shape 在多层、forward/backward 中反复出现。

默认 hipBLASLt heuristic 不保证为每个 shape 选择最快 kernel。固定 shape、长时间运行的 1T 训练非常适合一次性离线调优。

### 7.2 实施阶段

#### 阶段 1：收集 shape

使用 profiler 或 TunableOp 记录实际调用的：

- M/N/K；
- dtype（FP8/BF16/FP32 accumulate）；
- transpose/layout；
- bias/epilogue；
- forward、dgrad、wgrad。

#### 阶段 2：隔离调优

在相同 MI300X 软件环境中运行：

- hipBLASLt offline tuning；或
- PyTorch TunableOp。

调优不能在正式任务首次启动时在线进行，否则每种新 shape 的搜索会造成长暂停和 rank 不同步风险。

#### 阶段 3：加载结果

将 tuning result 作为版本化运行资产加载到 smoke job，确认所有节点看到相同结果，再做多节点吞吐测试。

### 7.3 验收标准

- 稳态端到端吞吐提高至少 **2%**；
- 不只看单个 GEMM microbenchmark；
- 所有节点 kernel 选择一致，无首次动态调优停顿；
- loss 完全不受影响。

---

## 8. P2：AITER / Primus-Turbo kernel

AMD Primus/Primus-Turbo 的优化重点包括 AITER Attention、融合 Norm/RoPE/SwiGLU、GEMM tuning 和通信调度。当前已有 `torch.compile`、SDPA、fused CE 和 fused AdamW，但以下算子仍可能存在空间：

1. AITER FlashAttention 替换 `F.scaled_dot_product_attention`；
2. fused RMSNorm；
3. fused RoPE；
4. fused SwiGLU；
5. 更适合 MI300X FP8 的 fused Linear/activation path。

### 8.1 测试原则

- 每个算子单独提供 feature flag；
- 优先替换 Attention，再测试 elementwise fusion；
- 使用相同输入做 forward/backward 数值对齐；
- 测试 QK-Norm、GQA、causal mask、sequence=4096 的兼容性；
- 不因 kernel 集成改变模型 state_dict key 或 HF export 映射。

### 8.2 为什么不是第一优先级

- 工程改动比 FSDP2 开关大；
- `torch.compile` 可能已经融合部分 RMSNorm/RoPE/SwiGLU；
- PyTorch SDPA 在 ROCm 上可能已经进入 flash backend；
- 需要 profiler 先确认热点，避免重复优化非瓶颈算子。

---

## 9. P2：其它可选项

### 9.1 BF16 gradient reduce

当前：

```python
MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
```

候选：

```python
MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
)
```

BF16 reduce 可将梯度通信字节数约减半，但会改变归约精度。之前在较小规模上没有观察到净收益；64 GPU/8 节点拓扑下可在 P0 后重新测一次。只有同时满足吞吐收益和数值稳定性时才启用。

### 9.2 `torch.compile` max-autotune

可以测试更激进的 compile/autotune mode，让 Inductor/Triton 搜索更多 kernel。风险是：

- 首次编译时间显著增加；
- 缓存管理更复杂；
- 某些 shape 可能比默认模式更慢；
- 对 hipBLASLt GEMM 未必有额外帮助。

排在离线 GEMM tuning 和 AITER profiling 之后。

### 9.3 TorchAO FP8 其它选项

可调研但不先启用：

- `force_recompute_fp8_weight_in_bwd`；
- `round_scales_to_power_of_2`；
- `pad_inner_dim`。

这些选项可能在显存、quantization overhead、kernel shape 和数值误差之间交换。必须基于当前安装版本和 profiler 数据逐项验证。

---

## 10. Profile 计划

在替换 kernel 前，采集一段不包含 compile 和 checkpoint 的稳态窗口。目标不是长时间全量 trace，而是回答以下问题：

1. 每 step 中 GEMM、Attention、elementwise、optimizer、collective 各占多少时间；
2. 每个 micro 是否都出现 gradient reduce-scatter；
3. forward/backward 前有多少 parameter all-gather；
4. all-gather/reduce-scatter 与 GEMM 的 overlap 是否有效；
5. top GEMM shape 及对应 hipBLASLt solution；
6. SDPA 实际进入 flash、memory-efficient 还是 math backend；
7. 是否存在 CPU launch gap、DataLoader gap 或单 rank straggler。

工具优先级：

1. PyTorch profiler：短窗口、算子和 collective 级别；
2. ROCm/rocprofiler-systems：kernel timeline 和通信重叠；
3. AMD TraceLens/Primus 工具链：如环境可用，再做深度分析。

Profiler 本身会降速，不能用 profiler 开启时的 tokens/s 作为正式吞吐数字。

---

## 11. 标准 A/B 实验矩阵

### 11.1 第一轮：FSDP2 通信

| ID | gradient sync | reshard after backward | reshard after forward | 目的 |
|---|---|---|---|---|
| A0 | 每个 micro | True | True | 当前基线 |
| A1 | 仅最后 micro | True | True | 单测 gradient no-sync |
| A2 | 仅最后 micro | 仅最后 micro=True | True | accumulation 间保留参数 |
| A3 | 同 A2 | 同 A2 | 8 | 节点内折中分片 |
| A4 | 同 A2 | 同 A2 | False | 最大化 HBM 换通信 |

### 11.2 第二轮：FP8 与 GEMM

在第一轮最优配置上继续：

| ID | 新增项 | 目的 |
|---|---|---|
| B0 | 第一轮 winner | 新基线 |
| B1 | FP8 all-gather | 压缩 parameter communication |
| B2 | offline GEMM tuning | 优化固定 GEMM shape |
| B3 | FP8 all-gather + tuning | 验证组合收益 |
| B4 | BF16 reduce | 验证 64-GPU gradient bandwidth |

### 11.3 第三轮：kernel

| ID | 新增项 |
|---|---|
| C1 | AITER FlashAttention |
| C2 | fused RMSNorm |
| C3 | fused RoPE |
| C4 | fused SwiGLU |
| C5 | 已验证 kernel 组合 |

---

## 12. Benchmark 方法

每个实验使用相同 checkpoint 和尽可能相同的数据 cursor：

1. 启动后允许 FP8 + `torch.compile` 完成首次编译；
2. 丢弃 compile 阶段和随后的 20 个 warmup step；
3. 连续测至少 100 个无 checkpoint 保存的 step；
4. 最好重复 2 次，避免网络瞬态误判；
5. 记录：
   - mean、median、P10/P90 tok/s；
   - step time；
   - max allocated/reserved HBM；
   - loss、gradient norm、skip count；
   - all-gather/reduce-scatter 次数与总时间；
   - GPU utilization/power（若可用）。

统一计算：

```text
tokens_per_step = micro_bsz × grad_accum × world_size × block_size
throughput = tokens_per_step / median_step_time
speedup = candidate_throughput / baseline_throughput - 1
```

不得将 checkpoint 保存 step、首次编译 step 或异常 rank 重连 step 混入平均值。

---

## 13. 正式上线门槛

候选配置必须同时满足：

### 性能

- 端到端稳态吞吐至少提高 **3%**；
- 不是仅单卡或单 GEMM microbenchmark 提升；
- 多节点 scaling 无退化。

### 数值

- 200-step loss 与基线无系统性偏移；
- gradient norm 分布相近；
- non-finite skip rate不升高；
- 无新增 NaN/Inf。

### 工程

- 64 GPU smoke 通过；
- checkpoint 保存、轮换、`--resume latest` 均通过；
- 所有节点代码版本和 tuning 文件一致；
- feature flag 可关闭，保留快速回滚路径；
- 不改变 state_dict/export 格式，或已提供兼容迁移。

---

## 14. 上线与回滚

### 14.1 上线流程

1. 等待正式任务生成新的完整 checkpoint；
2. 记录旧任务最后完成 step、吞吐和 checkpoint；
3. 停止旧进程，确认所有节点无残留 rank；
4. 部署已通过 A/B 的 commit 到所有节点；
5. 使用新 run ID 从 `latest` 恢复；
6. 首次编译期间不提前判死；
7. 连续观察至少 200 step；
8. 确认下一次 checkpoint 可写并可恢复。

### 14.2 回滚条件

出现任一条件立即回到旧 commit + 最近健康 checkpoint：

- throughput 无收益或下降超过 2%；
- OOM、RCCL hang、rank 间 step 不一致；
- loss/gnorm 明显偏离；
- skip rate 持续升高；
- checkpoint save/resume 失败；
- compile 时间异常或每次重启重复长时间 autotune。

所有优化必须由 CLI/config feature flag 控制，避免回滚时临时改代码。

---

## 15. 暂不优先的方向

| 方向 | 暂不优先原因 |
|---|---|
| 继续增大 micro-batch | 已观察到吞吐不再增长；会增加显存但不一定提高 kernel 效率 |
| 增大 grad accumulation/global batch | 可能改变优化行为和 token/step，不符合本轮约束 |
| activation checkpoint | 显存富余，重计算会降低吞吐 |
| Tensor Parallel | 8B 单卡可容纳，TP 会引入每层通信 |
| Context Parallel | sequence length 4096，无明显必要 |
| CPU/NVMe offload | 当前 HBM 富余，只会增加传输和同步 |
| 迁移 Megatron | 潜在收益存在，但迁移/checkpoint/数据兼容成本过高，不适合运行中的 1T 任务 |
| 直接套用完整第三方 recipe | 变量过多，难以归因且不符合当前可恢复训练约束 |

---

## 16. 推荐执行顺序

### 第一阶段：低风险高价值

1. 实现 FSDP2 gradient no-sync feature flag；
2. 实现 accumulation no-reshard-after-backward feature flag；
3. 跑 A0/A1/A2；
4. 再跑 `reshard_after_forward=8/False`；
5. 选出第一轮 winner。

### 第二阶段：FP8 通信与 GEMM

1. 验证当前 TorchAO API；
2. 测 FP8 all-gather；
3. 收集并离线调优 GEMM shape；
4. 测单项与组合收益。

### 第三阶段：profile 驱动 kernel 替换

1. 确认 Attention/Norm/RoPE/SwiGLU 热点；
2. 逐项接入 AITER/Primus-Turbo kernel；
3. 数值和吞吐均通过后组合。

### 目标区间

以约 600K–612K tok/s 为基线：

- 第一阶段合理目标：**650K–700K tok/s**；
- 若 FP8 communication、GEMM tuning 和 kernel 均有独立收益，进一步目标可设为 **700K–800K tok/s**；
- 目标仅用于规划，最终以严格 A/B 结果决定是否上线。

---

## 17. 参考资料

1. PyTorch FSDP2 `fully_shard` API：  
   <https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html>
2. PyTorch：Float8 training with FSDP2：  
   <https://pytorch.org/blog/training-using-float8-fsdp2/>
3. TorchAO `Float8LinearConfig`：  
   <https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.float8.Float8LinearConfig.html>
4. AMD Primus performance deep dive：  
   <https://rocm.blogs.amd.com/software-tools-optimization/primus-deep-dive/README.html>
5. AMD hipBLASLt offline tuning：  
   <https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt_offline_tuning/README.html>
6. AMD Primus-Turbo：  
   <https://github.com/AMD-AGI/Primus-Turbo>
7. LinkedIn Liger-Kernel：  
   <https://github.com/linkedin/Liger-Kernel>

---

## 18. 当前结论

当前最明确的机会不是继续加 batch，而是：

1. **gradient accumulation 前三个 micro-batch 不做 FSDP2 gradient synchronization**；
2. **利用富余 HBM，在 accumulation 和 forward/backward 之间减少 reshard/all-gather**；
3. **启用并验证 FP8 parameter all-gather**；
4. **对固定 FP8/BF16 GEMM shape 做离线 tuning**；
5. 最后再由 profiler 指导 AITER/Primus-Turbo kernel 替换。

其中第 1 项是代码中最明确、风险最低且最应优先验证的优化点。所有候选项均需在 checkpoint 边界进行短程 A/B，达到性能、数值和可恢复性门槛后才进入正式 1T 训练。