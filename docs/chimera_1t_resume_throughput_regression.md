# Chimera-8B 1T 训练 Resume 吞吐下降问题调查

**日期**：2026-07  
**状态**：问题可稳定复现，根因尚未完全定位；已排除多项常见原因  
**影响范围**：Chimera-8B、FSDP2、FP8 tensorwise、`torch.compile`、MI300X 训练  

---

## 1. 摘要

Chimera-8B 的 1T 预训练任务在 64 张 MI300X 上运行。任务停止前，在训练权重已经收敛到当前分布的情况下，稳态吞吐约为 **594–600K tok/s**。从 `ckpt_14000.pt` 恢复后，即使启动参数、代码、数据配置、GPU 数量和模型结构保持一致，稳态吞吐仍下降到约 **560–568K tok/s**，回退约 **5%–7%**。

后续在 ROCm 7.1 / 新版 PyTorch 上进行了独立复现：

- fresh random model：约 **83.9K tok/s**；
- resume `ckpt_14000.pt`：约 **77.2–77.5K tok/s**；
- 回退约 **7.8%**。

因此，该问题不是 ROCm 6.4.3、PyTorch 2.8 或 torchao 0.13 单一版本特有的问题。

调查期间先后排除了：

1. 数据流 fast-forward 到 step 14001；
2. optimizer state 恢复；
3. TensorBoard purge 或日志事件；
4. 残留进程、GPU 占用和节点 CPU 干扰；
5. checkpoint 在 `torch.compile` 前后加载的顺序；
6. checkpoint 在 FSDP2 sharding 前后加载造成的普通 shard layout 差异；
7. tied embedding / `lm_head` 的静态别名失效；
8. ROCm 6.4 / torch 2.8 特定版本问题。

目前最可能的方向是：**训练后权重与 FP8 runtime/scale 状态、allocator 状态或 lazy compile/autotune 选择之间的交互**。不过尚无单项实验能确认最终根因。

当前正式训练已经使用 A1+A2 FSDP2 优化。fresh-run 的吞吐为约 **637K tok/s**，从 checkpoint 恢复后的吞吐约为 **587–589K tok/s**，相对回退仍约 **7%–8%**。这说明 A1+A2 优化有效，但没有消除 resume 特有回退。

---

## 2. 训练背景

正式任务配置如下：

- 模型：Chimera-8B，实际约 7.602B 参数；
- 架构：Qwen3 风格 dense Transformer；
- 集群：8 节点 × 8 MI300X，共 64 GPU；
- sequence length：4096；
- micro batch：4；
- gradient accumulation：4；
- global batch：约 4.19M tokens/step；
- 参数精度：BF16；
- gradient reduce：FP32；
- FP8：torchao tensorwise；
- FSDP：PyTorch FSDP2；
- 编译：`torch.compile`；
- loss：fused cross entropy；
- checkpoint：每 2000 steps 保存一次 full state dict；
- resume：`--resume latest`，同时恢复 model、optimizer 和 step；
- 数据：`EpochMixtureDataset`，恢复时按 step 对每个 replica 的数据流执行确定性 fast-forward。

Checkpoint 是约 93GB 的 full CPU state dict。加载时每个 rank 从 checkpoint 读取完整状态，再通过 `torch.distributed.checkpoint.state_dict` 写入 FSDP2 模型和 optimizer shard。

---

## 3. 问题发现

### 3.1 第一次观测

任务停止前的 TensorBoard 原始数据显示：

- 稳态吞吐通常约 **594–599K tok/s**；
- 训练已经运行至约 step 14000；
- 此时模型并非随机初始化状态，而是已经训练过的权重。

从 `ckpt_14000.pt` 恢复后：

- checkpoint、optimizer 和 step 均成功恢复；
- 数据从 step 14001 对应位置继续；
- loss、gradient norm 和 LR 正常；
- 64 张 GPU 都处于工作状态；
- 但稳态吞吐仅约 **560–568K tok/s**。

初始窗口包含 checkpoint 加载后的 lazy compile，因此不用于比较。即使等待到 step 14020–14050，吞吐仍未恢复到停止前水平。

### 3.2 问题特征

该问题具有以下特征：

- 不是启动阶段短暂 warmup，而是持续性的稳态回退；
- loss 和 gradient norm 正常，没有数值异常；
- GPU HBM 和利用率正常；
- 不会导致训练错误或 checkpoint 损坏；
- fresh process + random weights 较快；
- fresh process + restored trained weights 较慢；
- 在另一套 ROCm/PyTorch 软件栈中仍可复现。

---

## 4. 测试方法

为了避免多个变量混在一起，调查采用单变量 A/B：

1. 每个 trial 使用独立输出目录；
2. 比较时忽略 step 0 和首个日志窗口；
3. 保持模型结构、batch size、FP8、compile、FSDP2 和 GPU 数量一致；
4. 分离数据位置、model state、optimizer state 和 checkpoint 加载顺序；
5. 使用 step 20/30 或 resume 后 step 14020/14030 的稳态窗口；
6. 每轮结束后清理全部训练进程并确认 GPU idle；
7. 不把共享存储 checkpoint 读取耗时计入训练吞吐。

这里需要特别注意：random fresh 与 trained resume 的对比本身包含权重分布差异。因此最强证据仍然是：**同一训练任务在停止前已经使用 trained weights 达到约 600K，恢复相同 checkpoint 后却降到约 562K**。

---

### 5. 已尝试的解决方法和实验结果

### 5.1 清理残留进程与节点资源干扰

恢复后最初发现 node-6 上存在数个历史 StarCoder tokenization retry/keep-alive 进程，合计占用约 38 个 CPU core，节点 load 一度约为 62。

执行了以下操作：

- 按 process group 清理历史 tokenization 进程；
- 停止所有 torchrun 和训练 worker；
- 在 8 个节点上检查 `rocm-smi --showpids`；
- 确认仅剩 0-VRAM 的系统 KFD 条目；
- 重新以 `aiscuser` 启动完整 64-GPU 任务。

结果：

- node-6 load 从约 62 降至约 24；
- 训练吞吐仅从约 560K 小幅变化到约 564K；
- 这些进程在停止前的高速阶段也已经存在。

**结论：残留 CPU 任务和 GPU 污染不是根因。**

---

### 5.2 使用正式 recipe 进行完全干净的重启

为排除旧 launcher 缺参数或节点代码不一致，重新执行了：

- 停止全部节点训练进程；
- 将相同 commit 部署至 8 个节点本地 `/scratch/code`；
- 使用 tracked `recipes/chimera_8b_1t.sh`；
- 明确确认 FP8 tensorwise、fused CE 和 compile 均开启；
- 确认 world size=64、7.602B 参数、BF16 params、FP32 reduce；
- 记录 runtime manifest。

结果：resume 后约 **562K tok/s**，持续到 step 14050，未恢复到约 600K。

**结论：不是旧 launcher、缺失优化参数或节点代码不一致。**

---

### 5.3 TensorBoard purge 与事件历史处理

Resume 时如果 TensorBoard 目录中已经有 checkpoint 之后的事件，曲线可能出现重复或回退。为此修改了恢复顺序：

1. 先加载 checkpoint，得到下一个训练 step；
2. 再创建 `SummaryWriter`；
3. resume 时设置 `purge_step=resume_step`。

从 `ckpt_14000.pt` 恢复时：

```text
purge_step=14001
```

EventAccumulator 验证：

- 清除了 204 个 checkpoint 之后的 stale scalar；
- 没有重复 step；
- loss、LR 和吞吐曲线能从正确位置延续。

对应提交：

```text
6d3ec4a fix: purge stale TensorBoard events on resume
```

但吞吐仍约为 560–568K。

**结论：该修改解决了 TensorBoard 曲线正确性，但没有解决训练吞吐回退。**

---

### 5.4 排除数据 fast-forward 的影响

正常 resume 会将每个 replica 的数据迭代器 fast-forward：

```text
resume_skip = resume_step × micro_bsz × grad_accum
```

为确认数据位置或 iterator 状态是否导致变慢，增加了临时诊断参数，在不加载 checkpoint 的 fresh model 上直接跳过到 step 14001 对应的数据位置。

结果：

| Trial | 稳态吞吐 |
|---|---:|
| fresh random model | 约 608–614K tok/s |
| fresh random model + data skip 14001 | 约 609–614K tok/s |
| normal resume | 约 562–565K tok/s |

**结论：数据位置、数据 mix 进度和 fast-forward 本身不是根因。**

诊断参数在实验完成后已从正式代码移除。

---

### 5.5 在 `torch.compile` 之前恢复 checkpoint

最初代码路径可能先包装 `torch.compile(model)`，再通过编译 wrapper 加载 checkpoint。怀疑 OptimizedModule 可能影响 FSDP shard storage 或 compile guard。

修改为：

1. 构造模型并应用 FSDP2；
2. 创建 optimizer；
3. 将 model/optimizer checkpoint 加载到未编译模型；
4. 最后执行 `torch.compile(model)`。

对应提交：

```text
0d326e9 fix: restore checkpoint before torch compile
```

结果：吞吐没有明显改善，仍约 **562–565K tok/s**。

需要注意，`torch.compile` 是 lazy 的，因此该实验排除了明显的 wrapper/load 顺序问题，但不能完全排除第一次真实执行时的 compile/autotune 根据 restored runtime state 选择了不同方案。

**结论：简单调整 restore 与 `torch.compile` 的顺序不能解决问题。**

该顺序更合理，已保留在正式代码中。

---

### 5.6 分离 model state 与 optimizer state

增加临时诊断参数，允许只恢复：

- model；
- optimizer；
- model + optimizer。

其中最关键的结果是：

| Trial | 稳态吞吐 |
|---|---:|
| fresh | 约 608–614K tok/s |
| resume model only | 约 565–566K tok/s |
| resume model + optimizer | 约 562–565K tok/s |

只恢复模型权重时，主要吞吐回退已经出现。因此 optimizer moment、optimizer shard 或 optimizer state restore 不是主要原因。

对应诊断提交：

```text
b03f816 test: isolate model and optimizer resume state
```

**结论：问题主要与 restored model 路径相关，optimizer 被排除。**

诊断参数随后已移除。

---

### 5.7 在 FSDP2 sharding 之前加载模型权重

正常路径是：

1. 模型转 FP8；
2. FSDP2 fully shard；
3. 使用 `set_model_state_dict` 写入 local shard。

怀疑 checkpoint restore 可能在 sharding 后替换参数 storage，导致不理想的 shard layout 或内存排列。因此测试了：

1. 模型转 FP8；
2. 在未 shard 的模型上通过 `model.load_state_dict` 加载完整 model state；
3. 再执行 FSDP2 `fully_shard`；
4. sharding 后恢复 optimizer state；
5. 删除 full checkpoint 对象并执行 `gc.collect()`。

对应提交：

```text
55682fd fix: restore model weights before FSDP2 sharding
319510a fix: import gc before resume cleanup
```

结果仍约为 **565–568K tok/s**。

**结论：普通的 post-shard checkpoint 写入和 shard storage layout 不是根因。**

该诊断实现较复杂且没有收益，最终已回退。

---

### 5.8 重置 `lm_head` 的诊断及其错误解读

为判断训练后 logits 分布是否让 fused CE 变慢，曾在加载 checkpoint 后重新初始化：

```python
torch.nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
```

结果：

- 吞吐恢复到约 **614–618K tok/s**；
- loss 上升到约 **7**。

最初曾据此怀疑“训练后的 output logits 使 fused CE 变慢”。这个解释后来被撤回。

原因是 Chimera 使用 tied embeddings：

```text
lm_head.weight = tok_emb.weight
```

重新初始化 `lm_head.weight` 同时也重置了 input token embeddings。因此该实验不只改变 CE 输入，而是改变了整个模型从第一层开始的 activation 分布。

对应诊断提交：

```text
65893e0 test: isolate resumed output-head performance
```

**结论：该实验说明权重/activation 分布与性能有关，但不能把问题定位到 fused CE 或 output head。**

诊断代码随后在以下提交中清理：

```text
04e75fb chore: remove resume throughput diagnostics
```

---

### 5.9 检查 tied embedding 是否在 restore/FSDP2 后断开

由于 checkpoint 中 `tok_emb.weight` 和 `lm_head.weight` 是两个内容相等的 CPU tensor，需要确认 restore 后是否仍然共享同一个模型参数。

进行了两类静态测试。

#### World size 1

构造顺序：

```text
build model → CUDA → FP8 conversion → FSDP2 → restore
```

检查：

- Python object identity；
- local storage pointer；
- 对一侧 mutation 是否反映到另一侧；
- checkpoint restore 是否同时传播。

全部通过。

#### World size 8

所有 rank 均通过：

- `python_same=True`；
- local pointers 相等；
- mutation linked；
- restore propagated；
- 355 个参数名、354 个唯一对象；
- local shard 77,791,232 elements。

日志：

```text
$S/logs/check_tied_resume_v1.log
$S/logs/check_tied_resume_world8_v2.log
```

脚本：

```text
_check_tied_resume_v1.py
_check_tied_resume_world8_v2.py
```

**结论：明显的 tied alias 断开问题已排除。**

world-64 独有的 tied metadata 问题理论上仍存在可能，但概率较低；vocab size 可以被 64 整除，也未观察到 alias 或 shard 异常证据。

---

### 5.10 ROCm 7.1 / 新版 PyTorch 交叉验证

为判断是否是主集群软件栈特有问题，在单节点 8×MI300X 的 `chec-test-env` 上进行了控制实验：

```text
PyTorch 2.10.0.dev20251112+rocm7.1
torchao 0.15.0
ROCm 7.1
```

结果：

| Trial | step 20/30 稳态吞吐 |
|---|---:|
| fresh random 8B | 83.9K tok/s |
| resume `ckpt_14000` | 77.5 / 77.2K tok/s |

回退约 **7.8%**，与主集群上的 5%–8% 同一量级。

首个日志窗口包含 lazy compile：

- fresh 首窗约 55.4K；
- resume 首窗约 36.4K。

这些首窗不纳入稳态比较。

日志：

```text
/scratch/rocm7_resume_ab_v1/fresh.log
/scratch/rocm7_resume_ab_v1/resume-local.log
/scratch/rocm7_resume_ab_v1/status.log
```

共享 blob 上直接恢复 93GB checkpoint 时，8 个 rank 曾卡在 FUSE `open`。将 checkpoint 复制到本地 `/scratch/rocm7_resume_ab_v1/ckpt_14000.pt` 后恢复成功。这个现象只影响 checkpoint 读取启动时间，与后续训练稳态吞吐是两个独立问题。

**结论：问题不是 ROCm 6.4.3 / PyTorch 2.8 / torchao 0.13 单版本 bug。**

---

## 6. 已排除和未排除的因素

### 6.1 已基本排除

| 因素 | 证据 |
|---|---|
| 数据位置 / data fast-forward | fresh + skip 14001 与 fresh 吞吐相同 |
| optimizer state | model-only restore 已出现主要回退 |
| TensorBoard purge | 只影响事件曲线，不影响计算吞吐 |
| 残留 GPU 进程 | 全节点彻底清理后仍复现 |
| node-6 CPU keep-alive | 清理后只有极小变化，且高速阶段也存在 |
| launcher 参数缺失 | 正式 recipe 干净重启仍复现 |
| compile wrapper 后加载 | restore-before-compile 无改善 |
| 普通 post-shard storage layout | pre-shard model load 无改善 |
| tied embedding 静态断链 | world-1/world-8 identity、pointer、mutation 全通过 |
| ROCm 6.4 / torch 2.8 特定缺陷 | ROCm 7.1 / torch 2.10.dev 仍复现 |

### 6.2 尚未排除

1. **FP8 runtime 或 scale state**  
   Checkpoint 保存模型和 optimizer，但没有保存所有 FP8 runtime/cache/scale 状态。trained weights 经 FP8 conversion 和 restore 后，可能触发与 fresh random weights 不同的 scale 或 kernel 路径。

2. **FP8 conversion 与 load 顺序的交互**  
   已测试 FP8 conversion 后再以不同 FSDP 时机加载，但尚未完成“先以 BF16 加载 checkpoint，再转换 FP8”的严格 A/B。

3. **allocator / GC 状态**  
   每个 rank 加载约 93GB full CPU checkpoint，随后生成/写入 GPU shard。即使 Python 对象被删除，allocator fragmentation 或缓存状态仍可能影响后续 kernel。

4. **lazy compile/autotune 选择**  
   `torch.compile` 在第一次真实执行时才编译。trained restored activations 或恢复后的 runtime 状态可能导致不同的 kernel/autotune 选择。简单的 restore-before-compile 不足以排除此项。

5. **权重/activation 分布对 kernel 的真实影响**  
   tied embedding 重置实验恢复吞吐，但同时改变了整个网络的 activation 分布。该结果提示数据值分布可能影响 FP8、scaled GEMM 或 fused kernel 性能，但尚未定位到具体算子。

6. **world-64 特有 FSDP tied metadata 问题**  
   world-8 静态测试通过，因此概率较低，但尚未做完整 world-64 动态 kernel 级验证。

---

## 7. 建议的后续诊断

如果继续调查，建议按以下顺序进行。

### 7.1 BF16 load → FP8 conversion A/B

当前主要路径是：

```text
build → FP8 convert → FSDP2 → load checkpoint
```

建议增加严格对照：

```text
build BF16 → load trained checkpoint → FP8 convert → FSDP2
```

如果后一条恢复 fresh 吞吐，根因将集中到 FP8 module conversion/load 的交互。

### 7.2 checkpoint 释放与 allocator 清理 A/B

在所有 rank 完成 restore 后统一执行：

```python
del checkpoint
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

需要与不清理版本分别运行足够多的稳态 steps。此前 pre-shard load 中执行过 `gc.collect()`，但没有形成完整的 GC + empty-cache + synchronize 单变量 A/B。

### 7.3 同一进程内 trained/random 权重热切换

在同一个已编译进程内，分别运行：

1. random tied weights；
2. trained tied weights；
3. 再切回 random tied weights。

这样可以区分：

- 权重/activation 数值本身导致 kernel 变慢；
- 首次 compile/autotune 根据初始权重状态选择了较慢方案；
- checkpoint restore/allocator 生命周期导致变慢。

### 7.4 算子级 profiler 对比

对 fresh 和 resume 各采集相同步数的 profiler trace，重点比较：

- FP8/scaled GEMM；
- FSDP all-gather/reduce-scatter；
- fused CE；
- graph break 数量；
- kernel shape 是否一致；
- kernel duration 和 launch count；
- CPU launch gap；
- allocator retry 或同步点。

目标不是只看总 GPU utilization，而是定位具体哪个算子或通信阶段多出约 7% 时间。

### 7.5 保存或预热 FP8 runtime state

调查 torchao tensorwise FP8 中未进入 model state dict 的状态，并尝试：

- 显式 checkpoint/restore scale state；
- resume 后固定步数预热再重新计时；
- fresh model 注入 trained FP8 scale state；
- trained model 清空并重新生成 FP8 runtime cache。

---

## 8. 最终结果与当前处理方式

截至本轮调查结束：

- resume 吞吐回退可以稳定复现；
- 回退幅度约为 **5%–8%**；
- 在 ROCm 6.4 和 ROCm 7.1 两套环境中都存在；
- 没有发现 loss、gradient norm、checkpoint 正确性或 tied weights 正确性问题；
- 多个常见修复方法均未恢复停止前吞吐；
- 根因仍未完全定位。

因此最终没有为了追求吞吐而采用未经证明的 checkpoint 加载改动。临时诊断参数和复杂的 pre-shard restore 实现已清理，只保留了更合理的正式行为：

- checkpoint 在 `torch.compile` 之前恢复；
- model、optimizer 和 step 正常恢复；
- 数据流按 step fast-forward；
- TensorBoard 使用 `purge_step` 清理 stale events；
- 正式训练继续使用数值安全的 FP32 gradient reduce。

在 A1+A2 FSDP2 优化上线后：

| 场景 | 吞吐 |
|---|---:|
| 原始 fresh 基线 | 约 600K tok/s |
| A1+A2 fresh-run | 约 637K tok/s |
| A1+A2 resume `ckpt_14000` | 约 587–589K tok/s |

Resume 任务的 loss、gradient norm、HBM 和 64-GPU 利用率均正常，因此当前策略是：

1. 不阻塞正式 1T 训练；
2. 接受约 7%–8% 的 resume 稳态回退；
3. 将其作为独立性能问题继续跟踪；
4. 后续优先使用 FP8 load-order A/B 和算子级 profiler 定位根因。

---

## 9. 相关提交与日志

### 正式功能提交

```text
87f73a5 resume: ckpt rotation + 'latest' auto-resume + data fast-forward
6d3ec4a fix: purge stale TensorBoard events on resume
0d326e9 fix: restore checkpoint before torch compile
```

### 临时诊断提交

```text
b03f816 test: isolate model and optimizer resume state
55682fd fix: restore model weights before FSDP2 sharding
319510a fix: import gc before resume cleanup
65893e0 test: isolate resumed output-head performance
04e75fb chore: remove resume throughput diagnostics
```

### 主要日志

```text
$S/logs/mn_node*.log
$S/logs/check_tied_resume_v1.log
$S/logs/check_tied_resume_world8_v2.log
/scratch/rocm7_resume_ab_v1/fresh.log
/scratch/rocm7_resume_ab_v1/resume-local.log
/scratch/rocm7_resume_ab_v1/status.log
```

其中：

```text
S=/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/chec/pretrain
```
