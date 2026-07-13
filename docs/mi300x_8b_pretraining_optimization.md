# MI300X 上从零预训练 8B 模型：框架、精度、并行与性能优化指南

## 1. 场景与目标

本文面向以下训练场景：

- 从零开始预训练约 8B 参数的 Decoder-only 大语言模型
- 硬件平台为 AMD MI300 系列 GPU
- 初始规划规模为 128 张 MI300X
- 当前实际运行规模为 32 张 MI300X
- 当前训练框架为 PyTorch FSDP2
- 当前训练精度为 BF16
- 当前每卡 micro-batch size 为 2
- micro-batch size 提高到 4 时发生 OOM
- 当前 32 卡总吞吐约为：

```text
1.9e5 tokens/s
```

折算到每张 GPU：

```text
190,000 / 32 ≈ 5,940 tokens/s/GPU
```

需要注意，这个数字是否理想，强烈依赖以下因素：

- sequence length
- 模型 hidden size、层数、head 数
- 是否使用 GQA
- 是否启用 activation checkpointing
- 是否使用 FlashAttention
- 单机卡数和节点数
- 网络拓扑和 RCCL 性能
- 是否使用 sequence packing
- 是否包含 padding token
- FSDP 分片组大小
- 是否启用 `torch.compile`

因此，优化时不能只看 tokens/s，还应同时看显存峰值、MFU、通信时间、有效 token 比例和扩展效率。

---

# 2. 总体推荐技术栈

对于 8B 稠密模型，推荐优先考虑以下组合：

```text
训练框架：
AMD ROCm Megatron-LM 官方优化栈
或
PyTorch FSDP2

基础精度：
BF16

后续精度优化：
FP8，仅对主要 Linear/GEMM 使用

Attention：
FlashAttention 2
或 ROCm 上经过验证的 fused SDPA

算子：
Fused RMSNorm
Fused RoPE
Fused SwiGLU
Fused AdamW

并行：
优先 Data Parallel / FSDP
避免不必要的 Pipeline Parallel
Tensor Parallel 尽量限制在单节点内

显存：
Selective Activation Checkpointing
Distributed Optimizer
避免 CPU/NVMe Offload

数据：
Pretokenized
Memory-mapped binary
Sequence packing
本地 NVMe cache
异步预取
```

核心原则是：

> 8B 模型在 MI300X 上通常不是参数装不下，而是激活、FSDP 参数 all-gather 峰值、通信效率和 kernel 利用率成为瓶颈。

---

# 3. 精度优化

## 3.1 BF16：推荐作为稳定基线

BF16 是 MI300X 上训练大模型的首选基础精度。

建议：

- 参数计算：BF16
- 激活：BF16
- 梯度通信：BF16
- Adam 一阶、二阶矩：FP32
- loss reduction：BF16 或 FP32
- softmax、norm 统计等敏感算子：BF16 或 FP32
- 不建议优先使用 FP16

BF16 的指数范围接近 FP32，通常比 FP16 更稳定，也不需要依赖复杂的 loss scaling。

第一版 BF16 训练应先建立以下基线：

- loss 曲线
- tokens/s
- 峰值显存
- checkpoint 保存和恢复
- 多节点稳定性
- 梯度范数
- 验证集 loss

---

## 3.2 FP8：后续主要加速方向

MI300X 支持 FP8。合理方式不是把整个模型粗暴改成 FP8，而是主要将 GEMM 密集的线性层改成 FP8。

适合使用 FP8 的部分：

- Q projection
- K projection
- V projection
- attention output projection
- MLP gate projection
- MLP up projection
- MLP down projection

建议继续保留 BF16 或 FP32 的部分：

- RMSNorm
- Softmax
- RoPE
- loss
- optimizer states
- 部分缩放和统计操作

推荐部署顺序：

1. 先使用 BF16 跑出稳定基线
2. 使用相同数据训练一段 FP8
3. 比较 train loss
4. 比较 validation loss
5. 比较 gradient norm
6. 检查 NaN/Inf
7. 检查 checkpoint 恢复后的连续性
8. 确认没有数值偏离后再进行正式长训

FP8 的理论 GEMM 吞吐很高，但端到端收益会受到 attention、通信、数据加载和 optimizer 等环节限制。

因此，实际收益必须以测试为准。

---

# 4. 训练框架选择

## 4.1 AMD ROCm Megatron-LM

如果目标是追求最高训练吞吐，AMD 官方优化的 ROCm Megatron-LM 是非常值得测试的方案。

优点：

- 针对 MI300X 做了专门优化
- 支持 BF16 和 FP8
- 支持 Transformer Engine
- 支持 GEMM tuning
- 支持 `torch.compile`
- 支持 Tensor Parallel
- 支持 Sequence Parallel
- 支持 Context Parallel
- 支持 Pipeline Parallel
- 支持 Distributed Optimizer
- 集成 FlashAttention 2
- 提供 fused kernels
- 有 Llama 3 8B 等预训练参考配置

如果当前 FSDP2 栈吞吐不理想，可以把 ROCm Megatron-LM 作为对照基准。

---

## 4.2 PyTorch FSDP2

FSDP2 的优点：

- 原生 PyTorch
- 代码结构清晰
- 修改模型结构方便
- 支持参数、梯度和 optimizer state 分片
- 与 DTensor 和 DeviceMesh 集成
- 适合模型研发和自定义架构

但 FSDP2 的性能高度依赖：

- wrapping 粒度
- all-gather 调度
- reshard 策略
- mixed precision policy
- 通信组大小
- activation checkpoint
- `torch.compile`
- attention kernel
- optimizer 实现

如果这些设置不合理，FSDP2 可能出现：

- 显存峰值高
- all-gather 跨节点
- 通信无法与计算重叠
- 每层频繁等待
- batch size 无法扩大
- GPU 利用率不高

---

## 4.3 DeepSpeed

DeepSpeed 可以使用，但在 8B + MI300X 192GB 的场景中，不建议优先使用 CPU 或 NVMe offload。

原因：

- MI300X 显存容量大
- 8B 模型参数状态本身通常不是主要问题
- offload 会把瓶颈转移到 PCIe、CPU 内存或 NVMe
- 容易严重降低训练吞吐

DeepSpeed 更适合在以下情况下使用：

- 团队已有成熟 DeepSpeed 平台
- 需要特定 ZeRO 功能
- 需要配套容错或调度系统
- 模型未来显著扩展

---

# 5. 128 卡时的并行策略

假设硬件是：

```text
16 台服务器 × 每台 8 张 MI300X
```

## 5.1 方案 A：纯数据并行或 FSDP

```text
TP = 1
PP = 1
CP = 1
DP = 128
Distributed Optimizer = On
```

适合：

- sequence length 为 4K 或 8K
- 单卡能够容纳合理 micro-batch
- 网络性能较好
- 不需要长上下文 Context Parallel

优点：

- 避免 TP 的高频通信
- 架构简单
- Pipeline bubble 为零

---

## 5.2 方案 B：节点内 TP=2

```text
TP = 2
PP = 1
DP = 64
Sequence Parallel = On
```

适合：

- micro-batch 较小
- activation 显存较大
- TP=2 后 GEMM shape 更适合硬件
- 节点内互联明显优于节点间网络

TP group 尽量不要跨节点。

---

## 5.3 方案 C：长上下文

对于 32K、64K 等长上下文，可以考虑：

```text
TP = 2 或 4
CP = 2、4 或 8
PP = 1
其余维度用于 DP
```

Context Parallel 主要适用于长上下文。

对于 4K 或 8K，上 CP 不一定有收益，通信开销可能大于显存收益。

---

## 5.4 Pipeline Parallel

8B 模型一般不建议使用 Pipeline Parallel。

原因：

- 容易产生 pipeline bubble
- 调度复杂
- checkpoint 复杂
- 多节点稳定性成本更高
- 8B 通常不需要靠 PP 才能装下

因此默认建议：

```text
PP = 1
```

---

# 6. 当前 32 卡 FSDP2 配置分析

当前情况：

```text
GPU 数量：32
总吞吐：190,000 tokens/s
每卡 micro-batch：2
精度：BF16
框架：FSDP2
micro-batch=4：OOM
```

这说明当前首先要解决的是显存峰值问题。

对于 8B 模型，32 路 FSDP 后：

- 参数被分片
- 梯度被分片
- optimizer state 被分片

因此，常驻模型状态通常不会占满 192GB HBM。

更可能的显存来源：

1. Transformer 激活
2. attention 中间 tensor
3. FSDP all-gather 后的完整参数
4. forward 和 backward 参数驻留重叠
5. backward communication buffer
6. optimizer 临时 buffer
7. `torch.compile` 工作区
8. Triton 或 fused kernel workspace
9. 完整 attention mask
10. 被保存的 hidden states 或 attention weights
11. allocator fragmentation

---

# 7. 先定位 OOM 来源

建议在 batch=2 的稳定配置下记录显存峰值。

```python
torch.cuda.reset_peak_memory_stats()

# 执行完整的 forward、backward、optimizer step
...

allocated = torch.cuda.max_memory_allocated() / 1024**3
reserved = torch.cuda.max_memory_reserved() / 1024**3

print(f"peak allocated: {allocated:.2f} GiB")
print(f"peak reserved:  {reserved:.2f} GiB")
```

ROCm 上 PyTorch 的 API 仍使用 `torch.cuda` 命名。

建议分别记录：

- forward 后
- backward 后
- optimizer step 前
- optimizer step 后
- 第一个 iteration
- warm-up 后第 10 个 iteration

判断方式：

### 情况 A：allocated 已接近 180GB

说明真实 tensor 和工作区确实接近显存上限。

主要方向：

- activation checkpoint
- fused attention
- 降低 all-gather 峰值
- selective recomputation
- 优化 optimizer

### 情况 B：allocated 约 130GB，但 reserved 接近 190GB

说明可能存在：

- allocator fragmentation
- 编译缓存
- 大小不均匀的临时 tensor
- workspace 没有及时复用
- 动态 shape 导致反复申请

此时应重点检查：

- shape 是否固定
- 是否每步发生重新编译
- 是否有大 tensor 生命周期重叠
- allocator 配置
- 是否存在动态 sequence length

---

# 8. 最高优先级：Activation Checkpointing

如果目前没有 activation checkpointing，应优先启用。

建议使用 non-reentrant checkpoint，并以 Transformer block 为单位。

示意代码：

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
)

def check_fn(module):
    return isinstance(module, TransformerBlock)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
        module,
        checkpoint_impl="no_reentrant",
        preserve_rng_state=False,
    ),
    check_fn=check_fn,
)
```

注意：

- 具体 API 可能随 PyTorch 版本变化
- 优先使用 non-reentrant
- 以 Transformer block 为单位
- 不建议 checkpoint 整个模型
- dropout=0 时可测试 `preserve_rng_state=False`
- 检查是否与 `torch.compile` 产生 graph break

建议测试：

| 配置 | Micro-batch | 目标 |
|---|---:|---|
| 无 checkpoint | 2 | 当前基线 |
| 每两层 checkpoint | 3 或 4 | 降低重计算 |
| 每层 checkpoint | 4 | 最大显存节省 |
| selective checkpoint | 4 | 平衡显存和算力 |

最终比较的是有效 tokens/s，而不是显存最低。

---

# 9. Selective Activation Checkpointing

如果已经每层 checkpoint，可以进一步测试 selective checkpointing。

目标是：

- 保存体积小但重算昂贵的 tensor
- 重算体积大但计算便宜的中间结果

可尝试：

- checkpoint attention
- checkpoint MLP
- 只保存 block 输入
- 不保存 attention probability
- 避免重算特别昂贵的部分

Selective checkpoint 通常比“所有层完整重算”更容易获得较好的吞吐。

---

# 10. 确保 Attention 使用高性能 fused kernel

这是最关键的检查项之一。

避免显式执行：

```python
attention_scores = q @ k.transpose(-1, -2)
attention_probs = softmax(attention_scores)
output = attention_probs @ v
```

这会显式产生：

```text
[batch, heads, sequence, sequence]
```

级别的巨大 tensor。

优先使用：

```python
torch.nn.functional.scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True,
)
```

或者 ROCm 上经过验证的 FlashAttention 2。

需要通过 profiler 确认真正执行的 kernel，而不是只看配置项。

检查内容：

- 是否显式构造 attention score
- 是否物化完整 causal mask
- 自定义 mask 是否导致回退
- GQA 是否走 fused path
- head dimension 是否为 64 或 128
- Q/K/V layout 是否合理
- 是否频繁发生 transpose
- 是否频繁调用 contiguous
- 是否存在 BF16 和 FP32 来回 cast
- attention dropout 是否影响 kernel
- 是否返回 attention weights

如果 attention probability 被完整保存，batch=4 OOM 很常见，尤其在 8K 或更长序列下。

---

# 11. FSDP2 Wrapping 粒度

FSDP2 应对每一个 Transformer block 进行分片，而不是只在模型最外层分片一次。

示意：

```python
from torch.distributed.fsdp import fully_shard

for block in model.layers:
    fully_shard(
        block,
        mesh=dp_mesh,
        reshard_after_forward=True,
    )

fully_shard(
    model,
    mesh=dp_mesh,
    reshard_after_forward=True,
)
```

正确的 wrapping 粒度可以：

- 控制每次 all-gather 的参数规模
- 降低多个完整层同时驻留的峰值
- 改善通信与计算重叠
- 避免全模型级别的大峰值

---

# 12. `reshard_after_forward`

当前受 OOM 限制时，建议优先使用：

```text
reshard_after_forward = True
```

优点：

- forward 后立即释放完整参数
- 降低 forward 和 backward 之间的显存占用
- 更容易将 micro-batch 提升到 3 或 4

缺点：

- backward 前需要重新 all-gather
- 会增加通信量

这是典型的显存换通信策略。

当前最重要的目标是先让 batch=3 或 batch=4 跑起来，再比较端到端吞吐。

---

# 13. 检查常见 FSDP2 错误

需要确认：

- embedding 没有意外保留完整副本
- LM head 没有意外保留完整副本
- tied embedding 没有在 FSDP 后失去共享
- 所有 Transformer block 都进入了 sharding
- optimizer 在 `fully_shard()` 之后创建
- optimizer 没有持有旧参数
- 没有保存全部 hidden states
- 没有输出 attention weights
- `use_cache=False`
- 没有重复保存 model output
- gradient clipping 没有触发完整梯度聚合
- checkpoint 没有导致参数重复驻留

建议配置：

```python
model.config.use_cache = False
output_hidden_states = False
output_attentions = False
```

---

# 14. 尝试 Micro-batch=3

不要只测试 batch size 2 和 4。

batch size 不需要是 2 的幂。

建议测试：

| Micro-batch | Checkpoint | 目标 |
|---:|---|---|
| 2 | 当前配置 | 基线 |
| 3 | 当前配置 | 检查是否可运行 |
| 3 | selective checkpoint | 测吞吐 |
| 4 | full checkpoint | 测吞吐 |

batch=3 可能改善：

- GEMM 的 M 维度
- kernel occupancy
- 固定开销摊销
- 通信与计算比例

并且显存压力明显低于 batch=4。

---

# 15. FSDP Mixed Precision Policy

不能只依赖 autocast。

需要确保 FSDP2 mixed precision policy 也正确配置。

示意：

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
)

fully_shard(
    block,
    mesh=mesh,
    mp_policy=mp_policy,
    reshard_after_forward=True,
)
```

重点检查：

- 参数计算是否 BF16
- gradient reduce 是否 BF16
- 是否意外使用 FP32 通信
- norm 和 loss 是否保持合适精度
- optimizer state 是否为 FP32
- 是否额外保留了完整 FP32 master parameters

如果 gradient reduce 意外使用 FP32，会增加：

- 通信量
- 通信时间
- 临时显存

---

# 16. `torch.compile`

如果当前未使用，可以测试对 block 进行编译。

示意：

```python
for i, block in enumerate(model.layers):
    model.layers[i] = torch.compile(
        block,
        dynamic=False,
        fullgraph=True,
    )
```

实际调用顺序应根据当前 PyTorch 和 ROCm 版本验证。

重点不是代码形式，而是以下条件：

- sequence shape 固定
- micro-batch 固定
- 没有频繁 graph break
- 没有 iteration 间重新编译
- compiled kernel 没有替换掉更快的 fused attention
- 编译后峰值显存没有明显上升

可使用：

```bash
TORCH_LOGS="graph_breaks,recompiles"
```

训练 shape 应尽量固定：

- 固定 sequence length
- 固定 micro-batch
- packing 后保持固定 tensor shape
- 避免每个 batch 使用不同 mask shape

---

# 17. Fused RMSNorm

应使用 fused RMSNorm，而不是多个 PyTorch operator 组合。

推荐结构：

- Pre-RMSNorm
- RMSNorm
- 避免 LayerNorm
- 使用经过验证的 epsilon

例如：

```text
1e-5
```

或：

```text
1e-6
```

具体取值需要依据模型设计和训练稳定性验证。

---

# 18. Fused RoPE

RoPE 应尽量融合以下过程：

- Q/K reshape
- position index
- rotary application
- layout transform

避免在主路径出现大量：

- transpose
- contiguous
- temporary tensor
- dtype cast

---

# 19. Fused SwiGLU

推荐使用 fused 版本的：

```text
SiLU(gate) * up
```

这样可以减少：

- kernel launch
- HBM 读写
- 临时激活
- pointwise operator 开销

---

# 20. Optimizer 优化

检查 AdamW 使用的实现：

- fused
- foreach
- 普通逐 tensor 实现

示意：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.95),
    fused=True,
)
```

前提是当前 PyTorch/ROCm 版本确实稳定支持。

也可以测试：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    foreach=False,
)
```

原因是：

- `foreach=True` 有时更快
- 但可能建立额外 tensor list
- 可能增加 optimizer step 显存峰值

应分别记录 OOM 发生位置：

- forward
- backward
- gradient clipping
- optimizer step
- optimizer state initialization

如果 OOM 只发生在 optimizer step，activation checkpointing 可能帮助有限。

此时重点检查：

- foreach buffer
- gradient norm
- FP32 master gradient
- optimizer state 初始化
- 完整参数 materialization

---

# 21. Gradient Clipping

普通写法：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm,
)
```

需要确认对 FSDP2 或 DTensor 参数使用的是分布式兼容实现。

潜在风险：

- all-gather 完整梯度
- 建立 FP32 gradient copy
- 每层重复同步
- 引入额外显存峰值

可临时关闭 gradient clipping 跑几十步，只用于性能和显存诊断。

需要比较：

```text
启用 clipping
vs
关闭 clipping
```

正式长训不建议未经验证永久关闭梯度保护。

---

# 22. 32-way FSDP 是否合理

如果 32 卡分布在 4 个节点，每节点 8 卡，那么当前纯 32-way FSDP 可能不是最优。

当前方案：

```text
FSDP shard group = 32
DP replica = 1
```

这种方式可能导致：

- 每层 all-gather 跨节点
- 通信延迟较高
- 节点间网络成为瓶颈
- 计算粒度变小
- 每张 GPU 的工作量不足

---

# 23. HSDP 方案

可以测试：

```text
FSDP shard group = 8
replicate group = 4
```

含义：

- 节点内 8 卡分片
- 4 个节点之间复制
- 节点间主要做梯度同步

优点：

- 参数 all-gather 尽量限制在节点内
- 避免每层跨节点 all-gather
- 更适合单节点高速互联拓扑
- 可能显著降低通信延迟

缺点：

- 每卡模型状态占用增加
- activation 可用空间减少
- micro-batch=4 可能更难容纳

因此建议配合 activation checkpoint 测试。

---

# 24. RCCL 和网络优化

32 卡训练时，需要确认网络不是主要瓶颈。

建议分别测试：

- 8 卡单节点
- 16 卡
- 32 卡

测试 collective：

- all-reduce
- all-gather
- reduce-scatter
- 实际 FSDP bucket size 对应的 message size

重点检查：

- GPU Direct RDMA
- GPU/NIC affinity
- NUMA affinity
- 多 NIC 是否均匀使用
- 是否出现流量集中
- RoCE 配置
- PFC
- ECN
- MTU
- 交换机 oversubscription
- RCCL topology
- 节点内和节点间带宽差异

---

# 25. 扩展效率

建议用 8 卡作为基线。

假设：

```text
8 卡吞吐 = 60,000 tokens/s
32 卡吞吐 = 190,000 tokens/s
```

扩展效率：

```text
190,000 / (60,000 × 4)
≈ 79.2%
```

经验判断：

- 85% 以上：扩展效率很好
- 75%～85%：可以接受
- 65%～75%：通信或调度值得优化
- 65% 以下：应优先排查网络和 FSDP 组配置

如果扩展效率较低，优先检查：

- FSDP 跨节点 all-gather
- RCCL collective 性能
- gradient bucket
- 通信与计算重叠
- NIC/GPU affinity
- 数据加载
- 每卡计算粒度太小

---

# 26. 数据 Pipeline

大规模训练很容易因为数据读取不足导致 GPU 空洞。

应记录：

```text
data loading time
forward time
backward time
optimizer time
total step time
```

建议：

- 数据预先 tokenize
- 不在训练时解析大量 JSONL
- 使用 memory-mapped binary
- 使用大 shard
- 每个 rank 独立读取
- 本地 NVMe cache
- 异步 prefetch
- persistent workers
- pinned memory
- 避免 CPU 端复杂预处理

---

# 27. Sequence Packing

Sequence packing 是非常重要的有效吞吐优化。

不要将每个文档单独 padding 到固定长度。

应将多个文档拼接到一个训练 sequence 中，同时维护：

- document boundary
- position ID
- loss mask
- EOS token
- attention boundary 策略
- 是否允许跨文档 attention

有效 token 比例：

```text
参与 loss 的 token 数
---------------------
送入模型的 token 总数
```

如果 padding 和无效 token 占 10%，那么：

```text
190,000 raw tokens/s
```

实际有效训练吞吐只有约：

```text
171,000 effective tokens/s
```

因此应同时记录：

- raw tokens/s
- loss-bearing tokens/s

---

# 28. Tokenizer 和 Vocabulary Size

词表大小也会影响训练速度。

一般情况：

- 32K：embedding 和输出层较轻
- 64K：更适合多语言和代码
- 128K：输出 projection 成本明显增加

对于 8B 模型，如果 vocabulary 很大，最终 LM head 可能成为显著计算瓶颈。

应根据：

- 语言数量
- 中文比例
- 代码比例
- 特殊 token 数量
- tokenizer 压缩效率

来确定词表大小，而不是机械照搬其他模型。

---

# 29. 模型结构优化

推荐现代 Llama 类结构：

- Decoder-only
- Pre-RMSNorm
- RoPE
- SwiGLU
- GQA
- 尽量无 bias
- hidden size 对齐硬件友好维度
- FFN size 对齐硬件友好维度
- vocabulary size 对齐 GEMM 友好倍数

---

# 30. GQA

例如：

```text
Query heads = 32
KV heads = 8
```

GQA 可以：

- 减少 K/V projection 计算
- 减少 activation
- 降低长上下文显存
- 改善推理 KV cache

在预训练时收益小于推理，但仍有价值。

---

# 31. 不建议第一版使用 MoE

MoE 虽然能够以较低 active parameters 获得更大总参数量，但会引入：

- all-to-all
- token routing
- load balancing
- expert capacity
- dropped token
- checkpoint 复杂性
- 容错复杂性
- 通信热点

如果目标是稳定训练约 8B 模型，第一版建议使用：

```text
8B Dense
```

而不是复杂 MoE。

---

# 32. Activation Checkpointing 的性能取舍

Activation checkpointing 不一定直接提高单步速度。

它的作用是：

- 降低显存
- 提高 micro-batch
- 改善 GEMM shape
- 增加 GPU 利用率

但代价是：

- backward 时重算 forward
- 增加 FLOPs

因此需要以最终吞吐判断，而不是只看显存。

推荐测试：

```text
无 checkpoint，batch=2
每两层 checkpoint，batch=3
每层 checkpoint，batch=4
selective checkpoint，batch=4
```

---

# 33. FP8 不应作为当前第一优先级

对于你目前的 32 卡 FSDP2 配置，优先级应是：

1. 确认 FlashAttention/fused SDPA
2. Activation checkpointing
3. FSDP wrapping 粒度
4. `reshard_after_forward`
5. micro-batch=3
6. HSDP
7. `torch.compile`
8. fused optimizer
9. 数据 packing
10. 最后测试 FP8

原因是 FP8 并不能自动解决：

- FSDP all-gather
- 数据读取
- attention fallback
- allocator fragmentation
- optimizer 峰值
- 通信延迟
- 不合理的 wrapping

---

# 34. 推荐的第一轮 32 卡实验矩阵

固定：

- 模型
- sequence length
- 数据
- global batch tokens
- warm-up steps
- logging 方式

建议至少测试：

| 实验 | Micro-batch | Checkpoint | FSDP Group | Compile |
|---|---:|---|---:|---|
| A | 2 | 当前配置 | 32 | 当前配置 |
| B | 3 | 当前配置 | 32 | 当前配置 |
| C | 4 | 每层 block | 32 | 当前配置 |
| D | 4 | selective | 32 | 当前配置 |
| E | 2 或 3 | selective | 8-way HSDP | 当前配置 |
| F | 最优 batch | 最优配置 | 最优配置 | 开启 |

每组至少记录：

```text
raw tokens/s
effective tokens/s
step time
forward time
backward time
optimizer time
data loading time
peak allocated memory
peak reserved memory
RCCL communication time
GPU utilization
HBM bandwidth
MFU
loss
gradient norm
```

---

# 35. 最推荐的当前配置方向

基于当前：

```text
FSDP2
BF16
32 × MI300X
micro-batch=2
micro-batch=4 OOM
190k tokens/s
```

优先尝试：

```text
BF16
FlashAttention 2 或 fused SDPA
每层或 selective activation checkpoint
micro-batch = 3
然后尝试 micro-batch = 4
FSDP2 每个 Transformer block 单独 fully_shard
reshard_after_forward = True
关闭 use_cache
关闭 output_attentions
关闭 output_hidden_states
FSDP reduce_dtype = BF16
固定 sequence shape
torch.compile A/B
Pretokenized 数据
Sequence packing
```

如果 32 卡是 4 个 8 卡节点，再重点测试：

```text
8-way 节点内 FSDP
×
4-way replicate HSDP
```

---

# 36. 预期优化空间

当前吞吐：

```text
190k tokens/s
```

在以下问题得到修复后：

- attention kernel 回退
- activation checkpoint 不合理
- FSDP wrap 粒度不合理
- 跨节点 all-gather 过多
- 数据 padding 较高
- micro-batch 太小
- `torch.compile` 未生效
- optimizer 产生大峰值

可以争取的工程目标大致为：

```text
230k～300k tokens/s
```

这只是合理的目标区间，不是性能保证。

实际空间主要取决于：

- sequence length
- 模型结构
- 当前 GPU 利用率
- 节点数量
- 网络带宽
- attention kernel
- packing 比例
- checkpoint 重算开销
- FSDP 通信方式

---

# 37. 128 卡正式训练前的验证顺序

## 阶段 1：单卡

验证：

- loss 是否正常下降
- 单卡 tokens/s
- 峰值显存
- attention kernel
- operator 时间分布
- checkpoint 恢复

## 阶段 2：单机 8 卡

比较：

```text
DP=8
TP=2, DP=4
TP=4, DP=2
FSDP=8
```

找出单节点最快组合。

## 阶段 3：扩展测试

依次运行：

```text
8 卡
16 卡
32 卡
64 卡
128 卡
```

计算 scaling efficiency。

## 阶段 4：FP8 A/B

比较：

- BF16 tokens/s
- FP8 tokens/s
- loss
- validation loss
- gradient norm
- NaN/Inf
- checkpoint 恢复稳定性

## 阶段 5：稳定性测试

连续运行至少 24～72 小时，检查：

- RCCL hang
- memory leak
- reserved memory 缓慢增长
- checkpoint 可恢复
- data shard 不重复
- data shard 不遗漏
- 节点故障恢复
- evaluation 是否阻塞
- 日志系统是否阻塞

---

# 38. 训练计算量粗略估算

Dense Transformer 预训练 FLOPs 常用粗略公式：

```text
Training FLOPs ≈ 6 × 参数量 × token 数
```

例如：

```text
参数量 = 8B
训练 token = 2T
```

则：

```text
6 × 8B × 2T
≈ 9.6 × 10^22 FLOPs
```

实际训练时间取决于：

- BF16 或 FP8
- 实际 MFU
- 多节点扩展效率
- checkpoint 重算
- 数据等待
- evaluation
- checkpoint 保存
- 故障停机
- 网络通信

因此，正式预算最重要的输入是：

```text
稳定运行的有效 tokens/s
```

而不是理论峰值。

---

# 39. 最值得优先投入的五项

综合优先级：

1. 确认 FlashAttention 2 或 fused SDPA 真正生效
2. 使用正确的 activation checkpointing
3. 优化 FSDP2 wrapping 和 reshard 策略
4. 测试节点内 FSDP + 节点间 replicate 的 HSDP
5. 使用 sequence packing，并记录 effective tokens/s

之后再测试：

- `torch.compile`
- fused optimizer
- FP8 Transformer Engine
- Megatron-LM 对照基准

---

# 40. 最终结论

对于 8B 模型和 MI300X：

- 模型参数通常不是主要显存问题
- 激活和 FSDP all-gather 峰值更值得关注
- micro-batch=4 OOM 不代表必须继续停留在 batch=2
- batch=3 是非常值得尝试的中间点
- activation checkpoint 可能允许 batch=4
- 32-way FSDP 不一定比节点内 8-way FSDP 更快
- attention kernel 是否真正 fused，可能直接决定显存和吞吐
- FP8 应放在 BF16 基础栈优化完成之后
- raw tokens/s 必须和 effective tokens/s 分开统计
- 最终优化目标应是稳定、可恢复和高有效吞吐，而不是单项 benchmark 峰值
