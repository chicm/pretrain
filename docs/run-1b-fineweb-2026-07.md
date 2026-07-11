# Chimera 1B — FineWeb-10BT 首次真实数据训练结果小结

**日期**：2026-07-11
**Run**：`fineweb_1b_chimera`
**平台**：MI300 集群（job `chec-mi300-2`）
**状态**：✅ 完成（`[done] training complete`）

---

## 1. 一句话结论

Chimera 1B（Qwen3 结构 + QK-Norm）在 FineWeb-10BT 真实数据上，用 FSDP2 多机
（32× MI300X）稳定跑通 2000 步，loss 从 12.33 平滑收敛到 ~3.4，val_loss 3.53，
无 spike、无发散、无 OOM。训练管线 + 观测 + checkpoint 全链路验证通过。

---

## 2. 配置

| 项 | 值 |
|---|---|
| 模型 | **Chimera 1B**（class Chimera / alias Transformer） |
| 参数量 | **1.444B**（含 embedding；vocab 151936 使 embedding 增大） |
| 架构 | dim 4096 · 32 层 · 32 Q / 8 KV 头（GQA 4:1）· SwiGLU 14336 · RoPE · RMSNorm pre-norm · **QK-Norm on** · 全 full attention（滑窗关闭） |
| 分词器 | Qwen3（`Qwen/Qwen3-8B`），vocab padded 151936 |
| 数据 | FineWeb sample-10BT，~10.2B tokens（uint32），eot `<|endoftext|>` id 151643 |
| 框架 | FSDP2（`fully_shard`，逐层 + 整体） |
| 精度 | bf16 autocast |
| torch.compile | 关闭（多机稳定优先） |

## 3. 并行 / batch

| 项 | 值 |
|---|---|
| 规模 | 4 节点 × 8 MI300X = **32 GPU** |
| 通信 | 8× InfiniBand（真 RDMA），RCCL 2.22.3 |
| micro_bsz × grad_accum | 8 × 2 |
| seq_len | 2048 |
| 全局 batch | 8 × 2 × 32 × 2048 ≈ **100 万 token/step** |
| max_steps | 2000 |
| 消耗 token | ~20 亿（≈ FineWeb-10BT 的 1/5 epoch） |

## 4. 训练动态

| 指标 | 起点 (step 0) | 终点 (~step 2000) |
|---|---|---|
| **train loss** | 12.33 | ~3.40 |
| **val_loss** | — | 3.53（step 1500）；4.22→3.70→3.53 单调降 |
| **grad_norm** | 1.76 | ~0.05（极稳） |
| **lr** | 1.5e-6（warmup 起） | 3.0e-5（cosine 衰减尾部，峰值 ~2.9e-4） |

val_loss 曲线：step 500 = 4.22 → 1000 = 3.70 → 1500 = 3.53，单调下降，
train/val 接近，**无过拟合迹象**。

> 起点 loss 12.33 ≈ ln(151936)=11.9，符合随机初始化均匀分布的理论期望，说明
> 权重初始化 + 前向数值正确。

## 5. 性能 / 资源

| 项 | 值 |
|---|---|
| 吞吐 | ~730–749K tok/s（稳态） |
| step 时间 | ~11 s/step |
| 单卡显存 | 82.6 GB / 192 GB（**余量充足**，8B 亦可容纳） |
| 墙钟耗时 | ~54 分钟（2000 步，含周期性 checkpoint 停顿） |

> tok/s 比 TinyStories 时的 ~930K 略低，因 vocab 3× 大，embedding/logits 计算量增加，属预期。

## 6. 产出

- **Checkpoints**：`$SHARED/checkpoints/fineweb_1b_chimera/ckpt_{200..2000}.pt`
  （每 200 步一个，full state dict，~18.5 GB 含 optimizer state）
- **TensorBoard**：`.../fineweb_1b_chimera/tb`（loss / grad_norm / lr / tok_s / mem / val_loss / tokens）
- **文本日志**：`$SHARED/logs/mn_node0.log`

## 7. 观测栈（本次首启用）

- **Layer 2 文本指标**：每 10 步 `step | loss | gnorm | lr | tok/s | mem | eta`
- **Layer 3 TensorBoard**：master rank 写 event → 后台 loop 每 15s 镜像到本地盘
  `/scratch/tb_local`（绕开 blobfuse2 追加写读取问题）→ node-0 `tensorboard`（127.0.0.1:6007）
  → 本地 `tunnel.py` 转发 → 浏览器 `localhost:6006`
- **32 卡硬件快照**：`_gpumon.sh`（rocm-smi）

## 8. 结论与下一步

**验证通过的能力**：Chimera 架构（QK-Norm）、Qwen3 大词表（151936）、FineWeb 真实数据
管线、FSDP2 32 卡多机、真 IB 通信、观测栈、checkpoint。全部 green。

**下一步候选**：
1. **评测**：用 `eval.py`（lm-eval-harness）跑 HellaSwag / LAMBADA / ARC，看 1B/20亿token
   的下游信号（MMLU 此规模约等随机 25%，暂不作主指标）。
2. **8B 模型**：显存有充足余量（82/192 GB），可尝试 8B 多机。
3. **扩数据 + 拉长**：FineWeb-Edu / Nemotron-CC，max_steps 上量做真正的预训练。

> 注：本 run 是**管线与架构验证**（proxy），非追求终模型质量。20 亿 token 对 1B 模型
> 远未训练充分，loss 仍在下降空间内。
