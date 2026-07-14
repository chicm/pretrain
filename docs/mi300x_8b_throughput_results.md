# 8B MI300X 吞吐优化 —— A/B 实测结果报告

对 `docs/mi300x_8b_pretraining_optimization.md` 指南中的候选优化项，在 **8B Chimera / MI300X ×32
（4 节点，IB 互联）** 上逐项 A/B 实测的结论。基线 = 原训练配置 189K tok/s（no-AC, mbsz=2,
grad_accum=8, fp32-reduce, 1D-FSDP2, no-compile）。

## TL;DR 最优配置

**no-AC + micro_bsz=3 + grad_accum=8 + fp32-reduce + 1D-FSDP2 + torch.compile ON → ~204K tok/s（较基线 +8%）。**
已固化为 `mi300_mn.sh` 默认（commit `1f773b1`）。

## 七方向 A/B 结果总表

| 方向 | tok/s vs 189K | 显存 | 代价 | 采纳 |
|---|---|---|---|---|
| **micro_bsz=3** | **202K (+6.9%)** | 131G | global batch 变 1.5× | ✅ **最大单项收益** |
| **torch.compile** | **191.2K (+1.2%)** | 89.3G | 首步一次性编译 | ✅ 采纳 |
| Selective AC, mbsz=6 | 196.4K (+4%) | 139G | 重算开销；仍 < mbsz=3；早期 gnorm inf | ❌ 留 flag（最佳 AC 变体）|
| bf16-reduce | 191K (+1%) | 89.3G | 32-way bf16 求和精度险 | ❌ 留 flag |
| HSDP shard=8×replicate=4 | 189.8K (0%) | +11.4G/卡 | 无收益 | ❌ 留 flag |
| Full AC, mbsz=12 | 175K (−7%) | 124.6G | 无脑重算 attention；ckpt 前缀污染 | ❌ |
| FP8 (torchao Float8Linear) | 1.00× (零加速) | — | 软件栈未成熟 | ❌ 推迟 |

## 根因：8B 在 MI300×32（有 IB）是**计算瓶颈**

GPU 满载 100%，主要 FLOPs（matmul）跑在高效率 rocBLAS/hipBLASLt 上。因此：
- **通信类优化（HSDP、bf16-reduce）净收益 ≈ 0** —— 计算已掩盖通信，prefetch 已重叠 all-gather。
- **显存类优化（各种 AC）净收益为负或次优** —— 重算是净增 FLOPs，在计算瓶颈下直接拖慢。
- **只有提升算术强度（更大 micro_bsz → GEMM M 维更大）和算子融合（torch.compile）真正有效。**

这是一个健康的「已接近最优」信号。

## 各项细节

### micro_bsz（最大收益）
mbsz 2→3：GEMM 的 M 维增大 → 算术强度/GPU 效率 ↑，grad_accum 步内通信被摊薄。
mbsz=4 = OOM（~180G）。mbsz=3 显存 131G，余量 ~60G 安全。
副作用：global batch 变 1.5×（原 2×8×32×4096 ≈ 2.1M/step → 3.15M/step）。如需保持 token/step，
把 grad_accum 从 8 调到 ~5。

### torch.compile
最重的 SDPA(→AOTriton FA2) 与 AdamW 本已 fused；RMSNorm/RoPE/SwiGLU 为手写未融合，靠 compile 自动 fuse。
零精度风险、显存持平、仅首步编译一次。唯一零代价正收益的算子融合路径，采纳。

### Selective AC (SAC)
实现（commit `c99c645`）：`checkpoint_wrapper(layer, NO_REENTRANT, preserve_rng_state=False,
context_fn=create_selective_checkpoint_contexts(policy))`，dropout=0 故 `preserve_rng_state=False`。
policy：MUST_SAVE = {mm, addmm, bmm, sdpa_flash, sdpa_efficient}（昂贵 matmul/SDPA 不重算）；
其余 PREFER_RECOMPUTE（便宜的逐元素/norm/RoPE 重算）。
- **SAC(196K) 显著优于 full-AC(175K)，+12%** —— 验证「只重算便宜算子」的指南洞察正确。
- 但保存 matmul 输出使显存节省有限：**mbsz=12 OOM，可用上限约 mbsz=6**。
- SAC mbsz=6 仍比 no-AC mbsz=3 慢 2.8%（计算瓶颈下重算即净增计算）。
- ⚠️ **观察风险**：SAC 运行时早期 gnorm 持续 inf（NaN guard 跳过），no-AC 早期为有限值；loss
  正常下降。若将来用 SAC 实训，需先确认该 inf 随 warmup 消失、不影响收敛。
- 结论：SAC 是最佳 AC 变体，作为 flag 保留，供未来 10B+ 显存吃紧时使用。

### FP8
微基准探路（`_fp8_probe*.py`，不碰训练代码）：`torch._scaled_mm` 直调返回
`HIPBLAS_STATUS_NOT_SUPPORTED`；torchao 0.13.0 `Float8Linear` 在 7 个大 Linear 尺寸上 GEMM
加速比 0.98–1.00×（零加速）。根因：torch 2.8 / ROCm 6.4.3 的 hipBLASLt FP8 kernel 尚未成熟。
分层策略（仅 Q/K/V/O + gate/up/down 用 FP8，norm/softmax/loss/optimizer 保 bf16/fp32）本身正确，
**推迟到 ROCm 7.x / 更新 torch 后重跑微基准再决策。**

## 保留的可选 flag（供未来 10B+/通信瓶颈场景）
`--activation_checkpoint`（full AC）、`--selective_ac`（SAC）、`--reduce_bf16`、`--hsdp_shard N`、
`COMPILE_FLAG`（compile 默认 ON）。

## 方法论
- **微基准探路**（如 FP8）在碰训练代码前先验证收益，避免盲改引入零加速 + NaN 风险的浪费。
- 计算瓶颈判定（GPU 100% + matmul 主导）能预判哪些方向注定无收益，聚焦真正有效的算术强度与算子融合。
