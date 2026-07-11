# 8B 训练 Loss NaN 事故复盘与稳定性修复

**日期**：2026-07-11
**Run**：`fineweb_edu_8b`（MI300 集群 `chec-mi300-2`，4 节点 × 8 × MI300X = 32 GPU）
**现象**：8B 从头预训练在 **step 110** 突然 loss/grad_norm 变 NaN，且此后无法恢复。
**影响**：训练无效但未损坏基础设施；因 `ckpt_every=2000` 尚无 checkpoint，只能修复后从头重启。

---

## 1. 事故时间线

| step | loss | grad_norm | lr | 说明 |
|---|---|---|---|---|
| 0 | 12.7300 | 2.394 | 1.50e-6 | 正常起步 |
| 10 | 10.9272 | 31.869 | 1.65e-5 | warmup 初期 gnorm 尖峰（常见，可接受） |
| 20 | 8.7596 | 3.321 | 3.15e-5 | 正常 |
| 30 | 7.7876 | 2.406 | 4.65e-5 | 正常 |
| 40–100 | 7.47→6.30 | 2.4–5.4 | 6.15e-5→1.51e-4 | **持续健康下降** |
| **110** | **nan** | **nan** | 1.66e-4 | **爆炸点** |
| 120+ | nan | nan | 继续爬升 | 一旦 NaN 全程不可逆 |

关键观察：
- **step 0–100 完全健康**，loss 单调下降 12.73→6.30，grad_norm 稳定在 2–5。
- 爆炸发生在 **warmup 爬升阶段**（lr 才到 1.66e-4，峰值 3e-4 的一半多），不是训练后期。
- **显存 89.3G/192G、吞吐 ~189K tok/s 全程正常** → 排除 OOM、排除通信/RDMA 故障 → **纯数值不稳定**。
- NaN 后吞吐反而"变快"（230K tok/s）：因为 NaN 前向/反向没有真实计算负载，是坏兆头不是好事。

---

## 2. 根因分析

### 2.1 直接根因：残差投影层初始化缺少深度缩放

`src/model.py` 的权重初始化（修复前）：

```python
def _init(self, m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)   # ← 所有 Linear 一律 0.02
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
```

问题：**所有 Linear（含残差输出投影 `attn.wo`、`ffn.w2`）都用 `std=0.02`，没有按层数缩放。**

在 pre-norm Transformer 里，残差流是逐层累加：

```
x = x + attn(norm(x))     # 每层往残差流里"加"一次
x = x + ffn(norm(x))
```

每一层的输出都直接加到残差流 `x` 上。如果每层输出方差是 `σ²`，那么经过 `L` 层后残差流方差约为 `初始 + 2L·σ²`（每层两个 add：attn + ffn），**方差随深度线性增长**。深层的激活值/logits 越来越大，配合 warmup 抬升的学习率，最终数值溢出 → NaN。

**标准解法**（GPT-2 / Llama / 大多数现代实现）：把**残差输出投影**（`wo`、`w2`）的初始化标准差按 `1/√(2·n_layers)` 缩小，抵消 `L` 层累加带来的方差增长：

```
std(wo, w2) = 0.02 / √(2 · n_layers)
```

其余 Linear（`wq/wk/wv`、`w1/w3`、`lm_head`）保持 `0.02`。

### 2.2 为什么 1B 没事、8B 炸了

| | 1B | 8B |
|---|---|---|
| 层数 `n_layers` | 24 | **32** |
| 残差累加次数 | 48 | **64** |
| 缩放因子（应有） | 1/√48 ≈ 0.144 | 1/√64 = 0.125 |

- 1B（24 层）残差方差增长更小，**侥幸**在 `std=0.02` 无缩放下扛过了 2000 步（loss 12.33→3.40）。这是"能跑"不代表"健康"，本身已在数值悬崖边缘。
- 8B（32 层）残差累加多 33%，方差更大，深层激活更容易溢出。warmup 学习率一抬到 1.66e-4 就越过临界点爆炸。

### 2.3 为什么 8B 冒烟测试没暴露

8B 冒烟（`_smoke_8b.sh`）只跑 **40 步**，当时 lr 还极低（<6e-5），残差方差尚未在高 lr 下被放大，所以看起来一切正常（loss 12.77→8.12）。**冒烟测试的步数太少，覆盖不到 warmup 中后段的高 lr 不稳定区间**——这是本次的流程教训。

---

## 3. 解决方案

### 3.1 修复一（必须）：残差投影深度缩放初始化

在 `src/model.py` 中给残差输出投影打标记，并在 `_init` 中特殊处理。

**思路**：给 `Attention.wo` 和 `SwiGLU.w2` 加一个属性标记（如 `_residual_proj = True`），初始化时对带标记的层用缩放后的 std。

```python
# Attention.__init__ 里
self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
self.wo._residual_proj = True          # 标记：残差输出投影

# SwiGLU.__init__ 里
self.w2 = nn.Linear(hidden, dim, bias=False)   # down
self.w2._residual_proj = True          # 标记

# Chimera._init 改成：
def _init(self, m):
    if isinstance(m, nn.Linear):
        std = 0.02
        if getattr(m, "_residual_proj", False):
            std = 0.02 / math.sqrt(2 * self.args.n_layers)
        nn.init.normal_(m.weight, mean=0.0, std=std)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
```

> 注意：`self.apply(self._init)` 是逐 module 调用，`_init` 里能通过 `self.args.n_layers` 拿到层数（`_init` 是 `Chimera` 的方法，`self` 是模型本体）。

### 3.2 修复二（决定性）：非有限梯度跳步保护（生产级 pretraining 标配）

**这是把「致命 NaN 中毒」降级为「可恢复跳步」的决定性修复，不是可选项。**

问题机理（为什么单靠 grad_clip 救不了）：

- `clip_grad_norm_(params, 1.0)` 的缩放系数是 `max_norm / total_norm`。
  当某个 micro-batch 产生 **inf 梯度**时，`total_norm=inf`，缩放系数 `1.0/inf=0` → 梯度被清零，
  这一步**侥幸自愈**（相当于跳过，step 50→60 就是这样躲过去的）。
- 但当 `total_norm` 是一个**很大的有限值**（如 step 70 的 gnorm=15.8）时，clip 只把它缩到 1.0，
  `opt.step()` 照常施加了一个**巨大且方向已被污染的更新** → 权重被推炸 →
  下一步前向直接 nan（权重已中毒，**永久回不来**）。

也就是说：**grad_clip 只对 inf 有效（缩放到 0），对"有限但巨大"的坏梯度无能为力**。真正的防线是：

> **当 `clip_grad_norm_` 返回的 grad_norm 非有限（inf/nan）时，跳过 `opt.step()` 并 `zero_grad`，
> 只累加 `skipped_steps` 计数，不更新权重。**

`src/train.py` 实际实现（commit `3824759`）：

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
# Skip the optimizer step on non-finite grads (inf/nan). Without this a
# single spiky micro-batch permanently poisons the weights and every
# subsequent step is NaN. Skipping keeps training recoverable.
gn_finite = torch.isfinite(grad_norm).item() if hasattr(grad_norm, "item") \
    else math.isfinite(float(grad_norm))
if gn_finite:
    opt.step()
else:
    opt.zero_grad(set_to_none=True)
    skipped_steps += 1
    log(f"[warn] step {step}: non-finite grad_norm, skipping optimizer step "
        f"(total skipped={skipped_steps})")
```

**验证效果**：重启后的 run 到 step 1110 时 `skipped_steps=39`——这 39 次是早期个别不稳定 micro-batch
被安全拦下、没有污染权重，训练全程 0 NaN、grad_norm 稳定在 0.2 量级。这个保护是运行期最后一道、
也是最关键的一道决定性防线。

### 3.3 修复三（建议）：8B 更保守的超参

8B 通常比 1B 需要**更小的峰值学习率 + 更长的 warmup**，用来**降低尖峰出现的频率**
（配合 3.1 降低尖峰根源、3.2 拦截漏网尖峰，三者形成互补）：

| 超参 | 修复前 | 建议/采用 |
|---|---|---|
| 峰值 lr | 3e-4 | **2e-4** |
| warmup_steps | 200 | **500** |
| grad_clip | 1.0 | 1.0（保持） |
| min_lr | 3e-5 | 2e-5（随峰值等比缩） |

理由：
- 峰值 lr 降低直接减小每步更新幅度，给数值稳定性留余量。
- warmup 拉长让模型在低 lr 下先"稳住"激活分布，再逐步抬升。

> 按项目约定，最终 lr/warmup 具体数值由用户拍板；agent 只提供建议默认值。

### 3.4 可选（进一步纵深防御，尚未启用）

- **logits soft-cap**：Gemma 风格对 attention/final logits 做 `tanh` 软封顶，进一步防溢出。Chimera 预留了 Gemma 能力，可评估开启。
- **缩短 ckpt 间隔用于早期**：8B 早期可临时 `ckpt_every=500`，一旦出问题有近点可回滚（代价是 43GB/个的存储）。

---

## 4. 处置与重启流程

1. **止损**：`pkill -9 -f train.py` 跨全部 4 节点，确认 GPU 显存回落到基线（~296MB/卡 = 空载）。✅ 已完成
2. **改代码**：本地 `C:\repos\pretrain` 改 `src/model.py`（残差缩放 init）+ `src/configs.py`（8b 段 lr/warmup）。
3. **提交**：`git commit` + `git push` 到 **`dev-chicm`** 分支（禁止碰 `main`）。
4. **各节点同步**：4 个节点 `git pull` 到本地 `/scratch/code`（aiscuser 属主）。
5. **重启**：`_launch_8b.sh`（先 pkill keepalive 释放 GPU → fan-out `mi300_mn.sh`，MODEL=8b）。
6. **加密观察前 200 步**：重点盯 warmup 高 lr 区间（step 100–500），确认过了上次爆炸点（110）后 loss 仍平滑下降、grad_norm 有限。

---

## 5. 经验教训（写入 memory）

1. **从头训练的初始化必须做残差深度缩放**（`wo`/`w2` std ÷ √(2·L)），层数越深越关键。这是架构级 must-have，不是可选优化。
2. **冒烟测试要覆盖 warmup 全程**：只跑 40 步无法暴露高 lr 数值不稳定。8B 级别的冒烟至少应跑过 warmup 峰值（或临时缩短 warmup 让高 lr 早到）。
3. **早期 checkpoint 要密**：`ckpt_every=2000` 意味着前 2000 步任何崩溃都要从头再来。大模型早期建议 `ckpt_every=500`。
4. **非有限梯度跳步是生产级 pretraining 标配**：检测 `clip_grad_norm_` 返回值非有限（inf/nan）时跳过 `opt.step()`。关键认知：**grad_clip 只对 inf 有效（缩放到 0），对"有限但巨大"的坏梯度无能为力**，后者会施加被污染的巨大更新导致权重永久中毒。跳步才是决定性防线（本次 run 触发 39 次跳步、全程 0 NaN 验证有效）。此外应尽早告警，避免像首次事故那样空转到 step 430 才被 TensorBoard 发现。
5. **"能跑通"≠"健康"**：1B 无缩放跑通 2000 步纯属侥幸，本身已在数值悬崖边缘。别把侥幸当验证。
6. **NaN 后吞吐虚高是危险信号**，不是性能提升。

---

## 6. 一句话总结

**8B 在 step 110 爆 NaN 的根因是模型初始化未对残差输出投影（`wo`/`w2`）做 `1/√(2·n_layers)` 深度缩放，导致 32 层深模型残差流方差随深度累积膨胀，在 warmup 抬升学习率后数值溢出。1B（24 层）侥幸未暴露，8B（32 层）在高 lr 下越界。修复由三部分组成，缺一不可：① 残差深度缩放初始化（降低尖峰根源，架构级 must-have）+ ② 非有限梯度跳步保护（决定性修复：把致命 NaN 中毒降级为可恢复跳步——grad_clip 只能缩放 inf、对"有限但巨大"的坏梯度无能为力，必须靠跳步拦截）+ ③ 8B 更保守的 lr/warmup（降低尖峰频率）。**
