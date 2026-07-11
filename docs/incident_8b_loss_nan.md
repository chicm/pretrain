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

### 3.2 修复二（建议）：8B 更保守的超参

8B 通常比 1B 需要**更小的峰值学习率 + 更长的 warmup**：

| 超参 | 修复前 | 建议 |
|---|---|---|
| 峰值 lr | 3e-4 | **2e-4** |
| warmup_steps | 200 | **500** |
| grad_clip | 1.0 | 1.0（保持） |
| min_lr | 3e-5 | 2e-5（随峰值等比缩） |

理由：
- 峰值 lr 降低直接减小每步更新幅度，给数值稳定性留余量。
- warmup 拉长让模型在低 lr 下先"稳住"激活分布，再逐步抬升。
- grad_clip=1.0 是最后一道防线，但它只在 grad_norm 是有限值时有用；一旦已经 NaN，clip 也救不回来 —— 所以根因修复（3.1）才是关键。

> 按项目约定，最终 lr/warmup 具体数值由用户拍板；agent 只提供建议默认值。

### 3.3 修复三（可选，纵深防御）

- **logits soft-cap**：Gemma 风格对 attention/final logits 做 `tanh` 软封顶，进一步防溢出。Chimera 预留了 Gemma 能力，可评估开启。
- **NaN 早停/跳过**：训练循环里检测 `loss.isnan()`，跳过该 step 的 optimizer.step（不更新），或直接告警退出，避免烧算力空转 400 步才被发现。
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
4. **加 NaN 守卫**：训练循环检测到 loss NaN 应立即告警/退出，而非空转数百步（本次空转到 step 430 才被 TensorBoard 发现）。
5. **"能跑通"≠"健康"**：1B 无缩放跑通 2000 步纯属侥幸，本身已在数值悬崖边缘。别把侥幸当验证。
6. **NaN 后吞吐虚高是危险信号**，不是性能提升。

---

## 6. 一句话总结

**8B 在 step 110 爆 NaN 的根因是模型初始化未对残差输出投影（`wo`/`w2`）做 `1/√(2·n_layers)` 深度缩放，导致 32 层深模型残差流方差随深度累积膨胀，在 warmup 抬升学习率后数值溢出。1B（24 层）侥幸未暴露，8B（32 层）在高 lr 下越界。修复 = 残差缩放初始化（必须）+ 8B 更保守的 lr/warmup（建议）+ NaN 守卫与更密 checkpoint（纵深防御）。**
