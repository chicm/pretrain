# FP8 训练优化结果（MI300X, ROCm 7.1）

分支：`dev-fp8`。集群：`chec-test-env`（单节点 8× MI300X VF, ROCm 7.1.0）。
栈：torch 2.10-dev+rocm7.1 / **torchao 0.15.0** / flash_attn 2.8.3 / triton 3.4（`/opt/venv`）。

## TL;DR
旧「FP8 无收益」结论（ROCm 6.4.3, torchao 0.13）**是测试方法漏洞，不能确定是硬件/kernel 极限**。
拿到加速靠两个修复，旧测试两个都没做：
1. **fnuz dtype**：MI300 hipBLASLt 只支持 `float8_e4m3fnuz`/`e5m2fnuz`，OCP `e4m3fn` 一律
   `HIPBLAS_STATUS_NOT_SUPPORTED`（**ROCm 6 与 7 都一样**）。torchao 靠 `is_MI300()=True` 自动切 fnuz。
2. **torch.compile 强制**：eager fp8 比 bf16 慢 ~2×（未融合的 quant/scale 开销）；compile 后才快。

## 微基准（node-0 单卡, HIP_VISIBLE_DEVICES=0）

### 裸 `torch._scaled_mm`（`_probe_fnuz.py`）
| dtype | 结果 |
|---|---|
| e4m3fn（OCP）| ❌ NOT_SUPPORTED |
| **e4m3fnuz** | ✅ ~1.1 PFLOP/s，real MLP shape **1.7–1.9× over bf16** |

### torchao Float8Linear（SwiGLU MLP, d=4096 h=14336, M=8×4096）
| 配置 | ms/it | vs bf16 |
|---|---|---|
| bf16 | 37.7 | 1.00× |
| bf16 + compile | — | ~1.0× |
| fp8 tensorwise **eager** | 83.7 | **0.45×**（← eager 陷阱，旧测试踩的坑）|
| **fp8 tensorwise + compile** | **24.2** | **1.56×** ✅ 最快 |
| fp8 rowwise + compile | 27.2 | 1.39× |

## 端到端验证（8×MI300X 单节点）
- **1B**：fp8 loss 与 bf16 **逐位一致**（step0 12.348 vs 12.349）、无 NaN、显存更低
  （17.8 vs 20.2G）。168/169 Linear 转换（lm_head 正确跳过）。1B gemm 太小 → 吞吐无增益（符合预期）。
- **8B**：A/B 进行中（真实 FineWeb 数据 10.2B tok, micro_bsz=4, --fused_ce）。8B 才是能体现
  1.5× 的规模（微基准就是 8B MLP shape）。

## 代码
- `src/fp8_utils.py`：`convert_model_to_fp8(model, recipe)`，在 fully_shard/compile **之前**调用。
  只转 attn/MLP 大 Linear，跳过 lm_head；in/out 需 16 对齐。
- `src/train.py`：新增 `--fp8` / `--fp8_recipe {tensorwise,rowwise}`。`--fp8` 强制要求 compile。

## 待办
- [ ] 8B A/B 吞吐数字（mbsz=4, 真实数据）。
- [ ] **在主力 MI300 集群（ROCm 6.4.3）单卡重跑 `_probe_fnuz.py`** —— 一锤定音「版本问题 vs 方法问题」：
  若 ROCm6 上 fnuz `_scaled_mm` 也快 → 生产集群现在就能用 fp8，不必等升级。
- [ ] fp8 长跑 loss 曲线 vs bf16（收敛性确认后再进 mi300_mn.sh 默认）。
