# data.py 向量化优化说明

> 本文解释 `src/data.py` 数据预处理的一次优化：**为什么改、改了什么、解决了什么问题**。

## 背景：这段代码在干什么

`data.py` 负责**预处理训练数据**：把原始文本（一篇篇文章）转成模型能吃的数字（token），再拼成一个大文件 `train.bin`。

举例：FineWeb `sample-10BT` 约有 **100 亿个 token**。

## 原来的写法及其问题

原实现大致如下：

```python
arr = np.zeros(总token数)          # 先开一个能装下 100 亿数字的大数组
idx = 0
for batch in 数据:
    for ids in batch:              # 一篇文章一篇文章地处理
        arr[idx : idx+len(ids)] = ids
        idx += len(ids)
arr.tofile("train.bin")            # 最后一次性写文件
```

存在两个问题：

### 问题 1：吃巨量内存
`np.zeros(100亿)` 会一次性在**内存里**开一个约 **20GB** 的数组。数据集越大，内存占用越大——大到一定程度直接 OOM（内存爆掉）崩溃。

### 问题 2：Python 逐篇循环慢
`for ids in batch` 是**纯 Python 循环**，要对几千万篇文章逐篇执行。Python 循环本身很慢（即"逐 doc Python 填充循环"）。TinyStories 小（4.7 亿 token）尚可忍受，FineWeb 大 20 倍就会慢得难受。

## 优化后的写法

```python
train_mm = np.memmap("train.bin", mode="w+")   # 直接映射到磁盘文件，不占内存
buf = []
for batch in 数据:
    buf.append(np.concatenate(batch))          # 一批文章一次性拼接（C 层，快）
    if 攒够约1600万token:
        train_mm[...] = 拼好的大块              # 整块写盘
```

对应解决两个问题：

### 解决 1：不再占 20GB 内存
`np.memmap` 是"**磁盘上的数组**"——数据直接落到磁盘文件，不在内存里堆着。数据集再大也不会 OOM。

### 解决 2：Python 循环换成向量化操作
"**向量化**"即：不用 Python 一条条处理，而是把一批数据交给底层 C 代码（`np.concatenate`）**一次性批量处理**。同样的活，C 层批处理比 Python 逐条快很多。循环粒度从"每篇文章"变成"每一大批"（约 1600 万 token 才落盘一次）。

## 一句话总结

> **让数据预处理既不爆内存、又快很多**，从而能顺利处理 FineWeb 这类上百亿 token 的大数据集；否则旧代码跑 FineWeb 很可能内存溢出或慢到不可用。

## 正确性验证

改完做了单元测试，构造 mock 文档，对比新旧两种写法的输出：

- ✅ 新写法产出的 `train.bin` 与旧写法**逐位完全一致**
- ✅ train/val 切分位置精确（val 取尾部，跨 flush 边界的文档也正确路由）

即：**只更快、更省内存，不改变数据本身。**

## 关联

- 代码：`src/data.py`（`prepare_data` 非流式路径、`_split_train_val` 辅助函数）
- 入口：`src/tokenize_data.py`
- 运行时注意（见 `docs/progress-2026-07.md`）：HF 下载缓存务必设到**本地快盘**，输出 `.bin` 写共享盘（大文件顺序写快）。
