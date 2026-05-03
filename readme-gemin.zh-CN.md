# UrbanEV AI 交接文档中文整理版

这份文件是对 `readme-gemin.md` 的中文整理版，方便人工阅读。后续让 Codex 继续读取原始 `readme-gemin.md` 或 `docs/handover.md` 即可。

## 1. 项目定位与当前状态

- 任务：城市级电动汽车充电负荷时空预测。
- 终极目标：开发融合“多模态动态图”和“时间分块 Patch”的归纳式时空模型，暂定名 `DyGraph-PatchFormer`。
- 目标论文级别：高水平期刊。

当前需要超越的 baseline：

- `DLinear`：MAPE `75.7%`，MAE `0.361`，`seq_len=96`
- `GWNET`：MAPE `13.08%`，MAE `0.173`，`seq_len=12`
- `PatchTST`：MAPE `11.0%`，MAE `0.118`，`seq_len=96`

## 2. 工程架构与约束

- `code/`：图时空模型阵营，主要处理 4D 张量 `[B, C, N, T]`
- `code-transformer/`：时序 Transformer 阵营，主要处理 3D 张量 `[B, L, N]`
- 后续开发主战场：`code/`

硬约束：

- 不直接修改第三方 baseline 源码，如 `GWNET`、`AGCRN`
- 维度对齐、接口适配、Bug 修复优先通过 `models/*_adapter.py` 完成
- 全局参数放 `parse.py`
- 数据装配和模型注册放 `utils.py`
- 训练与评估主循环放 `train.py`

## 3. 核心创新点

### 创新点一：多模态残差混合动态图

目标：

- 用天气、POI、时间戳等先验特征生成动态图残差
- 与静态图融合，增强稳定性和冷启动泛化能力

基本形式：

- 静态图：`A_static in R^{N x N}`
- 动态残差：`Delta A_dynamic in R^{B x P x N x N}`
- 融合图：`A_hybrid^(p) = A_static + lambda * Delta A_dynamic^(p)`

实现要求：

- 不走粗暴的全连接 `O(N^2)` 重构
- 更适合使用节点嵌入点积或注意力方式轻量生成

### 创新点二：Patch 级时空交替耦合

目标：

- 先把长序列切成 patch
- 在 patch 级别做空间聚合
- 再在 patch 维度做时间注意力

张量流：

1. 输入：`[B, C, N, L]`
2. 分块：`[B, C, N, P, P_len]`
3. 用 `A_hybrid^(p)` 在 `N` 维做 patch 内空间融合
4. 在 `P` 维做跨 patch 时间注意力

### 创新点三：节点级冷启动实验范式

目标：

- 模拟新建站点场景
- 输入端屏蔽部分站点历史占用率
- 训练只对老站点回传梯度
- 测试只在被屏蔽的新站点上统计指标

## 4. 下一步行动顺序

### Phase 1：冷启动机制植入

Task 1:

- 修改 `code/parse.py`
- 增加 `--cold_start`
- 增加 `--cold_ratio`，默认 `0.1`

Task 2:

- 修改 `code/utils.py`
- 生成长度为 275 的节点掩码
- 按 `cold_ratio` 将部分站点置为 0
- 在 batch 输出前，将被屏蔽站点的占用率历史输入清零
- 外部特征保持不变

Task 3:

- 修改 `code/train.py`
- 训练时只对老站点计算 loss
- 测试时只对新站点计算 MAE 和 MAPE

### Phase 2：原创模型骨架

Task 4:

- 在 `code/models/` 下新建 `Proposed/`
- 增加 `DynamicGraphGenerator.py`
- 跑通前向传播
- 输出张量形状满足 `[Batch, Num_Patches, Node, Node]`

