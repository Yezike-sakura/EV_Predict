# \# 🤖 AI Agent Handover Context: UrbanEV Spatio-Temporal Forecasting

# > \*\*To AI Agent (Codex/Cursor/Claude etc.)\*\*: 

# > Please read this document carefully before executing any tasks. This is the global context protocol for the `UrbanEV` project. It defines the project state, mathematical specifications of our proposed model, and strict engineering constraints.

# 

# \## 1. 项目全景与当前状态 (Project Overview \& State)

# \* \*\*任务定位\*\*：城市级电动汽车（EV）充电负荷时空预测。

# \* \*\*终极目标\*\*：开发一个融合“多模态动态图”与“时序分块（Patch）”的归纳式（Inductive）时空大模型（暂定名：`DyGraph-PatchFormer`），冲击高水平顶刊。

# \* \*\*当前基线成绩 (Baselines to Beat)\*\*：

# &#x20; \* \*\*纯时序下界 (DLinear)\*\*：MAPE `75.7%` | MAE `0.361` (Seq\_len=96)

# &#x20; \* \*\*时空图上限 (GWNET)\*\*：MAPE `13.08%` | MAE `0.173` (Seq\_len=12)

# &#x20; \* \*\*大模型上限 (PatchTST)\*\*：MAPE `11.0%` | MAE `0.118` (Seq\_len=96)

# 

# \## 2. 工程架构与开发铁律 (Architecture \& Constraints)

# \* \*\*双轨目录\*\*：`code/` 用于图网络阵营（处理 4D 张量 `\[B, C, N, T]`），`code-transformer/` 用于时序阵营（处理 3D 张量 `\[B, L, N]`）。\*\*我们接下来的所有开发将在 `code/` 目录下进行。\*\*

# \* \*\*🚫 绝对约束 (Open-Closed Principle)\*\*：

# &#x20; \* \*\*严禁修改\*\*任何第三方 Baseline 模型（如 GWNET, AGCRN）的原始代码文件。

# &#x20; \* 所有的维度对齐、Bug修复必须通过在 `models/` 下新建 `xxx\_adapter.py` 使用张量变换和 Monkey Patching 完成。

# \* \*\*参数总线\*\*：所有的全局参数配置在 `parse.py`，模型装配与数据分发在 `utils.py`，主循环在 `train.py`。

# 

# \---

# 

# \## 3. 核心创新点与技术规约 (Core Innovations \& Math Specifications)

# > \*\*AI Instruction\*\*: When implementing the `Proposed Model`, strictly follow the tensor shapes and mathematical formulas defined below.

# 

# \### 🌟 创新点一：多模态先验驱动的“残差混合动态图” (Residual Hybrid Graph)

# \* \*\*目的\*\*：解决纯动态图震荡问题，实现节点级冷启动预测（Inductive Learning）。

# \* \*\*数学规约\*\*：

# &#x20; \* 设静态距离矩阵为 $A\_{static} \\in \\mathbb{R}^{N \\times N}$。

# &#x20; \* 设多模态协变量（天气、POI、时间戳）为 $\\mathcal{X}\_{meta} \\in \\mathbb{R}^{B \\times N \\times T \\times C\_{meta}}$。

# &#x20; \* \*\*动态残差生成\*\*：利用多层感知机（MLP）将降维后的先验特征映射为动态权值：$\\Delta A\_{dynamic} = \\text{MLP}(\\mathcal{X}\_{meta}) \\in \\mathbb{R}^{B \\times P \\times N \\times N}$ （其中 $P$ 为切分后的 Patch 数量）。

# &#x20; \* \*\*图融合\*\*：$A\_{hybrid}^{(p)} = A\_{static} + \\lambda \\cdot \\Delta A\_{dynamic}^{(p)}$。

# \* \*\*张量实现要求\*\*：生成器不能使用 $O(N^2)$ 的全连接重构，需通过计算节点嵌入的点积（Dot-product Attention）来轻量化生成 $\\Delta A$。

# 

# \### 🌟 创新点二：基于 Patch 维度的时空交替深度耦合 (ST-Cross Integration)

# \* \*\*目的\*\*：消除 Transformer 的空间盲区，同时防止传统 GCN 在逐时间帧（Frame-by-frame）计算时导致参数/显存爆炸。

# \* \*\*张量流转规约 (Tensor Dataflow)\*\*：

# &#x20; 1. \*\*输入张量\*\*：$X\_{in} \\in \\mathbb{R}^{B \\times C \\times N \\times L}$ （$L$ 为长序列，如 96）。

# &#x20; 2. \*\*时序分块 (Patching)\*\*：将序列 $L$ 切分为 $P$ 个长度为 $P\_{len}$ 的块。张量重塑为 $X\_{patch} \\in \\mathbb{R}^{B \\times C \\times N \\times P \\times P\_{len}}$。

# &#x20; 3. \*\*局部空间聚合 (Intra-Patch Spatial Fusion)\*\*：在宏观的 $P$ 维度上，利用生成的 $A\_{hybrid}^{(p)}$ 对 $N$ 维度执行图卷积。

# &#x20;    \* 复杂度从 $O(L \\cdot N^2)$ 骤降至 $O(P \\cdot N^2)$。

# &#x20; 4. \*\*全局时序注意力 (Inter-Patch Temporal Attention)\*\*：展平空间维度后，在 $P$ 维度上执行 Multi-Head Self-Attention，提取长程周期性。

# 

# \### 🌟 创新点三：节点级冷启动实验范式 (Hold-out Masking for Inductive Inference)

# \* \*\*目的\*\*：从工程上验证创新点一对新建站点的预测能力。

# \* \*\*逻辑规约\*\*：需在 `train.py` 和 `utils.py` 中引入掩码机制（Mask $\\mathbf{M} \\in \\{0, 1\\}^N$）。其中 $1$ 代表老站点，$0$ 代表新建（被屏蔽）站点。

# 

# \---

# 

# \## 4. 下一步行动蓝图 (Actionable Next Steps)

# > \*\*AI Instruction\*\*: Begin execution from Task 1. Do not proceed to the next task until the current one is fully implemented and confirmed by the user.

# 

# \### 🔴 Phase 1: 冷启动（Inductive）验证机制植入

# \* \*\*Task 1 (Config)\*\*: 修改 `code/parse.py`，增加全局布尔参数 `--cold\_start`，以及浮点参数 `--cold\_ratio`（默认 0.1）。

# \* \*\*Task 2 (Dataloader)\*\*: 修改 `code/utils.py`。当检测到 `--cold\_start` 时：

# &#x20; \* 生成长度为 275 的布尔掩码 `Mask`，按 `--cold\_ratio` 将部分站点随机或特定置 0。

# &#x20; \* \*\*致盲操作\*\*：在输出 batch 前，强行将掩码为 0 的站点的 `Occupancy` 历史输入特征（`x`）全部清零（保留它们的外部特征 `x\_mark` 不变）。

# \* \*\*Task 3 (Train/Loss)\*\*: 修改 `code/train.py`。

# &#x20; \* \*\*训练期惩罚\*\*：计算 Loss 时，通过 `loss = (loss\_func(predict, label) \* mask).mean()` 使得梯度仅沿老站点回传。

# &#x20; \* \*\*测试期核算\*\*：测试集计算 MAE/MAPE 时，反转掩码 `inverse\_mask = 1 - mask`，\*\*只统计新建站点\*\*的预测精度。

# 

# \### 🔵 Phase 2: 原创模型骨架搭建

# \* \*\*Task 4 (Model Init)\*\*: 在 `code/models/` 下新建 `Proposed/` 文件夹。创建 `DynamicGraphGenerator.py` 模块，完成创新点一所述的残差图张量算子，确保前向传播能够跑通并输出形如 `\[Batch, Num\_Patches, Node, Node]` 的图矩阵。

