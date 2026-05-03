# UrbanEV 交接文档中文阅读版

这份文件对应 `docs/handover.md` 的中文整理版，目的是方便你阅读，不替代原始技术版。

## 项目状态

- 研究任务：城市级电动汽车充电负荷时空预测
- 目标：开发融合动态图与 Patch 时序建模的归纳式时空模型
- 当前阶段：已完成 baseline 扩展与部分框架改造，进入创新模型开发

## 当前目标线

- `DLinear`：MAPE `75.7%`，MAE `0.361`
- `GWNET`：MAPE `13.08%`，MAE `0.173`
- `PatchTST`：MAPE `11.0%`，MAE `0.118`

## 开发边界

- `code/`：后续主要开发目录
- `code-transformer/`：Transformer 阵营代码，当前不是主战场
- 不直接改第三方 baseline 原码
- 优先通过 adapter 接入和修复

## 核心创新

### 1. 残差混合动态图

- 以静态图为底座
- 用天气、POI、时间等多模态先验生成动态图残差
- 形成融合图进行空间建模

### 2. Patch 级时空深度耦合

- 长序列先切 patch
- patch 内做空间聚合
- patch 间做时间注意力

### 3. 节点级冷启动验证

- 用掩码模拟新建站点
- 输入端屏蔽其历史占用率
- 训练时只对老站点监督
- 测试时只在新站点统计指标

## 下一阶段任务

### Phase 1

- `parse.py` 新增 `cold_start` 和 `cold_ratio`
- `utils.py` 增加节点掩码和输入致盲
- `train.py` 改训练损失和测试评估口径

### Phase 2

- 新建 `code/models/Proposed/`
- 实现 `DynamicGraphGenerator.py`
- 跑通原创模型骨架前向传播

