# 项目状态

## 当前定位

- 项目主题：电动汽车充电时空预测
- 基础仓库：上游 UrbanEV 数据集与 benchmark 代码库
- 当前分支用途：在上游项目基础上扩展论文导向模型研究
- 当前主开发线：`code/`

## 已经发生的事

- 已新增或适配若干 baseline 模型
- 已对部分内部框架进行论文方向所需的改造
- 已存在 `AGCRN`、`GWNET`、`PDG2Seq` 等适配器式接入
- 根目录原始 `readme.md` 仍主要服务于公开数据集与 benchmark 项目说明

## 实际代码入口

- `code/parse.py`：全局实验参数
- `code/utils.py`：数据读取、特征拼接、DataLoader、模型注册
- `code/train.py`：训练、验证、checkpoint、评估流程
- `code/models/`：模型实现与适配器

## 当前目标线

交接文档中给出的当前目标包括：

- `DLinear`：MAPE `75.7%`，MAE `0.361`，`seq_len=96`
- `GWNET`：MAPE `13.08%`，MAE `0.173`，`seq_len=12`
- `PatchTST`：MAPE `11.0%`，MAE `0.118`，`seq_len=96`

这些数字目前应视为工作目标，后续还需要在你当前代码库中重新验证。

## 当前论文阶段

- 已经过了 baseline 对比阶段
- 当前进入创新实现阶段
- 第一工程里程碑是冷启动归纳机制
- 第二工程里程碑是原创模型骨架

## 当前缺口

- 之前没有项目级 `AGENTS.md`
- 之前没有规范化的 `docs/` 目录
- 之前没有专门的 `experiments/experiment_log.md`
- Gemini 交接文档之前不在标准项目文档位置

