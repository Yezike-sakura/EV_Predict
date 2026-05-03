# UrbanEV Codex 规则

## 项目目标

这个仓库用于电动汽车充电时空预测研究。
当前阶段是围绕论文方向进行创新模型开发：

- 电动汽车充电预测
- 基于 UrbanEV 的时空预测
- 当前重点：冷启动归纳设置与原创模型骨架

## 阅读顺序

在开始修改前，建议按以下顺序阅读：

1. `AGENTS.md`
2. `docs/handover.md`
3. `docs/project_status.md`
4. `docs/innovation_spec.md`
5. `code/parse.py`
6. `code/utils.py`
7. `code/train.py`

## 工程边界

- 后续模型开发主要在 `code/` 下进行。
- 不要直接修改第三方 baseline 的原始源码，除非你明确要求。
- 优先通过 `code/models/*/*_adapter.py` 的适配器方式接入。
- `parse.py` 作为全局参数总线。
- `utils.py` 作为数据加载、特征拼接和模型装配中心。
- `train.py` 作为训练与评估主入口。

## 当前优先级

1. 先完成 cold-start 掩码机制。
2. 验证新的训练与评估逻辑。
3. Phase 1 稳定后，再进入 `code/models/Proposed/` 的原创模型骨架开发。

## 改动原则

- 改代码前先说明会改哪些文件。
- 改动保持收敛、可追踪。
- 实验记录写入 `experiments/experiment_log.md`。
- 把“真实测得的结论”和“推测”分开写。

