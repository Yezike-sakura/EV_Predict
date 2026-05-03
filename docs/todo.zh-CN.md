# 待办清单

## Phase 0：上下文稳定化

- [x] 创建项目级 `AGENTS.md`
- [x] 将 AI 交接文档纳入 `docs/handover.md`
- [x] 创建 `docs/project_status.md`
- [x] 创建 `docs/innovation_spec.md`
- [x] 创建 `experiments/experiment_log.md`

## Phase 1：冷启动归纳机制

- [ ] 在 `code/parse.py` 中增加 `--cold_start`
- [ ] 在 `code/parse.py` 中增加 `--cold_ratio`
- [ ] 在 `code/utils.py` 中生成节点 mask
- [ ] 在 `code/utils.py` 中对被屏蔽节点历史占用率输入做致盲
- [ ] 外部特征保持不变
- [ ] 在 `code/train.py` 中使训练 loss 仅作用于老站点
- [ ] 在 `code/train.py` 中使测试指标仅统计新站点
- [ ] 增加最小验证命令和预期行为说明

## Phase 2：原创模型骨架

- [ ] 创建 `code/models/Proposed/`
- [ ] 增加 `DynamicGraphGenerator.py`
- [ ] 定义输入输出张量契约
- [ ] 将模型注册进现有加载流程

## Phase 3：论文导向整理

- [ ] 重写根目录 `readme.md`，让论文项目说明排在上游数据集说明之前
- [ ] 增加你这条论文线可复现实验命令
- [ ] 在当前代码库中重新验证 baseline 结果

