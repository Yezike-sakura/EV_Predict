import sys
import torch
import torch.nn as nn

# ---------------------------------------------------------
# 🧙‍♂️ 动态引用劫持魔法 (Dynamic Import Hijacking)
# ---------------------------------------------------------
# 原作者的源码中写死了: from model.AGCRNCell import ...
# 为了不修改原文件，我们将全局的 'model' 模块直接重定向到我们的 'models.AGCRN'
import models.AGCRN

sys.modules['model'] = models.AGCRN

# 劫持成功后，现在可以直接导入原封不动的官方模型了
from models.AGCRN.AGCRN import AGCRN


class AGCRN_Adapter(nn.Module):
    def __init__(self, args):
        super(AGCRN_Adapter, self).__init__()
        self.args = args

        # ---------------------------------------------------------
        # 🔌 参数桥接 (Parameter Bridging)
        # 动态给 args 挂载原模型所需的特定命名变量，不影响全局 Parse
        # ---------------------------------------------------------
        if not hasattr(args, 'input_dim'): args.input_dim = 1
        if not hasattr(args, 'output_dim'): args.output_dim = 1
        if not hasattr(args, 'horizon'): args.horizon = args.pred_len  # 桥接预测步长
        if not hasattr(args, 'default_graph'): args.default_graph = True

        # 直接实例化原汁原味的官方模型
        self.model = AGCRN(args)

        self._init_weights()
        self._register_gradient_clipping()

    def _init_weights(self):
        """防止模型在初始阶段梯度爆炸"""
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _register_gradient_clipping(self):
        """防爆盾：限制 AGCRN 反向传播梯度范围 (-3.0 to 3.0)"""
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, min=-3.0, max=3.0))

    def forward(self, occupancy, extra_feat=None):
        """
        满足 train.py 铁三角契约: predict = net(occupancy, extra_feat)
        """
        # 🚨 核心排雷: 纠正 utils.py 造成的 [Batch, Nodes, Seq_len] 维度错位
        # 我们需要把它扭转回 AGCRN 需要的 [Batch, Seq_len, Nodes, Features]

        # 1. 维度纠偏与扩充: [Batch, Nodes, Seq_len] -> [Batch, Seq_len, Nodes] -> [Batch, Seq_len, Nodes, 1]
        x = occupancy.permute(0, 2, 1).unsqueeze(-1)

        # 2. 动态融合外部特征
        if type(extra_feat) != str:  # 识别不是 'None' 字符串时
            # extra_feat 同样受 utils.py 影响，形状为 [Batch, Nodes, Seq_len, Features]
            # 纠偏为 [Batch, Seq_len, Nodes, Features]
            e_feat = extra_feat.permute(0, 2, 1, 3)
            x = torch.cat([x, e_feat], dim=-1)  # 拼接后，特征维刚好等于 n_fea

        # 3. 前向推演
        out = self.model(x, targets=None)  # 官方输出形状: [Batch, Pred_len, Nodes, 1]

        # 4. 降维回传 (train.py 契约需要 [Batch, Pred_len, Nodes])
        out = out.squeeze(-1)

        # 5. 完美适配 Broadcasting 漏洞修复机制
        if self.args.pred_len == 1:
            out = out.squeeze(1)  # -> [Batch, Nodes]

        return out