import torch.nn as nn
# 导入原封不动的官方 DLinear 代码
from models.DLinear.DLinear import Model as DLinear_Original

class Model(nn.Module):
    """
    DLinear 适配器：用于消除 Transformer 训练引擎与纯线性模型之间的参数代沟
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # 实例化原始的 DLinear
        self.model = DLinear_Original(configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 🎯 核心劫持：引擎塞进了 4 个参数，但 DLinear 只需要历史时序矩阵 (x_enc)
        # 我们在这里静默接收，并将后面 3 个对 DLinear 无用的参数（时间特征与解码器输入）丢弃
        out = self.model(x_enc)
        return out