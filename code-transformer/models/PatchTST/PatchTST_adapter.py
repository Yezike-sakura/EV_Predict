import torch.nn as nn
# 导入同目录下的官方 PatchTST 核心模型
from models.PatchTST.PatchTST import Model as PatchTST_Original

class Model(nn.Module):
    """
    PatchTST 适配器：用于消除 Transformer 训练引擎与纯时序大模型输入接口的参数代沟
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # 实例化原始的 PatchTST
        self.model = PatchTST_Original(configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 🎯 核心劫持：引擎传入 4 个参数，但 PatchTST 只需要历史序列矩阵 (x_enc)
        # 丢弃无用的时间特征与解码器输入，保持原模型代码纯洁性
        out = self.model(x_enc)
        return out