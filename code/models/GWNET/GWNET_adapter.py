import torch
import torch.nn as nn
from models.GWNET.model import gwnet


class Model(nn.Module):
    """
    GWNET 适配器：负责张量维度对齐、高级 Monkey Patching 漏洞修复与参数劫持
    """

    def __init__(self, args, data_mean=None, data_std=None):
        super(Model, self).__init__()
        self.args = args
        self.in_dim = args.in_dim

        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

        self.model = gwnet(
            device=device,
            num_nodes=275,
            in_dim=self.in_dim,
            out_dim=args.pred_len,
            kernel_size=2,
            blocks=4,
            layers=2
        )

        # 🎯 高级劫持 (Monkey Patching)：内存级修复官方源码中的 Typo 漏洞
        # 坚守不开箱改源码原则，在内存中动态将写错的 Conv1d 升级为 Conv2d
        self._monkey_patch_gwnet()

    def _monkey_patch_gwnet(self):
        # 安全转换元组格式 (spatial维度默认值：kernel/stride/dilation填1，padding填0)
        def to_2d(val, spatial_default):
            if isinstance(val, int): return (spatial_default, val)
            if isinstance(val, tuple): return (spatial_default, val[0]) if len(val) == 1 else val
            return (spatial_default, 1)

        # 遍历模型子图，升级所有错误的 Conv1d 为 Conv2d
        for name, module in self.model.named_children():
            if isinstance(module, nn.ModuleList):
                for i, old_conv in enumerate(module):
                    if isinstance(old_conv, nn.Conv1d):
                        new_conv = nn.Conv2d(
                            in_channels=old_conv.in_channels,
                            out_channels=old_conv.out_channels,
                            kernel_size=to_2d(old_conv.kernel_size, 1),
                            stride=to_2d(old_conv.stride, 1),
                            padding=to_2d(old_conv.padding, 0),
                            dilation=to_2d(old_conv.dilation, 1),
                            groups=old_conv.groups,
                            bias=(old_conv.bias is not None)
                        )
                        # 权重无损继承与重塑
                        new_conv.weight.data = old_conv.weight.data.clone().view_as(new_conv.weight)
                        if old_conv.bias is not None:
                            new_conv.bias.data = old_conv.bias.data.clone()
                        module[i] = new_conv

    def forward(self, x, x_mark=None):
        # 维度增补
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)

            # 外部特征融合
        if x_mark is not None and self.args.add_feat == 'time':
            if len(x_mark.shape) == 3:
                x_mark = x_mark.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
            x = torch.cat([x, x_mark], dim=-1)

        # 对齐 GWNET 原生要求: [Batch, Channels, Nodes, Seq_len]
        x = x.permute(0, 3, 1, 2)

        # 前向传播
        out = self.model(x)

        # 精准切片，剥离多余维度，对齐 Loss 计算标准: [Batch, Nodes]
        out = out.squeeze(-1)
        if out.shape[1] == 1:
            out = out.squeeze(1)

        return out