# models/PDG2Seq/pdg2seq_adapter.py
import torch
import torch.nn as nn
from .PDG2Seq import PDG2Seq  # 导入原作者的模型


class UrbanEV_PDG2Seq(nn.Module):
    def __init__(self, args):
        super(UrbanEV_PDG2Seq, self).__init__()
        # 强制同步全局参数到模型专属参数
        args.horizon = args.pred_len
        self.pred_len = args.pred_len
        self.core_model = PDG2Seq(args)

        # =========================================================
        # 全局权重“洗礼”：显式初始化所有参数
        # 原代码大量使用 nn.Parameter(torch.FloatTensor(...)) / torch.empty(...)，
        # 仅仅申请了内存而没有真正初始化，会把上一段程序残留的垃圾数值当作权重。
        # 这里统一用 Xavier 初始化权重、将偏置置零，保证训练从“干净”的起点出发。
        # =========================================================
        for p in self.core_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, min=-3.0, max=3.0))

    def forward(self, occ, extra_feat):
        # 1. 强制维度对齐
        B, N, T = occ.shape
        x = occ.permute(0, 2, 1).unsqueeze(-1)  # -> [B, T, N, 1]

        assert extra_feat != 'None', "PDG2Seq requires '--add_feat time'"

        # 🎯 注意这里：将原本的 ef 改名为 ef_raw，表示这是包含 4 个维度的“原始食材”
        ef_raw = extra_feat.permute(0, 2, 1, 3).clone()  # -> [B, T, N, 4]

        # =======================================================
        # 🎯 厨师加工区：把通用食材转换为 PDG2Seq 的专属格式
        # PDG2Seq 需要: 0维是 Time-of-Day (0-287), 1维是 Day-of-Week (0-6)
        # =======================================================
        t_i_d = torch.div(ef_raw[..., 0], 5, rounding_mode='floor')  # 分钟数 // 5 -> 0~287
        d_i_w = ef_raw[..., 1]

        # 拼装成模型真正需要的历史时间特征 ef [B, T, N, 2]
        ef = torch.stack([t_i_d, d_i_w], dim=-1)

        # 2. 自动推演未来时间戳 (使用绝对整数进行逻辑推演)
        last_t_i_d = ef[:, -1:, :, 0]
        last_d_i_w = ef[:, -1:, :, 1]

        future_ef_list = []
        curr_t_i_d, curr_d_i_w = last_t_i_d, last_d_i_w
        for _ in range(self.pred_len):
            curr_t_i_d = (curr_t_i_d + 1) % 288
            rollover_mask = (curr_t_i_d == 0).float()
            curr_d_i_w = (curr_d_i_w + rollover_mask) % 7
            future_ef_list.append(torch.stack([curr_t_i_d, curr_d_i_w], dim=-1))

        future_ef = torch.cat(future_ef_list, dim=1)  # -> [B, pred_len, N, 2]

        # =========================================================
        # 🎯 核心修复区：迎合原作者的数据口味
        # 原模型内部有 * 288 的操作，说明它需要 [0, 1) 的浮点数输入。
        # 我们在这里把 Time-of-day (第 0 维特征) 压缩到 0~1 之间。
        # =========================================================
        ef[..., 0] = ef[..., 0] / 288.0
        future_ef[..., 0] = future_ef[..., 0] / 288.0
        # =========================================================

        # 3. 拼装成 Source 和 Target 送入模型
        source = torch.cat([x, ef], dim=-1)  # -> [B, T, N, 3]
        dummy_y = torch.zeros(B, self.pred_len, N, 1, device=occ.device)
        target = torch.cat([dummy_y, future_ef], dim=-1)

        # 4. 前向传播
        # 关闭 Teacher Forcing
        out = self.core_model(source, target, batches_seen=40000)
        out = out[:, -1, :, 0]

        return out