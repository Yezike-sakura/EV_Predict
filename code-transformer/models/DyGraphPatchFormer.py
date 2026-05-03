import os
import sys
import importlib.util
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.PatchTST_backbone import PatchTST_backbone
from models.PatchDynamicGCN import PatchDynamicGCN


def _load_stage1_graph_generator():
    current_file = Path(__file__).resolve()
    repo_root = current_file.parents[2]
    target_file = repo_root / "code" / "models" / "DyGraphPatchFormer" / "residual_hybrid_graph.py"
    if not target_file.exists():
        raise FileNotFoundError(f"Stage-1 graph generator file not found: {target_file}")

    module_name = "stage1_residual_hybrid_graph"
    spec = importlib.util.spec_from_file_location(module_name, str(target_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from: {target_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "ResidualHybridGraphGenerator"):
        raise ImportError("ResidualHybridGraphGenerator not found in stage-1 module")
    return module.ResidualHybridGraphGenerator


ResidualHybridGraphGenerator = _load_stage1_graph_generator()


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.num_nodes = int(getattr(configs, "num_nodes", getattr(configs, "c_out", getattr(configs, "enc_in", 275))))
        self.mark_dim = int(getattr(configs, "graph_mark_dim", 4))

        if self.patch_len <= 0 or self.stride <= 0:
            raise ValueError(f"patch_len and stride must be positive, got {self.patch_len}, {self.stride}")
        if self.patch_len > self.seq_len:
            raise ValueError(f"patch_len ({self.patch_len}) cannot exceed seq_len ({self.seq_len})")

        static_adj = self._load_static_adj(getattr(configs, "static_adj_path", None), self.num_nodes)
        self.register_buffer("static_adj", static_adj)

        self.graph_gen = ResidualHybridGraphGenerator(
            num_nodes=self.num_nodes,
            mark_dim=self.mark_dim,
            hidden_dim=getattr(configs, "graph_hidden_dim", 64),
            lambda_init=getattr(configs, "lambda_init", 0.1),
            graph_norm=getattr(configs, "graph_norm", "row"),
            sym_graph=getattr(configs, "sym_graph", False),
            nonneg_mode=getattr(configs, "graph_nonneg_mode", "relu"),
        )

        self.patch_gcn = PatchDynamicGCN(
            patch_len=self.patch_len,
            dropout=getattr(configs, "dropout", 0.0),
            eps=getattr(configs, "graph_eps", 1e-6),
            use_bias=True,
            normalize_adj=False,
        )

        self.backbone = PatchTST_backbone(
            c_in=self.num_nodes,
            context_window=self.seq_len,
            target_window=self.pred_len,
            patch_len=self.patch_len,
            stride=self.stride,
            max_seq_len=getattr(configs, "seq_len", 1024),
            n_layers=getattr(configs, "e_layers", 2),
            d_model=getattr(configs, "d_model", 512),
            n_heads=getattr(configs, "n_heads", 8),
            d_k=None,
            d_v=None,
            d_ff=getattr(configs, "d_ff", 2048),
            norm="BatchNorm",
            attn_dropout=getattr(configs, "dropout", 0.0),
            dropout=getattr(configs, "dropout", 0.0),
            act=getattr(configs, "activation", "gelu"),
            key_padding_mask="auto",
            padding_var=None,
            attn_mask=None,
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pe="zeros",
            learn_pe=True,
            fc_dropout=getattr(configs, "fc_dropout", 0.0),
            head_dropout=getattr(configs, "head_dropout", 0.0),
            padding_patch=getattr(configs, "padding_patch", "end"),
            pretrain_head=False,
            head_type="flatten",
            individual=getattr(configs, "individual", False),
            revin=bool(getattr(configs, "revin", 1)),
            affine=bool(getattr(configs, "affine", 0)),
            subtract_last=bool(getattr(configs, "subtract_last", 0)),
            verbose=False,
        )

    @staticmethod
    def _load_static_adj(static_adj_path, num_nodes):
        if static_adj_path is None:
            raise ValueError("configs.static_adj_path is required for DyGraphPatchFormer")
        if not os.path.exists(static_adj_path):
            raise FileNotFoundError(f"static_adj_path not found: {static_adj_path}")

        df = pd.read_csv(static_adj_path, header=0)
        arr = df.values
        if arr.shape[0] != arr.shape[1]:
            df = pd.read_csv(static_adj_path, header=0, index_col=0)
            arr = df.values
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"static adj must be square, got {arr.shape} from {static_adj_path}")
        if arr.shape[0] != num_nodes:
            raise ValueError(f"static adj node mismatch: adj {arr.shape[0]} vs num_nodes {num_nodes}")

        adj = torch.tensor(arr, dtype=torch.float32)
        adj = torch.clamp(adj, min=0.0)
        denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj / denom

    def _prepare_mark(self, x_mark_enc, bsz, t, n, device, dtype):
        if x_mark_enc is None:
            return torch.zeros(bsz, n, t, self.mark_dim, device=device, dtype=dtype)

        if x_mark_enc.dim() == 3:
            if x_mark_enc.shape[0] != bsz or x_mark_enc.shape[1] != t:
                raise ValueError(
                    f"x_mark_enc 3D shape mismatch: got {tuple(x_mark_enc.shape)}, expected ({bsz}, {t}, Fm)"
                )
            mark = x_mark_enc.unsqueeze(1).expand(-1, n, -1, -1)
        elif x_mark_enc.dim() == 4:
            if x_mark_enc.shape[0] != bsz or x_mark_enc.shape[1] != n or x_mark_enc.shape[2] != t:
                raise ValueError(
                    f"x_mark_enc 4D shape mismatch: got {tuple(x_mark_enc.shape)}, expected ({bsz}, {n}, {t}, Fm)"
                )
            mark = x_mark_enc
        else:
            raise ValueError(f"x_mark_enc must be 3D/4D, got {x_mark_enc.dim()}D")

        fm = mark.shape[-1]
        if fm < self.mark_dim:
            pad = torch.zeros(bsz, n, t, self.mark_dim - fm, device=device, dtype=dtype)
            mark = torch.cat([mark.to(dtype), pad], dim=-1)
        elif fm > self.mark_dim:
            mark = mark[..., : self.mark_dim].to(dtype)
        else:
            mark = mark.to(dtype)
        return mark

    def _patchify(self, x_hist):
        return x_hist.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def _fold_patch_to_series(self, x_patch):
        bsz, n, patch_num, patch_len = x_patch.shape
        t_eff = (patch_num - 1) * self.stride + patch_len

        x_cols = x_patch.permute(0, 1, 3, 2).reshape(bsz * n, patch_len, patch_num)
        recon = F.fold(
            x_cols,
            output_size=(1, t_eff),
            kernel_size=(1, patch_len),
            stride=(1, self.stride),
        ).squeeze(2).squeeze(1)
        recon = recon.view(bsz, n, t_eff)

        ones_cols = torch.ones_like(x_cols)
        denom = F.fold(
            ones_cols,
            output_size=(1, t_eff),
            kernel_size=(1, patch_len),
            stride=(1, self.stride),
        ).squeeze(2).squeeze(1)
        denom = denom.view(bsz, n, t_eff).clamp_min(1e-6)
        return recon / denom

    def _align_series_length(self, x_series):
        curr_len = x_series.shape[-1]
        if curr_len == self.seq_len:
            return x_series
        if curr_len > self.seq_len:
            return x_series[..., : self.seq_len]
        return F.pad(x_series, (0, self.seq_len - curr_len))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if x_enc.dim() != 3:
            raise ValueError(f"x_enc must be [B,T,N], got {tuple(x_enc.shape)}")

        bsz, t, n = x_enc.shape
        if n != self.num_nodes:
            raise ValueError(f"node mismatch: x_enc N={n}, expected {self.num_nodes}")

        x_hist = x_enc.permute(0, 2, 1).contiguous()
        x_mark_node = self._prepare_mark(
            x_mark_enc,
            bsz=bsz,
            t=t,
            n=n,
            device=x_enc.device,
            dtype=x_enc.dtype,
        )

        a_hybrid, _ = self.graph_gen(x_hist, x_mark_node, self.static_adj)
        if not torch.isfinite(a_hybrid).all():
            raise FloatingPointError("a_hybrid contains NaN/Inf")

        x_patch = self._patchify(x_hist)
        if not torch.isfinite(x_patch).all():
            raise FloatingPointError("x_patch contains NaN/Inf before PatchDynamicGCN")

        x_patch_fused = self.patch_gcn(x_patch, a_hybrid)
        x_series = self._fold_patch_to_series(x_patch_fused)
        x_series = self._align_series_length(x_series)

        backbone_out = self.backbone(x_series)
        out = backbone_out.permute(0, 2, 1).contiguous()

        if not torch.isfinite(out).all():
            raise FloatingPointError("DyGraphPatchFormer output contains NaN/Inf")
        if out.shape[1] != self.pred_len or out.shape[2] != self.num_nodes:
            raise ValueError(
                f"output shape mismatch: got {tuple(out.shape)}, expected ({bsz}, {self.pred_len}, {self.num_nodes})"
            )
        return out


if __name__ == "__main__":
    class _Cfg:
        pred_len = 24
        seq_len = 96
        patch_len = 12
        stride = 8
        num_nodes = 275
        c_out = 275
        e_layers = 2
        d_model = 128
        n_heads = 8
        d_ff = 256
        dropout = 0.1
        activation = "gelu"
        fc_dropout = 0.0
        head_dropout = 0.0
        padding_patch = "end"
        individual = False
        revin = 1
        affine = 0
        subtract_last = 0
        graph_hidden_dim = 64
        lambda_init = 0.1
        graph_norm = "row"
        sym_graph = False
        graph_nonneg_mode = "relu"
        graph_mark_dim = 4
        static_adj_path = "../../data/adj.csv"

    cfg = _Cfg()
    model = Model(cfg)
    xb = torch.randn(2, 96, cfg.num_nodes)
    xmb = torch.randn(2, 96, 4)
    y = model(xb, xmb, None, None)
    assert y.shape == (2, 24, cfg.num_nodes), f"Unexpected shape: {y.shape}"
    print("Shape smoke test passed:", y.shape)

