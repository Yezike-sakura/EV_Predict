import torch
import torch.nn as nn

from .residual_hybrid_graph import ResidualHybridGraphGenerator


class DyGraphPatchFormerAdapter(nn.Module):
    """
    Stage-1 adapter:
    - Generate A_hybrid [B, N, N]
    - Run a tiny prediction head for end-to-end trainability
    """

    def __init__(self, args, static_adj):
        super().__init__()
        self.args = args
        self.num_nodes = args.num_nodes
        self.mark_dim = self._infer_mark_dim(args)

        static_adj = static_adj.float()
        static_adj = self._normalize_static_graph(static_adj)
        self.register_buffer("a_static", static_adj)

        self.graph_generator = ResidualHybridGraphGenerator(
            num_nodes=self.num_nodes,
            mark_dim=self.mark_dim,
            hidden_dim=args.graph_hidden_dim,
            lambda_init=args.lambda_init,
            graph_norm=args.graph_norm,
            sym_graph=args.sym_graph,
            nonneg_mode=args.graph_nonneg_mode,
        )

        self.pred_head = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.latest_graph_stats = {}

    @staticmethod
    def _infer_mark_dim(args):
        if args.add_feat == "None":
            return 1
        dim = 0
        for feat_name in args.add_feat.split("+"):
            dim += 4 if feat_name == "time" else 1
        return max(dim, 1)

    @staticmethod
    def _normalize_static_graph(adj):
        adj = torch.clamp(adj, min=0.0)
        if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"static adjacency must be square [N,N], got {tuple(adj.shape)}")
        row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return adj / row_sum

    def _pack_mark(self, occ, extra_feat):
        bsz, n, seq_len = occ.shape
        if isinstance(extra_feat, torch.Tensor):
            if extra_feat.dim() != 4:
                raise ValueError(f"extra_feat must be [B,N,T,F], got {tuple(extra_feat.shape)}")
            if extra_feat.shape[0] != bsz or extra_feat.shape[1] != n or extra_feat.shape[2] != seq_len:
                raise ValueError(
                    f"extra_feat shape mismatch, got {tuple(extra_feat.shape)}, expected ({bsz}, {n}, {seq_len}, F)"
                )
            if extra_feat.shape[3] != self.mark_dim:
                raise ValueError(
                    f"extra_feat feature dim mismatch, got {extra_feat.shape[3]}, expected {self.mark_dim}"
                )
            return extra_feat

        return torch.zeros(bsz, n, seq_len, self.mark_dim, device=occ.device, dtype=occ.dtype)

    def forward(self, occ, extra_feat):
        if occ.dim() != 3:
            raise ValueError(f"occupancy input must be [B,N,T], got {tuple(occ.shape)}")
        if occ.shape[1] != self.num_nodes:
            raise ValueError(f"node dim mismatch: got {occ.shape[1]}, expected {self.num_nodes}")

        x_mark = self._pack_mark(occ, extra_feat)
        a_hybrid, lam = self.graph_generator(occ, x_mark, self.a_static)
        assert a_hybrid.shape == (occ.shape[0], self.num_nodes, self.num_nodes)

        x_last = occ[:, :, -1]
        x_agg = torch.bmm(a_hybrid, x_last.unsqueeze(-1)).squeeze(-1)
        node_feat = torch.stack([x_last, x_agg], dim=-1)
        pred = self.pred_head(node_feat).squeeze(-1)

        self.latest_graph_stats = {
            "lambda": float(lam.detach().item()),
            "adj_mean": float(a_hybrid.detach().mean().item()),
            "adj_std": float(a_hybrid.detach().std().item()),
            "adj_zero_ratio": float((a_hybrid.detach() == 0).float().mean().item()),
        }

        if self.num_nodes == 1:
            pred = pred.unsqueeze(-1)
        return pred

