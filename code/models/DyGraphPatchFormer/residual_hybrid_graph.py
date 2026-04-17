import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualHybridGraphGenerator(nn.Module):
    """
    Build batch-wise hybrid adjacency:
    A_hybrid = A_static + lambda * DeltaA_dynamic
    """

    def __init__(
        self,
        num_nodes,
        mark_dim,
        hidden_dim=64,
        lambda_init=0.1,
        graph_norm="row",
        sym_graph=True,
        nonneg_mode="relu",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.graph_norm = graph_norm
        self.sym_graph = sym_graph
        self.nonneg_mode = nonneg_mode

        temporal_dim = max(8, hidden_dim // 2)
        mark_hidden_dim = max(8, hidden_dim // 2)

        self.temporal_encoder = nn.Sequential(
            nn.Linear(3, temporal_dim),
            nn.ReLU(),
            nn.Linear(temporal_dim, temporal_dim),
        )
        self.mark_encoder = nn.Sequential(
            nn.Linear(mark_dim, mark_hidden_dim),
            nn.ReLU(),
            nn.Linear(mark_hidden_dim, mark_hidden_dim),
        )
        self.node_fuser = nn.Sequential(
            nn.Linear(temporal_dim + mark_hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        lambda_tensor = torch.tensor([lambda_init], dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        self.raw_lambda = nn.Parameter(torch.logit(lambda_tensor))

    def _normalize(self, adj):
        if self.graph_norm == "none":
            return adj
        if self.graph_norm == "sym":
            d = adj.sum(dim=-1).clamp_min(1e-6)
            d_inv_sqrt = torch.pow(d, -0.5)
            d_left = d_inv_sqrt.unsqueeze(-1)
            d_right = d_inv_sqrt.unsqueeze(-2)
            return d_left * adj * d_right

        d = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj / d

    def _apply_nonneg(self, adj):
        if self.nonneg_mode == "softplus":
            return F.softplus(adj)
        return F.relu(adj)

    def forward(self, x_hist, x_mark, a_static):
        if x_hist.dim() != 3:
            raise ValueError(f"x_hist must be [B,N,T], got {tuple(x_hist.shape)}")
        if a_static.dim() != 2:
            raise ValueError(f"a_static must be [N,N], got {tuple(a_static.shape)}")

        bsz, n, _ = x_hist.shape
        if n != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: expected {self.num_nodes}, got {n}")
        if a_static.shape[0] != n or a_static.shape[1] != n:
            raise ValueError(f"a_static shape mismatch: {tuple(a_static.shape)} vs ({n}, {n})")

        x_last = x_hist[:, :, -1]
        x_mean = x_hist.mean(dim=-1)
        x_max = x_hist.max(dim=-1).values
        temporal_stats = torch.stack([x_last, x_mean, x_max], dim=-1)
        h_t = self.temporal_encoder(temporal_stats)

        if x_mark is None:
            x_mark_summary = torch.zeros(
                bsz, n, self.mark_encoder[0].in_features, device=x_hist.device, dtype=x_hist.dtype
            )
        else:
            if x_mark.dim() != 4:
                raise ValueError(f"x_mark must be [B,N,T,Fm], got {tuple(x_mark.shape)}")
            if x_mark.shape[0] != bsz or x_mark.shape[1] != n:
                raise ValueError(f"x_mark first dims mismatch: got {tuple(x_mark.shape[:2])}, need ({bsz}, {n})")
            x_mark_summary = x_mark.mean(dim=2)

        h_m = self.mark_encoder(x_mark_summary)
        h = self.node_fuser(torch.cat([h_t, h_m], dim=-1))

        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        pair_feat = torch.cat([h_i, h_j, torch.abs(h_i - h_j)], dim=-1)
        delta = torch.tanh(self.pair_scorer(pair_feat).squeeze(-1))

        eye = torch.eye(n, device=x_hist.device, dtype=x_hist.dtype).unsqueeze(0)
        delta = delta * (1.0 - eye)
        if self.sym_graph:
            delta = 0.5 * (delta + delta.transpose(1, 2))

        lam = torch.sigmoid(self.raw_lambda)
        a_hybrid = a_static.unsqueeze(0) + lam * delta
        a_hybrid = self._apply_nonneg(a_hybrid)
        a_hybrid = self._normalize(a_hybrid)

        if not torch.isfinite(a_hybrid).all():
            raise FloatingPointError("A_hybrid contains NaN/Inf")

        return a_hybrid, lam

