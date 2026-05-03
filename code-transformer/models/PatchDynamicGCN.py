import torch
import torch.nn as nn


class PatchDynamicGCN(nn.Module):
    """
    Patch-level dynamic graph aggregation.

    Input:
        x: [B, N, Num_Patches, Patch_Len]
        a_hybrid: [B, N, N]
    Output:
        out: [B, N, Num_Patches, Patch_Len]
    """

    def __init__(
        self,
        patch_len: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
        use_bias: bool = True,
        normalize_adj: bool = False,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.eps = eps
        self.normalize_adj = normalize_adj
        self.use_layernorm = use_layernorm

        self.proj = nn.Linear(patch_len, patch_len, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(patch_len) if use_layernorm else nn.Identity()

    def extra_repr(self) -> str:
        return (
            f"patch_len={self.patch_len}, dropout={self.dropout.p}, eps={self.eps}, "
            f"normalize_adj={self.normalize_adj}, use_layernorm={self.use_layernorm}"
        )

    def _check_inputs(self, x: torch.Tensor, a_hybrid: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"x must be 4D [B,N,P,L], got shape {tuple(x.shape)}")
        if a_hybrid.dim() != 3:
            raise ValueError(f"a_hybrid must be 3D [B,N,N], got shape {tuple(a_hybrid.shape)}")

        bsz, n, _, l = x.shape
        if l != self.patch_len:
            raise ValueError(f"patch_len mismatch: expected {self.patch_len}, got {l}")
        if a_hybrid.shape[0] != bsz:
            raise ValueError(f"batch mismatch: x has {bsz}, a_hybrid has {a_hybrid.shape[0]}")
        if a_hybrid.shape[1] != n or a_hybrid.shape[2] != n:
            raise ValueError(
                f"node mismatch: x has N={n}, a_hybrid has shape {tuple(a_hybrid.shape)}"
            )
        if x.device != a_hybrid.device:
            raise ValueError(f"device mismatch: x on {x.device}, a_hybrid on {a_hybrid.device}")

        if not torch.isfinite(x).all():
            raise FloatingPointError("x contains NaN/Inf before PatchDynamicGCN")
        if not torch.isfinite(a_hybrid).all():
            raise FloatingPointError("a_hybrid contains NaN/Inf before PatchDynamicGCN")

    def _maybe_normalize_adj(self, a_hybrid: torch.Tensor) -> torch.Tensor:
        if not self.normalize_adj:
            return a_hybrid
        denom = a_hybrid.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return a_hybrid / denom

    def forward(self, x: torch.Tensor, a_hybrid: torch.Tensor) -> torch.Tensor:
        self._check_inputs(x, a_hybrid)
        a_use = self._maybe_normalize_adj(a_hybrid)

        # Node-wise graph aggregation without explicit loops:
        # [B,N,N] x [B,N,P,L] -> [B,N,P,L]
        x_gcn = torch.einsum("bij,bjpl->bipl", a_use, x)

        x_mapped = self.proj(x_gcn)
        # Keep micro-temporal amplitudes by default: no local LayerNorm.
        out = self.norm(x + self.dropout(x_mapped))

        if not torch.isfinite(out).all():
            raise FloatingPointError("PatchDynamicGCN output contains NaN/Inf")
        return out

