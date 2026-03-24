"""
resonance_folding.algebra
=========================
Fano-plane octonion operations, all GPU-ready via PyTorch.
Every function operates on (..., 8) tensors — batch-vectorized.
"""

import torch
import torch.nn.functional as F


def oct_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batch octonion product using the Fano-plane convention.

    Args:
        A: (..., 8) tensor of octonions
        B: (..., 8) tensor of octonions

    Returns:
        (..., 8) tensor — element-wise Fano-plane product
    """
    p = A.unbind(-1)
    q = B.unbind(-1)
    return torch.stack([
        p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3]-p[4]*q[4]-p[5]*q[5]-p[6]*q[6]-p[7]*q[7],
        p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2]+p[4]*q[5]-p[5]*q[4]+p[6]*q[7]-p[7]*q[6],
        p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1]-p[4]*q[6]+p[5]*q[7]+p[6]*q[4]-p[7]*q[5],
        p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0]-p[4]*q[7]-p[5]*q[6]+p[6]*q[5]+p[7]*q[4],
        p[0]*q[4]-p[1]*q[5]+p[2]*q[6]-p[3]*q[7]+p[4]*q[0]+p[5]*q[1]-p[6]*q[2]+p[7]*q[3],
        p[0]*q[5]+p[1]*q[4]-p[2]*q[7]-p[3]*q[6]-p[4]*q[1]+p[5]*q[0]+p[6]*q[3]+p[7]*q[2],
        p[0]*q[6]-p[1]*q[7]-p[2]*q[4]+p[3]*q[5]+p[4]*q[2]-p[5]*q[3]+p[6]*q[0]+p[7]*q[1],
        p[0]*q[7]+p[1]*q[6]+p[2]*q[5]+p[3]*q[4]-p[4]*q[3]-p[5]*q[2]-p[6]*q[1]+p[7]*q[0],
    ], dim=-1)


def oct_conj(O: torch.Tensor) -> torch.Tensor:
    """
    Octonion conjugate: negate imaginary components 1–7.

    Args:
        O: (..., 8) tensor

    Returns:
        (..., 8) tensor — conjugate
    """
    c = O.clone()
    c[..., 1:] = -c[..., 1:]
    return c


def oct_normalize(O: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Project octonions onto the unit 7-sphere S⁷.

    Args:
        O:   (..., 8) tensor
        eps: numerical stability floor

    Returns:
        (..., 8) unit octonions — each row has norm 1.0
    """
    return O / (O.norm(dim=-1, keepdim=True) + eps)


def holo_loss(O: torch.Tensor) -> torch.Tensor:
    """
    Holographic coherence loss: |O * O† - I|² / 8

    Measures violation of the unit-norm property.
    For perfect unit octonions: holo_loss = 0.0 exactly.

    Args:
        O: (N, 8) tensor of (ideally unit) octonions

    Returns:
        Scalar loss — lower is better, 0.0 is perfect
    """
    oo = oct_mul(O, oct_conj(O))
    identity = torch.zeros_like(oo)
    identity[..., 0] = 1.0
    return ((oo - identity) ** 2).sum(-1).mean() / 8.0


def assoc_loss(O: torch.Tensor) -> torch.Tensor:
    """
    Associator loss: mean |(AB)C - A(BC)|²

    Measures departure from associativity — useful as a
    regularizer to keep octonion shards on the algebraic manifold.

    Args:
        O: (N, 8) tensor

    Returns:
        Scalar loss — 0.0 for purely associative triples
    """
    N = O.shape[0]
    if N < 3:
        return torch.tensor(0.0, device=O.device)
    ia = torch.randint(0, N, (N,), device=O.device)
    ib = torch.randint(0, N, (N,), device=O.device)
    ic = torch.randint(0, N, (N,), device=O.device)
    ab_c = oct_mul(oct_mul(O[ia], O[ib]), O[ic])
    a_bc = oct_mul(O[ia], oct_mul(O[ib], O[ic]))
    return ((ab_c - a_bc) ** 2).sum(-1).mean()


def oct_slerp(A: torch.Tensor, B: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation between unit octonions on S⁷.

    Follows the great-circle arc — the geometrically correct path.
    Every intermediate point is a valid unit octonion.
    Falls back to LERP + renorm for nearly-identical vectors.

    Args:
        A: (N, 8) unit octonions — start
        B: (N, 8) unit octonions — end
        t: interpolation parameter in [0, 1]

    Returns:
        (N, 8) interpolated unit octonions
    """
    dot       = (A * B).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta     = torch.acos(dot)
    sin_theta = torch.sin(theta)
    safe      = (sin_theta.abs() > 1e-6).float()

    ca = torch.where(
        safe.bool(),
        torch.sin((1.0 - t) * theta) / (sin_theta + 1e-12),
        torch.full_like(sin_theta, 1.0 - t),
    )
    cb = torch.where(
        safe.bool(),
        torch.sin(t * theta) / (sin_theta + 1e-12),
        torch.full_like(sin_theta, t),
    )
    return oct_normalize(ca * A + cb * B)


def task_vector(base: torch.Tensor, finetuned: torch.Tensor) -> torch.Tensor:
    """
    Compute the tangent-space task vector from base to finetuned.
    Implements the logarithmic map log_base(finetuned) on S⁷.

    Args:
        base:      (N, 8) unit octonions — base model shards
        finetuned: (N, 8) unit octonions — fine-tuned model shards

    Returns:
        (N, 8) tangent vectors — magnitude encodes geodesic distance
    """
    dot   = (base * finetuned).sum(dim=-1, keepdim=True)
    perp  = finetuned - dot * base
    theta = torch.acos(dot.clamp(-1 + 1e-7, 1 - 1e-7))
    perp_norm = perp.norm(dim=-1, keepdim=True) + 1e-12
    return (perp / perp_norm) * theta


def task_vector_apply(
    base: torch.Tensor,
    tv: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Apply a task vector at a given scale (exp map on S⁷).

    scale=1.0 reaches the fine-tuned position.
    scale=0.5 is halfway.
    scale=2.0 extrapolates beyond fine-tuned.
    scale=-1.0 moves in the opposite direction (unlearn).

    Args:
        base:  (N, 8) unit octonions
        tv:    (N, 8) tangent vectors from task_vector()
        scale: float — how far to walk along the geodesic

    Returns:
        (N, 8) unit octonions
    """
    tv_dir   = tv / (tv.norm(dim=-1, keepdim=True) + 1e-12)
    tv_mag   = tv.norm(dim=-1, keepdim=True)
    theta    = tv_mag * scale
    return oct_normalize(torch.cos(theta) * base + torch.sin(theta) * tv_dir)
