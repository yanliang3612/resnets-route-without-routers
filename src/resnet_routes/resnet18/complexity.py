"""Per-image complexity measures over residual interaction strengths.

Inputs are the 255-dim magnitude vectors $a_S^h(x)$ (vector ranking,
from :mod:`resnet_routes.resnet18.topk`) or $a_S^v(x)$ (scalar
ranking).  Outputs are the six measures from the plan:

* ``N_eff``     -- effective number of experts (Hill / participation ratio)
* ``N_ent``     -- entropy effective number ``exp(H)``
* ``kbar``      -- strength-weighted average order
* ``kappa``     -- energy-weighted effective order
* ``C_le_K``    -- low-order mass fraction (default K = 3)
* ``T_gt_K``    -- high-order tail mass = 1 - C_le_K
* ``K_tau_err``, ``K_eta_mass`` -- effective expert counts at thresholds.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

EPS = 1e-12


def participation_ratio(mags: torch.Tensor) -> torch.Tensor:
    """``N_eff = (sum a)^2 / sum a^2``.  Hill diversity at q=2."""
    s1 = mags.sum(dim=-1)
    s2 = mags.pow(2).sum(dim=-1)
    return s1.pow(2) / (s2 + EPS)


def entropy_effective_number(mags: torch.Tensor) -> torch.Tensor:
    """``exp(H)`` with ``p_S = a_S^2 / sum a^2``."""
    sq = mags.pow(2)
    p = sq / (sq.sum(dim=-1, keepdim=True) + EPS)
    H = -(p * (p.clamp_min(EPS).log())).sum(dim=-1)
    return H.exp()


def average_order_strength(mags: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
    """``kbar = sum |S| a_S / sum a_S``."""
    num = (mags * orders.unsqueeze(0)).sum(dim=-1)
    den = mags.sum(dim=-1)
    return num / (den + EPS)


def average_order_energy(mags: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
    """``kappa = sum |S| a_S^2 / sum a_S^2``."""
    sq = mags.pow(2)
    num = (sq * orders.unsqueeze(0)).sum(dim=-1)
    den = sq.sum(dim=-1)
    return num / (den + EPS)


def low_order_mass(
    mags: torch.Tensor, orders: torch.Tensor, K: int,
) -> torch.Tensor:
    """``C_le_K = sum_{|S|<=K} a_S^2 / sum a_S^2``."""
    sq = mags.pow(2)
    keep = (orders <= K).float().unsqueeze(0)
    num = (sq * keep).sum(dim=-1)
    den = sq.sum(dim=-1)
    return num / (den + EPS)


def k_eff_thresholds(
    cum: torch.Tensor, threshold: float, mode: str,
) -> torch.Tensor:
    """Smallest K such that the curve crosses the threshold.

    Args:
        cum: ``(B, R)`` cumulative curve indexed by K-1.
        threshold: numeric threshold.
        mode: ``"err"`` (curve <= threshold) or ``"mass"`` (curve >= threshold).
    Returns:
        ``(B,)`` integer K (1..R).  R if never satisfied.
    """
    if mode == "err":
        sat = cum <= threshold
    elif mode == "mass":
        sat = cum >= threshold
    else:
        raise ValueError(mode)
    sat_any = sat.any(dim=-1)
    first = sat.float().argmax(dim=-1)
    Kmax = cum.shape[-1]
    return torch.where(sat_any, first + 1, torch.full_like(first, Kmax))


def all_complexity_measures(
    mags: torch.Tensor,
    orders: torch.Tensor,
    *,
    K_low: int = 3,
) -> Dict[str, torch.Tensor]:
    """One-stop computation of the six per-image complexity measures.

    Args:
        mags: ``(B, R)`` non-negative interaction strengths.
        orders: ``(R,)`` long tensor of subset cardinality |S|.
        K_low: low-order cutoff for ``C_le_K`` (default 3).

    Returns: dict of (B,) tensors.  ``K_tau_err`` and ``K_eta_mass`` are
    NOT included here (they require the cumulative reconstruction curve;
    see ``step_1_evaluate``).
    """
    return {
        "N_eff":   participation_ratio(mags),
        "N_ent":   entropy_effective_number(mags),
        "kbar":    average_order_strength(mags, orders.float()),
        "kappa":   average_order_energy(mags, orders.float()),
        "C_le_K":  low_order_mass(mags, orders, K_low),
        "T_gt_K":  1.0 - low_order_mass(mags, orders, K_low),
    }
