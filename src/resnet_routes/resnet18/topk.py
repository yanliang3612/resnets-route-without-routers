"""Top-$K$ sparse residual-expert reconstruction (Experiment 3).

Given the (B, 2^L, C) Möbius coefficients ``Delta_S h_x`` produced by
``resnet_routes.resnet18.mobius`` and the scalar coefficients
``Delta_S v_x = Delta_S h_{x, hat_y(x)}`` produced by
``resnet_routes.resnet18.spectrum``, this module:

* ranks the 2^L - 1 = 255 *non-empty* residual subsets per image by
  magnitude (scalar |Delta_S v_x| or vector ||Delta_S h_x||_2);
* computes top-K cumulative reconstructions for every K = 1..|R| using
  cumulative sums along the sorted axis, so a full K-curve costs one
  pass over (B, 255, C);
* implements three plan-mandated baselines: random-K (with multiple
  seeds), low-order truncation (S: |S| <= r), and order-matched random
  (sample n_k per order to match the top-K composition);
* reduces every metric in the plan -- captured mass Gamma_K, scalar /
  vector reconstruction error, cosine, top-1 agreement, margin and
  margin ratio, effective sparse expert counts K_tau / K_eta, and the
  order composition Pi_K(k) of the selected top-K experts.

The baseline-only reconstruction (no residual) corresponds to K=0 and
is reported alongside as a sanity floor.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Magnitudes and rankings
# --------------------------------------------------------------------------- #

def residual_indices(L: int = 8) -> torch.Tensor:
    """Mask-space indices of the 2^L - 1 non-empty subsets (0 is baseline)."""
    return torch.arange(1, 1 << L)


def scalar_magnitudes(deltas_v: torch.Tensor) -> torch.Tensor:
    """``a_S^v(x) = |Delta_S v_x|`` for S != empty.

    Args:
        deltas_v: ``(B, 2^L)`` scalar Möbius coefficients.
    Returns:
        ``(B, 2^L - 1)`` magnitudes for residual subsets in mask order.
    """
    return deltas_v[:, 1:].abs()


def vector_magnitudes(deltas_h: torch.Tensor) -> torch.Tensor:
    """``a_S^h(x) = ||Delta_S h_x||_2`` for S != empty.

    Args:
        deltas_h: ``(B, 2^L, C)`` vector Möbius coefficients.
    Returns:
        ``(B, 2^L - 1)`` logit-space norms for residual subsets.
    """
    return deltas_h[:, 1:, :].pow(2).sum(dim=-1).sqrt()


def descending_order(mags: torch.Tensor) -> torch.Tensor:
    """Argsort each row by magnitude, descending."""
    return mags.argsort(dim=-1, descending=True)


# --------------------------------------------------------------------------- #
# Cumulative reconstructions along a sorted axis
# --------------------------------------------------------------------------- #

def gather_along(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index ``(B, M, ...) `` by ``(B, M)`` along axis 1.  Pure torch.gather."""
    if t.dim() == 2:
        return torch.gather(t, dim=1, index=idx)
    if t.dim() == 3:
        # idx is (B, M); broadcast it along the trailing channel axis.
        idx_b = idx.unsqueeze(-1).expand(-1, -1, t.shape[-1])
        return torch.gather(t, dim=1, index=idx_b)
    raise ValueError(f"unsupported tensor rank {t.dim()}")


def cumulative_topk_reconstruction(
    deltas_residual: torch.Tensor,    # (B, 255) or (B, 255, C)
    sort_idx: torch.Tensor,           # (B, 255) -- index INTO deltas_residual
    baseline: torch.Tensor,           # (B,) or (B, C)
) -> torch.Tensor:
    """Returns ``recon[..., K-1] = baseline + sum_{rank<=K} delta_residual[rank]``.

    Output shape:
        scalar: ``(B, 255)`` -- reconstructions for K = 1..255
        vector: ``(B, 255, C)``

    The K=0 (baseline-only) case is omitted; callers should report it as
    a separate reference point if needed.
    """
    sorted_deltas = gather_along(deltas_residual, sort_idx)
    cum = sorted_deltas.cumsum(dim=1)
    if baseline.dim() == 1:
        return cum + baseline.unsqueeze(-1)
    return cum + baseline.unsqueeze(1)


# --------------------------------------------------------------------------- #
# Per-image metrics on the K-curve
# --------------------------------------------------------------------------- #

def captured_mass(mags: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    """``Gamma_K(x) = sum_{S in A_K} a_S^2 / sum_{S in R} a_S^2``.

    Returns ``(B, 255)`` with entry ``K-1`` = ``Gamma_K``.
    """
    sorted_sq = gather_along(mags.pow(2), sort_idx)
    total = sorted_sq.sum(dim=-1, keepdim=True)
    return sorted_sq.cumsum(dim=-1) / (total + EPS)


def scalar_metrics(
    recon_scalar: torch.Tensor,    # (B, 255) reconstructions of v_x
    v_full: torch.Tensor,          # (B,)
) -> Dict[str, torch.Tensor]:
    err = (v_full.unsqueeze(-1) - recon_scalar).abs() / (v_full.abs().unsqueeze(-1) + EPS)
    return {"err_v": err}           # (B, 255)


def vector_metrics(
    recon_vec: torch.Tensor,       # (B, 255, C)
    h_full: torch.Tensor,          # (B, C)
    top1_full: torch.Tensor,       # (B,)
) -> Dict[str, torch.Tensor]:
    diff = h_full.unsqueeze(1) - recon_vec
    err = torch.linalg.vector_norm(diff, dim=-1) / (
        torch.linalg.vector_norm(h_full, dim=-1).unsqueeze(-1) + EPS
    )
    inner = (recon_vec * h_full.unsqueeze(1)).sum(dim=-1)
    cos = inner / (
        torch.linalg.vector_norm(recon_vec, dim=-1)
        * torch.linalg.vector_norm(h_full, dim=-1).unsqueeze(-1)
        + EPS
    )
    pred_K = recon_vec.argmax(dim=-1)
    agree = (pred_K == top1_full.unsqueeze(-1)).float()

    B, K, C = recon_vec.shape
    y = top1_full.view(B, 1, 1).expand(B, K, 1)
    h_y = torch.gather(recon_vec, dim=-1, index=y).squeeze(-1)         # (B, K)
    masked = recon_vec.scatter(-1, y, float("-inf"))
    h_other = masked.max(dim=-1).values                                 # (B, K)
    margin = h_y - h_other

    full_y = torch.gather(h_full, dim=-1, index=top1_full.unsqueeze(-1)).squeeze(-1)
    full_masked = h_full.scatter(-1, top1_full.unsqueeze(-1), float("-inf"))
    full_other = full_masked.max(dim=-1).values
    full_margin = full_y - full_other
    margin_ratio = margin / (full_margin.unsqueeze(-1) + EPS)

    return {
        "err_h": err, "cos": cos, "agree": agree,
        "margin": margin, "margin_ratio": margin_ratio,
    }


# --------------------------------------------------------------------------- #
# Effective sparse expert counts and order composition
# --------------------------------------------------------------------------- #

def k_effective(curve: torch.Tensor, threshold: float, mode: str) -> torch.Tensor:
    """Smallest K such that the curve crosses the threshold.

    Args:
        curve: ``(B, 255)`` per-image curve indexed by K-1.
        threshold: numeric threshold.
        mode: ``"err"`` (curve <= threshold) or ``"mass"`` (curve >= threshold).

    Returns:
        ``(B,)`` integer K (1..255).  255 if never satisfied.
    """
    if mode == "err":
        sat = curve <= threshold
    elif mode == "mass":
        sat = curve >= threshold
    else:
        raise ValueError(mode)
    sat_any = sat.any(dim=-1)
    first = sat.float().argmax(dim=-1)
    Kmax = curve.shape[-1]
    return torch.where(sat_any, first + 1, torch.full_like(first, Kmax))


def order_composition_curve(
    sort_idx: torch.Tensor,    # (B, 255) ranks over residual indices 0..254
    orders_residual: torch.Tensor,    # (255,) popcounts, looked up by residual index
    L: int,
) -> torch.Tensor:
    """``Pi_K(k) = (1/K) #{S in A_K : |S| = k}`` for k=0..L, K=1..255.

    Note: order 0 column will be all zeros since residual indices skip
    the empty subset.
    """
    sorted_orders = orders_residual.to(sort_idx.device)[sort_idx]    # (B, 255)
    B, M = sorted_orders.shape
    # one_hot is (B, M, L+1); cumulative sum gives counts per order at each K.
    one_hot = torch.zeros(B, M, L + 1, device=sort_idx.device)
    one_hot.scatter_(2, sorted_orders.long().unsqueeze(-1), 1.0)
    counts = one_hot.cumsum(dim=1)
    Ks = torch.arange(1, M + 1, device=sort_idx.device, dtype=counts.dtype)
    return counts / Ks.view(1, M, 1)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

def random_permutation(B: int, M: int, generator: torch.Generator) -> torch.Tensor:
    """Per-row independent random permutation of [0, M).  Shape (B, M)."""
    rand = torch.rand(B, M, generator=generator)
    return rand.argsort(dim=-1)


def low_order_indices_per_r(
    orders_residual: torch.Tensor, L: int,
) -> Dict[int, torch.Tensor]:
    """For each r=1..L, return the residual indices with order <= r.

    Returns a dict mapping r -> 1D long tensor of residual-mask positions
    (i.e. indices into the (B, 255) tensors used elsewhere).
    """
    out: Dict[int, torch.Tensor] = {}
    for r in range(1, L + 1):
        out[r] = (orders_residual <= r).nonzero(as_tuple=True)[0].long()
    return out


def order_matched_indices(
    sort_idx: torch.Tensor,           # (B, 255) magnitude ranking (vector)
    orders_residual: torch.Tensor,    # (255,)
    K_grid: Tuple[int, ...],
    generator: torch.Generator,
    L: int,
) -> Dict[int, torch.Tensor]:
    """For each K in K_grid, sample residual indices order-matched to top-K.

    Returns a dict mapping K -> (B, K) long tensor of residual indices.
    """
    B = sort_idx.shape[0]
    sorted_orders = orders_residual.to(sort_idx.device)[sort_idx]   # (B, 255)
    by_order = {
        k: (orders_residual == k).nonzero(as_tuple=True)[0].long()
        for k in range(1, L + 1)
    }
    out: Dict[int, torch.Tensor] = {}
    for K in K_grid:
        composition = torch.zeros(B, L + 1, dtype=torch.long)
        for k in range(1, L + 1):
            composition[:, k] = (sorted_orders[:, :K] == k).sum(dim=-1)
        sampled = torch.empty(B, K, dtype=torch.long)
        for b in range(B):
            chunks = []
            for k in range(1, L + 1):
                n_k = int(composition[b, k].item())
                if n_k == 0:
                    continue
                pool = by_order[k]
                if pool.numel() < n_k:
                    pick = pool
                else:
                    perm = torch.randperm(pool.numel(), generator=generator)
                    pick = pool[perm[:n_k]]
                chunks.append(pick)
            packed = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)
            sampled[b, : packed.numel()] = packed
        out[K] = sampled
    return out
