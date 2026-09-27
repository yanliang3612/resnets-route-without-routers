"""Naive additive reconstructions and per-image metrics."""

from typing import Dict

import torch


EPS = 1e-12


def reconstruction_raw(h_masks: torch.Tensor) -> torch.Tensor:
    """``A_x^raw = sum_{S subseteq [L]} h_x(1_S)``.   Shape ``(B, C)``."""
    return h_masks.sum(dim=1)


def reconstruction_mean(h_masks: torch.Tensor) -> torch.Tensor:
    """``A_x^mean = (1 / 2^L) sum_S h_x(1_S)``."""
    return h_masks.mean(dim=1)


def reconstruction_centered(h_masks: torch.Tensor) -> torch.Tensor:
    """``A_x^ctr = h_x(0) + sum_{S != 0} (h_x(1_S) - h_x(0))``."""
    h_empty = h_masks[:, 0, :]
    deviations = h_masks[:, 1:, :] - h_empty.unsqueeze(1)
    return h_empty + deviations.sum(dim=1)


def reconstruction_mobius(deltas: torch.Tensor) -> torch.Tensor:
    """``sum_S Delta_S h_x``; equals the full output up to numerical precision."""
    return deltas.sum(dim=1)


def _norm(v: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(v, dim=-1)


def per_image_metrics(
    A: torch.Tensor, h_full: torch.Tensor, top1_full: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute per-image (B,) tensors for every metric in C.4.

    Includes the optimally rescaled reconstruction error
    ``alpha_x^* = <h_full, A> / (||A||^2 + eps)``.
    """
    norm_A = _norm(A)
    norm_full = _norm(h_full)
    diff = h_full - A
    inner = (h_full * A).sum(dim=-1)

    norm_ratio = norm_A / (norm_full + EPS)
    rel_err = _norm(diff) / (norm_full + EPS)
    alpha_star = inner / (norm_A.pow(2) + EPS)
    rel_err_scaled = _norm(h_full - alpha_star.unsqueeze(-1) * A) / (norm_full + EPS)
    cosine = inner / (norm_A * norm_full + EPS)
    top1_A = A.argmax(dim=-1)
    top1_agree = (top1_A == top1_full).float()

    return {
        "norm_ratio": norm_ratio,
        "rel_err": rel_err,
        "rel_err_scaled": rel_err_scaled,
        "cosine": cosine,
        "top1_agree": top1_agree,
        "alpha_star": alpha_star,
    }


def mobius_reconstruction_error(
    h_full: torch.Tensor, h_full_recon: torch.Tensor
) -> torch.Tensor:
    """Relative error between the full output and its Möbius reconstruction."""
    return _norm(h_full - h_full_recon) / (_norm(h_full) + EPS)
