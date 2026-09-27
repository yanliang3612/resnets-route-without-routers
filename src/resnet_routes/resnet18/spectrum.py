"""Möbius interaction order spectrum -- scalar and vector readouts.

Given the (B, 2^L, C) logits ``h_x(1_S)`` for every binary mask of the L=8
residual gates of a ResNet-18, this module computes:

* scalar Möbius coefficients over the full-model predicted-class logit
  ``v_x(m) = h_{x, hat_y(x)}(m)``;
* vector Möbius coefficients over the whole logit vector;
* per-order energy spectra E_k^v, E_k^h (k=0..L);
* residual-only normalized spectra E_tilde (k=1..L);
* baseline-inclusive normalized spectra E_bar (k=0..L);
* average per-interaction energy M_k = E_k / binom(L, k);
* cumulative low-order energy C_<=K, effective order kappa, tail T_>K
  (residual-only, k=1..L).
"""

from __future__ import annotations

import math
from typing import Dict

import torch

EPS = 1e-12


def order_indices(L: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return ``(2^L,)`` int tensor whose entry s is the popcount ``|S|``.

    Uses the same canonical subset ordering as
    :func:`resnet_routes.resnet18.mobius.enumerate_masks` (bit l = 1 iff branch l is active).
    """
    idx = torch.arange(1 << L, device=device)
    counts = torch.zeros_like(idx)
    for l in range(L):
        counts = counts + ((idx >> l) & 1)
    return counts


def scalar_mobius_predicted(
    h_masks: torch.Tensor, top1_full: torch.Tensor
) -> torch.Tensor:
    """Scalar Möbius coefficients for the predicted-class logit.

    Args:
        h_masks: ``(B, 2^L, C)`` logits for every mask.
        top1_full: ``(B,)`` integer class index from the full mask.

    Returns:
        ``(B, 2^L)`` tensor of ``Delta_S v_x`` with
        ``v_x(m) = h_{x, hat_y(x)}(m)``.
    """
    B, M, _ = h_masks.shape
    L = M.bit_length() - 1
    assert M == 1 << L
    idx = top1_full.view(B, 1, 1).expand(B, M, 1)
    v_masks = torch.gather(h_masks, dim=2, index=idx).squeeze(-1)  # (B, M)
    return _fast_mobius_1d(v_masks, L)


def _fast_mobius_1d(v: torch.Tensor, L: int) -> torch.Tensor:
    """In-place fast Möbius transform along the last axis (size 2^L)."""
    out = v.clone()
    M = 1 << L
    idx = torch.arange(M, device=out.device)
    for l in range(L):
        bit = 1 << l
        lo = idx[(idx & bit) == 0]
        hi = lo | bit
        out[..., hi] = out[..., hi] - out[..., lo]
    return out


def order_energies_scalar(deltas_v: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
    """``E_k^v(x) = sum_{|S|=k} (Delta_S v_x)^2``.

    Args:
        deltas_v: ``(B, 2^L)`` scalar Möbius coefficients.
        orders: ``(2^L,)`` popcount of each subset (output of ``order_indices``).

    Returns:
        ``(B, L+1)`` per-image order energies (k = 0..L).
    """
    B, M = deltas_v.shape
    L = M.bit_length() - 1
    sq = deltas_v.pow(2)
    out = torch.zeros(B, L + 1, dtype=sq.dtype, device=sq.device)
    out.scatter_add_(1, orders.view(1, M).expand(B, M).to(torch.long), sq)
    return out


def order_energies_vector(deltas_h: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
    """``E_k^h(x) = sum_{|S|=k} ||Delta_S h_x||_2^2``.

    Args:
        deltas_h: ``(B, 2^L, C)`` vector-valued Möbius coefficients.
        orders: ``(2^L,)`` popcount of each subset.

    Returns:
        ``(B, L+1)`` per-image order energies.
    """
    B, M, C = deltas_h.shape
    L = M.bit_length() - 1
    sq = deltas_h.pow(2).sum(dim=2)  # (B, M) sum over channels
    out = torch.zeros(B, L + 1, dtype=sq.dtype, device=sq.device)
    out.scatter_add_(1, orders.view(1, M).expand(B, M).to(torch.long), sq)
    return out


def derived_metrics(E: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Convert per-image order energies ``(B, L+1)`` into the plan's metrics.

    Returns a dict of per-image tensors:
      * ``E_tilde``  ``(B, L)``  residual-only normalized spectrum, k=1..L
      * ``E_bar``    ``(B, L+1)`` baseline-inclusive spectrum,    k=0..L
      * ``M``        ``(B, L+1)`` E_k / binom(L, k),              k=0..L
      * ``cum``      ``(B, L)``   C_{<=K} (residual-only),        K=1..L
      * ``tail``     ``(B, L)``   T_{>K} = 1 - C_{<=K},           K=1..L
      * ``kappa``    ``(B,)``     effective interaction order
    """
    B, Lp1 = E.shape
    L = Lp1 - 1

    sum_resid = E[:, 1:].sum(dim=1, keepdim=True) + EPS  # (B, 1)
    sum_all = E.sum(dim=1, keepdim=True) + EPS

    E_tilde = E[:, 1:] / sum_resid
    E_bar = E / sum_all

    binom = torch.tensor(
        [math.comb(L, k) for k in range(L + 1)],
        dtype=E.dtype, device=E.device,
    )
    M_k = E / binom.view(1, Lp1)

    # Cumulative residual-only energy: cum[:, K-1] = (E_1+..+E_K)/sum_resid
    cum = torch.cumsum(E[:, 1:], dim=1) / sum_resid
    tail = 1.0 - cum

    ks = torch.arange(1, L + 1, dtype=E.dtype, device=E.device)
    kappa = (E[:, 1:] * ks.view(1, L)).sum(dim=1) / sum_resid.squeeze(1)

    return {
        "E_tilde": E_tilde,
        "E_bar": E_bar,
        "M": M_k,
        "cum": cum,
        "tail": tail,
        "kappa": kappa,
    }
