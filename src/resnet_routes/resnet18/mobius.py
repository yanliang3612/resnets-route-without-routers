"""Mask enumeration and Möbius interaction coefficients on logits."""

from typing import Tuple

import torch


def enumerate_masks(L: int = 8, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a ``(2^L, L)`` float matrix containing every binary mask.

    Row ``s`` corresponds to the indicator vector of the subset whose binary
    expansion is ``s``: bit ``l`` (LSB) is 1 iff residual branch ``l`` is
    active.  This is a fixed canonical ordering used everywhere downstream.
    """
    idx = torch.arange(1 << L, device=device)
    bits = (idx.unsqueeze(1) >> torch.arange(L, device=device).unsqueeze(0)) & 1
    return bits.to(torch.float32)


def mobius_coefficients(h_masks: torch.Tensor) -> torch.Tensor:
    """Compute ``Delta_S h_x`` for every ``S subset [L]``.

    Args:
        h_masks: ``(B, 2^L, C)`` tensor of logits for every mask, indexed in the
            canonical order produced by :func:`enumerate_masks`.

    Returns:
        ``(B, 2^L, C)`` tensor of Möbius coefficients in the same ordering.

    Implementation uses the standard fast Möbius transform: ``L`` butterfly
    sweeps, each ``O(2^L)``.  ``Delta_S = sum_{T subseteq S} (-1)^{|S|-|T|} v(T)``.
    """
    B, M, C = h_masks.shape
    L = M.bit_length() - 1
    assert M == 1 << L, f"h_masks has {M} masks; expected a power of two"
    out = h_masks.clone()
    for l in range(L):
        bit = 1 << l
        # For each pair (s with bit=0, s|bit), set v(s|bit) -= v(s).
        idx = torch.arange(M, device=out.device)
        lo = idx[(idx & bit) == 0]
        hi = lo | bit
        out[:, hi, :] = out[:, hi, :] - out[:, lo, :]
    return out


def mobius_reconstruct(deltas: torch.Tensor) -> torch.Tensor:
    """Inverse Möbius: ``h_x(1_S) = sum_{T subseteq S} Delta_T h_x``.

    Returns the reconstructed logits at the full mask ``S = [L]``.
    """
    return deltas.sum(dim=1)


def full_index(L: int = 8) -> int:
    return (1 << L) - 1


def empty_index() -> int:
    return 0
