"""Numerically stable, view-based subset Möbius transforms.

Mask values use the canonical integer/LSB ordering.  The mask dimension is the
middle dimension of a ``(B, 2**L, C)`` tensor.  Reshaping each butterfly stage
into ``(B, groups, 2, stride, C)`` exposes disjoint lower/upper views and avoids
the large index tensors produced by advanced indexing at ``L=16``.
"""

from __future__ import annotations

import torch


def _num_gates(values: torch.Tensor) -> int:
    if values.ndim != 3:
        raise ValueError(
            f"expected a (B, 2**L, C) tensor, got shape {tuple(values.shape)}"
        )
    mask_count = values.shape[1]
    if mask_count <= 0 or mask_count & (mask_count - 1):
        raise ValueError(
            f"mask dimension must be a positive power of two, got {mask_count}"
        )
    if values.shape[0] <= 0 or values.shape[2] <= 0:
        raise ValueError(f"batch and class dimensions must be positive: {values.shape}")
    return mask_count.bit_length() - 1


def _require_inplace_compatible(values: torch.Tensor) -> int:
    gates = _num_gates(values)
    if not values.is_contiguous():
        raise ValueError("in-place Möbius transforms require a contiguous tensor")
    if not (values.is_floating_point() or values.is_complex()):
        raise TypeError("Möbius transforms require floating-point or complex values")
    return gates


def mobius_transform_(values: torch.Tensor) -> torch.Tensor:
    """Transform mask evaluations to subset coefficients in place.

    The returned object is ``values`` itself.  No advanced-index arrays or
    mask-sized gather temporaries are allocated.
    """
    gates = _require_inplace_compatible(values)
    batch, mask_count, classes = values.shape
    for gate in range(gates):
        stride = 1 << gate
        stage = values.view(batch, mask_count // (2 * stride), 2, stride, classes)
        upper = stage[:, :, 1, :, :]
        lower = stage[:, :, 0, :, :]
        upper.sub_(lower)
    return values


def inverse_mobius_transform_(coefficients: torch.Tensor) -> torch.Tensor:
    """Apply the inverse subset transform in place and return the input object."""
    gates = _require_inplace_compatible(coefficients)
    batch, mask_count, classes = coefficients.shape
    for gate in range(gates):
        stride = 1 << gate
        stage = coefficients.view(
            batch, mask_count // (2 * stride), 2, stride, classes
        )
        upper = stage[:, :, 1, :, :]
        lower = stage[:, :, 0, :, :]
        upper.add_(lower)
    return coefficients


def mobius_coefficients(
    mask_values: torch.Tensor,
    *,
    dtype: torch.dtype | None = torch.float64,
) -> torch.Tensor:
    """Return Möbius coefficients without modifying ``mask_values``.

    FP64 is the experiment default because sixteen subtraction sweeps can
    amplify cancellation.  Pass ``dtype=None`` to preserve the input dtype.
    """
    _num_gates(mask_values)
    target_dtype = mask_values.dtype if dtype is None else dtype
    if not (target_dtype.is_floating_point or target_dtype.is_complex):
        raise TypeError("dtype must be floating-point or complex")
    out = mask_values.to(dtype=target_dtype).contiguous().clone()
    return mobius_transform_(out)


def inverse_mobius_transform(
    coefficients: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return all reconstructed mask values without modifying coefficients."""
    _num_gates(coefficients)
    target_dtype = coefficients.dtype if dtype is None else dtype
    if not (target_dtype.is_floating_point or target_dtype.is_complex):
        raise TypeError("dtype must be floating-point or complex")
    out = coefficients.to(dtype=target_dtype).contiguous().clone()
    return inverse_mobius_transform_(out)


def full_reconstruction(coefficients: torch.Tensor) -> torch.Tensor:
    """Reconstruct the all-open logits as the sum of all subset coefficients."""
    _num_gates(coefficients)
    return coefficients.sum(dim=1)


def mobius_reconstruct(coefficients: torch.Tensor) -> torch.Tensor:
    """Compatibility alias for :func:`full_reconstruction`."""
    return full_reconstruction(coefficients)


def full_index(num_gates: int = 16) -> int:
    """Return the integer index of the all-open mask."""
    if num_gates < 0:
        raise ValueError(f"num_gates must be nonnegative, got {num_gates}")
    return (1 << num_gates) - 1


def empty_index() -> int:
    """Return the integer index of the all-closed mask."""
    return 0
