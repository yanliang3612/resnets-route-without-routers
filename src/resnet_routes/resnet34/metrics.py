"""Dimension-independent metrics shared by the missing ResNet-34 experiments."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from resnet_routes.resnet18.spectrum import derived_metrics
from resnet_routes.resnet18.complexity import all_complexity_measures
from resnet_routes.resnet18.difficulty import all_difficulty_scores

from .common import EPS


RECONSTRUCTION_NAMES = ("raw", "mean", "ctr", "mob")


def interaction_orders(num_gates: int, device: torch.device | str) -> torch.Tensor:
    ids = torch.arange(1 << num_gates, dtype=torch.int64, device=device)
    orders = torch.zeros_like(ids)
    for gate in range(num_gates):
        orders.add_((ids >> gate) & 1)
    return orders


def _vector_metrics(
    estimate: torch.Tensor,
    full: torch.Tensor,
    top1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    estimate = estimate.to(torch.float64)
    full = full.to(torch.float64)
    norm_estimate = torch.linalg.vector_norm(estimate, dim=-1)
    norm_full = torch.linalg.vector_norm(full, dim=-1)
    inner = (estimate * full).sum(dim=-1)
    alpha = inner / (norm_estimate.square() + EPS)
    return {
        "norm_ratio": norm_estimate / (norm_full + EPS),
        "rel_err": torch.linalg.vector_norm(full - estimate, dim=-1) / (norm_full + EPS),
        "rel_err_scaled": torch.linalg.vector_norm(
            full - alpha.unsqueeze(-1) * estimate, dim=-1
        ) / (norm_full + EPS),
        "cosine": inner / (norm_estimate * norm_full + EPS),
        "top1_agree": (estimate.argmax(dim=-1) == top1).to(torch.float64),
        "alpha_star": alpha,
    }


def experiment1_metrics(
    mask_logits: torch.Tensor,
    coefficients: torch.Tensor,
) -> dict[str, float]:
    """Compute the four Experiment-1 reconstructions for one or more images."""
    if mask_logits.ndim != 3 or coefficients.shape != mask_logits.shape:
        raise ValueError("mask_logits and coefficients must share shape (B, 2^L, C)")
    mask_count = mask_logits.shape[1]
    if mask_count < 2 or mask_count & (mask_count - 1):
        raise ValueError("mask dimension must be a power of two")
    full = mask_logits[:, -1, :]
    top1 = full.argmax(dim=-1)
    raw = mask_logits.sum(dim=1)
    empty = mask_logits[:, 0, :]
    reconstructions = {
        "raw": raw,
        "mean": mask_logits.mean(dim=1),
        # Preserve the original Experiment-1 summation order.  The equivalent
        # closed form raw-(2^L-1)h(0) has different FP32 cancellation error.
        "ctr": empty + (mask_logits[:, 1:, :] - empty[:, None, :]).sum(dim=1),
        "mob": coefficients.to(torch.float64).sum(dim=1),
    }
    output: dict[str, float] = {}
    for prefix, estimate in reconstructions.items():
        for name, value in _vector_metrics(estimate, full, top1).items():
            if value.numel() != 1:
                raise ValueError("experiment1_metrics currently expects B=1")
            output[f"{prefix}/{name}"] = float(value.item())
    mob = reconstructions["mob"]
    output["mobius_recon_err"] = float(
        (torch.linalg.vector_norm(full - mob, dim=-1) /
         (torch.linalg.vector_norm(full, dim=-1) + EPS)).item()
    )
    return output


def _order_energies(
    coefficients: torch.Tensor,
    scalar_coefficients: torch.Tensor,
    orders: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, mask_count, _ = coefficients.shape
    gates = mask_count.bit_length() - 1
    if orders.shape != (mask_count,):
        raise ValueError("orders has the wrong shape")
    # vector_norm avoids retaining a second coefficient-sized square tensor.
    vector_sq = torch.linalg.vector_norm(coefficients, dim=-1).square()
    scalar_sq = scalar_coefficients.square()
    e_h = torch.zeros(batch, gates + 1, dtype=torch.float64, device=coefficients.device)
    e_v = torch.zeros_like(e_h)
    expanded_orders = orders.view(1, -1).expand(batch, -1)
    e_h.scatter_add_(1, expanded_orders, vector_sq)
    e_v.scatter_add_(1, expanded_orders, scalar_sq)
    return e_v, e_h


def experiment2_metrics(
    coefficients: torch.Tensor,
    full_logits: torch.Tensor,
) -> dict[str, Any]:
    if coefficients.ndim != 3 or full_logits.ndim != 2:
        raise ValueError("expected coefficients (B,M,C) and full_logits (B,C)")
    if coefficients.shape[0] != 1:
        raise ValueError("experiment2_metrics currently expects B=1")
    top1 = full_logits.argmax(dim=-1)
    gather = top1.view(-1, 1, 1).expand(-1, coefficients.shape[1], 1)
    scalar = coefficients.gather(2, gather).squeeze(2)
    orders = interaction_orders(
        coefficients.shape[1].bit_length() - 1, coefficients.device
    )
    e_v, e_h = _order_energies(coefficients, scalar, orders)
    output: dict[str, Any] = {}
    for tag, energy in (("v", e_v), ("h", e_h)):
        derived = derived_metrics(energy)
        output[f"{tag}/E"] = energy[0].detach().cpu().tolist()
        for name in ("E_tilde", "E_bar", "M", "cum", "tail"):
            output[f"{tag}/{name}"] = derived[name][0].detach().cpu().tolist()
        output[f"{tag}/kappa"] = float(derived["kappa"][0].item())
    return output


def complexity_metrics(
    vector_magnitudes: torch.Tensor,
    scalar_magnitudes: torch.Tensor,
    orders_residual: torch.Tensor,
    *,
    low_order_cutoff: int = 3,
) -> dict[str, float]:
    if vector_magnitudes.ndim != 2 or scalar_magnitudes.ndim != 2:
        raise ValueError("complexity magnitudes must have shape (B,M)")
    if vector_magnitudes.shape != scalar_magnitudes.shape:
        raise ValueError("vector and scalar magnitudes must share shape")
    if vector_magnitudes.shape[0] != 1:
        raise ValueError("complexity_metrics currently expects B=1")
    output: dict[str, float] = {}
    for prefix, magnitudes in (("h", vector_magnitudes), ("v", scalar_magnitudes)):
        values = all_complexity_measures(
            magnitudes, orders_residual, K_low=low_order_cutoff
        )
        for name, value in values.items():
            normalized = name.replace("_le_K", f"_le_{low_order_cutoff}")
            normalized = normalized.replace("_gt_K", f"_gt_{low_order_cutoff}")
            output[f"{prefix}/{normalized}"] = float(value.item())
    return output


def difficulty_metrics(full_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if full_logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("difficulty inputs must have shape (B,C) and (B,)")
    if full_logits.shape[0] != 1 or labels.shape != (1,):
        raise ValueError("difficulty_metrics currently expects B=1")
    values = all_difficulty_scores(full_logits, labels)
    return {name: float(value.item()) for name, value in values.items()}


def signature_k_grid(
    residual_count: int,
    fractions: Sequence[float] = (0.01, 0.03, 0.05, 0.10, 0.13, 0.25),
    fixed: Sequence[int] = (32,),
) -> tuple[int, ...]:
    if residual_count < 1:
        raise ValueError("residual_count must be positive")
    values = []
    for fraction in fractions:
        if not (0.0 < float(fraction) <= 1.0):
            raise ValueError("signature fractions must lie in (0,1]")
        values.append(min(residual_count, max(1, math.ceil(fraction * residual_count))))
    for value in fixed:
        value = int(value)
        if value < 1 or value > residual_count:
            raise ValueError(
                f"fixed signature K={value} must lie in [1,{residual_count}]"
            )
        values.append(value)
    return tuple(sorted(set(values)))


def render_spectrum_table(metrics: Mapping[str, Mapping[str, Any]], gates: int) -> str:
    sections: list[str] = ["# Experiment 2 -- ResNet-34 Mobius interaction order spectrum", ""]
    for readout, label in (("v", "predicted-class scalar"), ("h", "logit vector")):
        spectrum = metrics[f"{readout}/E_tilde"]
        average = metrics[f"{readout}/M"]
        cumulative = metrics[f"{readout}/cum"]
        tail = metrics[f"{readout}/tail"]
        kappa = metrics[f"{readout}/kappa"]
        sections.extend([
            f"## Readout: {label}", "",
            "| k | normalized energy | mean per-interaction energy | cumulative | tail |",
            "| -: | :-- | :-- | :-- | :-- |",
        ])
        for index in range(gates):
            order = index + 1
            sections.append(
                f"| {order} | {spectrum['mean'][index]:.6f} +/- {spectrum['stderr'][index]:.2g} | "
                f"{average['mean'][order]:.6g} +/- {average['stderr'][order]:.2g} | "
                f"{cumulative['mean'][index]:.6f} | {tail['mean'][index]:.6f} |"
            )
        sections.extend([
            "",
            f"Effective order: **{kappa['mean']:.6f} +/- {kappa['stderr']:.2g}**",
            "",
        ])
    return "\n".join(sections).rstrip() + "\n"


__all__ = [
    "RECONSTRUCTION_NAMES", "complexity_metrics", "difficulty_metrics",
    "experiment1_metrics", "experiment2_metrics", "interaction_orders",
    "render_spectrum_table", "signature_k_grid",
]
