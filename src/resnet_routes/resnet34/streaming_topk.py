"""Dimension-independent, FP64 streaming Top-K analysis.

Only a ``k_chunk x C`` reconstruction tensor is materialized at once.  Metric
curves are small (one scalar per K) and can either be returned or consumed via
a callback for immediate online aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

import torch


EPS = 1e-12
CurveCallback = Callable[[int, int, Mapping[str, torch.Tensor]], None]


@dataclass
class TopKResult:
    """Metrics produced by a signed Top-K reconstruction."""

    curves: dict[str, torch.Tensor]
    k_effective: dict[str, torch.Tensor]
    k_grid: torch.Tensor
    at_k: dict[str, torch.Tensor]
    order_composition: torch.Tensor | None = None
    order_enrichment: torch.Tensor | None = None


@dataclass
class SubsetGridResult:
    """Metrics for arbitrary selected subsets (low-order/order-matched)."""

    grid: torch.Tensor
    subset_sizes: torch.Tensor
    metrics: dict[str, torch.Tensor]


def infer_l_from_residual_count(m: int) -> int:
    """Infer L from M=2^L-1, rejecting non-mask-space sizes."""
    if m < 1 or (m + 1) & m:
        raise ValueError(f"residual count must equal 2^L-1, got {m}")
    return (m + 1).bit_length() - 1


def residual_orders(l: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Popcount of integer masks 1..2^L-1."""
    if l < 1:
        raise ValueError("L must be positive")
    return torch.tensor(
        [mask.bit_count() for mask in range(1, 1 << l)],
        dtype=torch.long,
        device=device,
    )


def vector_magnitudes(deltas_residual: torch.Tensor) -> torch.Tensor:
    if deltas_residual.ndim != 3:
        raise ValueError("vector residual coefficients must have shape (B, M, C)")
    return torch.linalg.vector_norm(deltas_residual.to(torch.float64), dim=-1)


def scalar_magnitudes(deltas_residual: torch.Tensor) -> torch.Tensor:
    if deltas_residual.ndim != 2:
        raise ValueError("scalar residual coefficients must have shape (B, M)")
    return deltas_residual.to(torch.float64).abs()


def descending_order(magnitudes: torch.Tensor) -> torch.Tensor:
    if magnitudes.ndim != 2:
        raise ValueError("magnitudes must have shape (B, M)")
    return magnitudes.argsort(dim=1, descending=True, stable=True)


def _validate_common(
    residual: torch.Tensor,
    sort_idx: torch.Tensor,
    baseline: torch.Tensor,
    full: torch.Tensor,
) -> tuple[int, int]:
    if residual.ndim not in (2, 3):
        raise ValueError("residual coefficients must be scalar (B,M) or vector (B,M,C)")
    b, m = residual.shape[:2]
    if sort_idx.shape != (b, m):
        raise ValueError(f"sort_idx must have shape {(b, m)}, got {tuple(sort_idx.shape)}")
    if sort_idx.dtype != torch.long:
        raise TypeError("sort_idx must be torch.long")
    if sort_idx.numel() and (int(sort_idx.min()) < 0 or int(sort_idx.max()) >= m):
        raise IndexError("sort_idx contains an out-of-range residual index")
    expected = (b,) if residual.ndim == 2 else (b, residual.shape[-1])
    if tuple(baseline.shape) != expected or tuple(full.shape) != expected:
        raise ValueError(f"baseline and full must both have shape {expected}")
    return b, m


def _normalize_k_grid(k_grid: Sequence[int] | None, m: int, device: torch.device) -> torch.Tensor:
    values = sorted(set(int(k) for k in (k_grid or ())))
    if any(k < 1 or k > m for k in values):
        raise ValueError(f"all reporting K values must lie in [1, {m}]")
    return torch.tensor(values, dtype=torch.long, device=device)


def _gather_residual(residual: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if residual.ndim == 2:
        return torch.gather(residual, 1, indices)
    expanded = indices.unsqueeze(-1).expand(-1, -1, residual.shape[-1])
    return torch.gather(residual, 1, expanded)


def _full_margin(full: torch.Tensor, top1: torch.Tensor) -> torch.Tensor:
    target = full.gather(1, top1[:, None]).squeeze(1)
    other = full.clone()
    other.scatter_(1, top1[:, None], float("-inf"))
    return target - other.max(dim=1).values


def _vector_metrics(
    reconstruction: torch.Tensor,
    full: torch.Tensor,
    top1: torch.Tensor,
    full_norm: torch.Tensor,
    full_margin: torch.Tensor,
) -> dict[str, torch.Tensor]:
    diff = full[:, None, :] - reconstruction
    err = torch.linalg.vector_norm(diff, dim=-1) / (full_norm[:, None] + EPS)
    recon_norm = torch.linalg.vector_norm(reconstruction, dim=-1)
    cosine = (reconstruction * full[:, None, :]).sum(dim=-1) / (
        recon_norm * full_norm[:, None] + EPS
    )
    prediction = reconstruction.argmax(dim=-1)
    agreement = (prediction == top1[:, None]).to(torch.float64)
    target_idx = top1[:, None, None].expand(-1, reconstruction.shape[1], 1)
    target = reconstruction.gather(2, target_idx).squeeze(2)
    other = reconstruction.clone()
    other.scatter_(2, target_idx, float("-inf"))
    margin = target - other.max(dim=2).values
    return {
        "err_h": err,
        "cosine": cosine,
        "agreement": agreement,
        "margin": margin,
        "margin_ratio": margin / (full_margin[:, None] + EPS),
    }


def _first_crossing_update(
    destination: torch.Tensor,
    curve: torch.Tensor,
    *,
    start_k: int,
    threshold: float,
    direction: str,
) -> None:
    unresolved = destination == 0
    if direction == "le":
        satisfied = curve <= threshold
    elif direction == "ge":
        satisfied = curve >= threshold
    else:
        raise ValueError(direction)
    active = unresolved & satisfied.any(dim=1)
    if active.any():
        first_local = satisfied.to(torch.int8).argmax(dim=1)
        destination[active] = start_k + first_local[active]


def _collect_reporting_points(
    destination: dict[str, list[torch.Tensor]],
    metrics: Mapping[str, torch.Tensor],
    *,
    start_k: int,
    stop_k: int,
    k_values: Sequence[int],
) -> None:
    for k in k_values:
        if start_k <= k <= stop_k:
            offset = k - start_k
            for name, curve in metrics.items():
                destination.setdefault(name, []).append(curve[:, offset].detach().cpu())


def _order_statistics(
    sort_idx: torch.Tensor,
    orders: torch.Tensor,
    k_grid: torch.Tensor,
    l: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, m = sort_idx.shape
    if tuple(orders.shape) != (m,):
        raise ValueError(f"orders_residual must have shape {(m,)}")
    if orders.numel() and (int(orders.min()) < 1 or int(orders.max()) > l):
        raise ValueError("residual interaction orders must lie in [1, L]")
    if k_grid.numel() == 0:
        empty = torch.empty(b, 0, l + 1, dtype=torch.float64)
        return empty, empty.clone()
    sorted_orders = orders.to(sort_idx.device)[sort_idx]
    compositions: list[torch.Tensor] = []
    for k in k_grid.tolist():
        counts = torch.stack(
            [(sorted_orders[:, :k] == order).sum(dim=1) for order in range(l + 1)],
            dim=1,
        ).to(torch.float64)
        compositions.append(counts / k)
    composition = torch.stack(compositions, dim=1).cpu()
    base = torch.zeros(l + 1, dtype=torch.float64)
    for order in range(1, l + 1):
        base[order] = math.comb(l, order) / m
    enrichment = torch.zeros_like(composition)
    enrichment[:, :, 1:] = composition[:, :, 1:] / base[None, None, 1:]
    return composition, enrichment


def evaluate_vector_topk(
    deltas_h: torch.Tensor,
    *,
    full_output: torch.Tensor | None = None,
    sort_idx: torch.Tensor | None = None,
    magnitudes: torch.Tensor | None = None,
    k_chunk: int = 512,
    k_grid: Sequence[int] | None = None,
    orders_residual: torch.Tensor | None = None,
    error_thresholds: Sequence[float] = (0.10, 0.05),
    mass_thresholds: Sequence[float] = (0.90, 0.95, 0.99),
    return_curves: bool = True,
    callback: CurveCallback | None = None,
    accumulation_dtype: torch.dtype = torch.float64,
) -> TopKResult:
    """Evaluate the complete vector Top-K curve using K-sized chunks.

    ``deltas_h`` contains the empty coefficient at index 0 and has shape
    ``(B, M+1, C)``.  Sorting indexes the non-empty slice.  FP64 is the
    experiment default and is used for all reconstruction/metric arithmetic.
    Callback bounds are inclusive one-based K values.
    """
    if deltas_h.ndim != 3 or deltas_h.shape[1] < 2:
        raise ValueError("deltas_h must have shape (B, M+1, C)")
    if not accumulation_dtype.is_floating_point:
        raise TypeError("accumulation_dtype must be floating point")
    residual = deltas_h[:, 1:, :].to(accumulation_dtype)
    baseline = deltas_h[:, 0, :].to(accumulation_dtype)
    full = (
        deltas_h.to(accumulation_dtype).sum(dim=1)
        if full_output is None
        else full_output.to(device=residual.device, dtype=accumulation_dtype)
    )
    if magnitudes is None:
        magnitudes = vector_magnitudes(residual)
    else:
        magnitudes = magnitudes.to(device=residual.device, dtype=accumulation_dtype)
    if sort_idx is None:
        sort_idx = descending_order(magnitudes)
    else:
        sort_idx = sort_idx.to(residual.device)
    b, m = _validate_common(residual, sort_idx, baseline, full)
    if tuple(magnitudes.shape) != (b, m):
        raise ValueError(f"magnitudes must have shape {(b, m)}")
    if k_chunk < 1:
        raise ValueError("k_chunk must be positive")
    report_grid = _normalize_k_grid(k_grid, m, residual.device)
    report_lists: dict[str, list[torch.Tensor]] = {}
    curve_lists: dict[str, list[torch.Tensor]] = {}
    top1 = full.argmax(dim=1)
    full_norm = torch.linalg.vector_norm(full, dim=1)
    full_margin = _full_margin(full, top1)
    total_mass = magnitudes.square().sum(dim=1)
    running_mass = torch.zeros(b, dtype=accumulation_dtype, device=residual.device)
    running = baseline.clone()
    k_eff = {
        **{f"err_h_le_{value:g}": torch.zeros(b, dtype=torch.long, device=residual.device)
           for value in error_thresholds},
        **{f"mass_ge_{value:g}": torch.zeros(b, dtype=torch.long, device=residual.device)
           for value in mass_thresholds},
    }

    for start in range(0, m, k_chunk):
        stop = min(start + k_chunk, m)
        indices = sort_idx[:, start:stop]
        selected = _gather_residual(residual, indices)
        recon = selected.cumsum(dim=1) + running[:, None, :]
        running = recon[:, -1, :].clone()
        selected_sq = torch.gather(magnitudes.square(), 1, indices)
        mass = (selected_sq.cumsum(dim=1) + running_mass[:, None]) / (total_mass[:, None] + EPS)
        running_mass = running_mass + selected_sq.sum(dim=1)
        metrics = _vector_metrics(recon, full, top1, full_norm, full_margin)
        metrics["captured_mass"] = mass
        start_k, stop_k = start + 1, stop
        for threshold in error_thresholds:
            _first_crossing_update(
                k_eff[f"err_h_le_{threshold:g}"], metrics["err_h"],
                start_k=start_k, threshold=float(threshold), direction="le",
            )
        for threshold in mass_thresholds:
            _first_crossing_update(
                k_eff[f"mass_ge_{threshold:g}"], mass,
                start_k=start_k, threshold=float(threshold), direction="ge",
            )
        if return_curves:
            for name, value in metrics.items():
                curve_lists.setdefault(name, []).append(value.detach().cpu())
        _collect_reporting_points(
            report_lists, metrics, start_k=start_k, stop_k=stop_k,
            k_values=report_grid.tolist(),
        )
        if callback is not None:
            callback(start_k, stop_k, metrics)

    # M is the documented sentinel if numerical noise prevents a crossing.
    for value in k_eff.values():
        value[value == 0] = m
    curves = {
        name: torch.cat(parts, dim=1) for name, parts in curve_lists.items()
    }
    at_k = {
        name: torch.stack(parts, dim=1) if parts else torch.empty(b, 0)
        for name, parts in report_lists.items()
    }
    l = infer_l_from_residual_count(m)
    if orders_residual is None:
        orders_residual = residual_orders(l)
    composition, enrichment = _order_statistics(
        sort_idx, orders_residual, report_grid, l
    )
    return TopKResult(
        curves=curves,
        k_effective={name: value.cpu() for name, value in k_eff.items()},
        k_grid=report_grid.cpu(),
        at_k=at_k,
        order_composition=composition,
        order_enrichment=enrichment,
    )


def evaluate_scalar_topk(
    deltas_v: torch.Tensor,
    *,
    full_output: torch.Tensor | None = None,
    sort_idx: torch.Tensor | None = None,
    magnitudes: torch.Tensor | None = None,
    k_chunk: int = 512,
    k_grid: Sequence[int] | None = None,
    error_thresholds: Sequence[float] = (),
    mass_thresholds: Sequence[float] = (0.90, 0.95, 0.99),
    return_curves: bool = True,
    callback: CurveCallback | None = None,
    accumulation_dtype: torch.dtype = torch.float64,
) -> TopKResult:
    """Scalar counterpart of :func:`evaluate_vector_topk`."""
    if deltas_v.ndim != 2 or deltas_v.shape[1] < 2:
        raise ValueError("deltas_v must have shape (B, M+1)")
    if not accumulation_dtype.is_floating_point:
        raise TypeError("accumulation_dtype must be floating point")
    residual = deltas_v[:, 1:].to(accumulation_dtype)
    baseline = deltas_v[:, 0].to(accumulation_dtype)
    full = (
        deltas_v.to(accumulation_dtype).sum(dim=1)
        if full_output is None
        else full_output.to(device=residual.device, dtype=accumulation_dtype)
    )
    if magnitudes is None:
        magnitudes = scalar_magnitudes(residual)
    else:
        magnitudes = magnitudes.to(device=residual.device, dtype=accumulation_dtype)
    if sort_idx is None:
        sort_idx = descending_order(magnitudes)
    else:
        sort_idx = sort_idx.to(residual.device)
    b, m = _validate_common(residual, sort_idx, baseline, full)
    if tuple(magnitudes.shape) != (b, m):
        raise ValueError(f"magnitudes must have shape {(b, m)}")
    if k_chunk < 1:
        raise ValueError("k_chunk must be positive")
    report_grid = _normalize_k_grid(k_grid, m, residual.device)
    report_lists: dict[str, list[torch.Tensor]] = {}
    curve_lists: dict[str, list[torch.Tensor]] = {}
    total_mass = magnitudes.square().sum(dim=1)
    running_mass = torch.zeros(b, dtype=accumulation_dtype, device=residual.device)
    running = baseline.clone()
    k_eff = {
        **{f"err_v_le_{value:g}": torch.zeros(b, dtype=torch.long, device=residual.device)
           for value in error_thresholds},
        **{f"mass_ge_{value:g}": torch.zeros(b, dtype=torch.long, device=residual.device)
           for value in mass_thresholds},
    }
    for start in range(0, m, k_chunk):
        stop = min(start + k_chunk, m)
        indices = sort_idx[:, start:stop]
        selected = _gather_residual(residual, indices)
        reconstruction = selected.cumsum(dim=1) + running[:, None]
        running = reconstruction[:, -1].clone()
        err = (full[:, None] - reconstruction).abs() / (full.abs()[:, None] + EPS)
        selected_sq = torch.gather(magnitudes.square(), 1, indices)
        mass = (selected_sq.cumsum(dim=1) + running_mass[:, None]) / (total_mass[:, None] + EPS)
        running_mass = running_mass + selected_sq.sum(dim=1)
        metrics = {"err_v": err, "captured_mass": mass}
        start_k, stop_k = start + 1, stop
        for threshold in error_thresholds:
            _first_crossing_update(
                k_eff[f"err_v_le_{threshold:g}"], err,
                start_k=start_k, threshold=float(threshold), direction="le",
            )
        for threshold in mass_thresholds:
            _first_crossing_update(
                k_eff[f"mass_ge_{threshold:g}"], mass,
                start_k=start_k, threshold=float(threshold), direction="ge",
            )
        if return_curves:
            for name, value in metrics.items():
                curve_lists.setdefault(name, []).append(value.detach().cpu())
        _collect_reporting_points(
            report_lists, metrics, start_k=start_k, stop_k=stop_k,
            k_values=report_grid.tolist(),
        )
        if callback is not None:
            callback(start_k, stop_k, metrics)
    for value in k_eff.values():
        value[value == 0] = m
    return TopKResult(
        curves={name: torch.cat(parts, dim=1) for name, parts in curve_lists.items()},
        k_effective={name: value.cpu() for name, value in k_eff.items()},
        k_grid=report_grid.cpu(),
        at_k={name: torch.stack(parts, dim=1) for name, parts in report_lists.items()},
    )


def random_permutations(
    batch_size: int,
    residual_count: int,
    seeds: Iterable[int],
    *,
    device: torch.device | str = "cpu",
) -> dict[int, torch.Tensor]:
    """Create deterministic independent random rankings for baseline seeds."""
    if batch_size < 1 or residual_count < 1:
        raise ValueError("batch_size and residual_count must be positive")
    output: dict[int, torch.Tensor] = {}
    for seed in seeds:
        seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows = [torch.randperm(residual_count, generator=generator) for _ in range(batch_size)]
        output[seed] = torch.stack(rows).to(device=device)
    return output


def order_matched_indices(
    magnitude_sort_idx: torch.Tensor,
    orders_residual: torch.Tensor,
    k_grid: Sequence[int],
    *,
    seed: int,
) -> dict[int, torch.Tensor]:
    """Sample subsets matching the exact per-image Top-K order histogram.

    ``seed`` should already encode global seed, sample ID and baseline seed.
    Each K is mixed into it here, so the sample at K is unchanged if the
    reporting grid is reordered or expanded.
    """
    if magnitude_sort_idx.ndim != 2:
        raise ValueError("magnitude_sort_idx must have shape (B, M)")
    b, m = magnitude_sort_idx.shape
    l = infer_l_from_residual_count(m)
    if tuple(orders_residual.shape) != (m,):
        raise ValueError(f"orders_residual must have shape {(m,)}")
    grid = _normalize_k_grid(k_grid, m, magnitude_sort_idx.device).tolist()
    orders_cpu = orders_residual.detach().cpu()
    ranked_orders = orders_residual.to(magnitude_sort_idx.device)[magnitude_sort_idx].cpu()
    pools = {order: torch.where(orders_cpu == order)[0] for order in range(1, l + 1)}
    output: dict[int, torch.Tensor] = {}
    for k in grid:
        # SplitMix64 constants give a stable, non-Python-hash seed derivation.
        k_seed = (int(seed) + 0x9E3779B97F4A7C15 * int(k)) & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(k_seed)
        rows: list[torch.Tensor] = []
        for row in range(b):
            chunks: list[torch.Tensor] = []
            for order in range(1, l + 1):
                count = int((ranked_orders[row, :k] == order).sum())
                if count:
                    pool = pools[order]
                    perm = torch.randperm(pool.numel(), generator=generator)
                    chunks.append(pool[perm[:count]])
            rows.append(torch.cat(chunks))
        output[k] = torch.stack(rows).to(magnitude_sort_idx.device)
    return output


def evaluate_subset_grid(
    deltas: torch.Tensor,
    selected_indices: Mapping[int, torch.Tensor],
    *,
    full_output: torch.Tensor | None = None,
    magnitudes: torch.Tensor | None = None,
    accumulation_dtype: torch.dtype = torch.float64,
) -> SubsetGridResult:
    """Evaluate arbitrary non-empty subsets at named integer grid points.

    This is used by order-matched random and can also represent hand-specified
    baselines.  Values in ``selected_indices`` index ``deltas[:, 1:]``.
    """
    if deltas.ndim not in (2, 3) or deltas.shape[1] < 2:
        raise ValueError("deltas must have shape (B,M+1) or (B,M+1,C)")
    residual = deltas[:, 1:].to(accumulation_dtype)
    baseline = deltas[:, 0].to(accumulation_dtype)
    full = (
        deltas.to(accumulation_dtype).sum(dim=1)
        if full_output is None
        else full_output.to(device=residual.device, dtype=accumulation_dtype)
    )
    b, m = residual.shape[:2]
    grid = sorted(int(k) for k in selected_indices)
    if not grid:
        raise ValueError("selected_indices cannot be empty")
    all_metrics: dict[str, list[torch.Tensor]] = {}
    sizes: list[int] = []
    if deltas.ndim == 3:
        top1 = full.argmax(dim=1)
        full_norm = torch.linalg.vector_norm(full, dim=1)
        full_margin = _full_margin(full, top1)
    if magnitudes is not None:
        magnitudes = magnitudes.to(device=residual.device, dtype=accumulation_dtype)
        if tuple(magnitudes.shape) != (b, m):
            raise ValueError(f"magnitudes must have shape {(b, m)}")
        total_mass = magnitudes.square().sum(dim=1)
    for key in grid:
        indices = selected_indices[key].to(residual.device)
        if indices.ndim != 2 or indices.shape[0] != b:
            raise ValueError(f"indices at grid {key} must have shape (B,K)")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= m):
            raise IndexError(f"indices at grid {key} are outside [0,{m})")
        sizes.append(indices.shape[1])
        selected_sum = _gather_residual(residual, indices).sum(dim=1)
        reconstruction = baseline + selected_sum
        if deltas.ndim == 2:
            metrics = {
                "err_v": (full - reconstruction).abs() / (full.abs() + EPS),
            }
        else:
            metrics = {
                name: value[:, 0]
                for name, value in _vector_metrics(
                    reconstruction[:, None, :], full, top1, full_norm, full_margin
                ).items()
            }
        if magnitudes is not None:
            chosen_mass = torch.gather(magnitudes.square(), 1, indices).sum(dim=1)
            metrics["captured_mass"] = chosen_mass / (total_mass + EPS)
        for name, value in metrics.items():
            all_metrics.setdefault(name, []).append(value.detach().cpu())
    return SubsetGridResult(
        grid=torch.tensor(grid, dtype=torch.long),
        subset_sizes=torch.tensor(sizes, dtype=torch.long),
        metrics={name: torch.stack(values, dim=1) for name, values in all_metrics.items()},
    )


def evaluate_low_order(
    deltas: torch.Tensor,
    orders_residual: torch.Tensor,
    *,
    max_order: int | None = None,
    full_output: torch.Tensor | None = None,
    magnitudes: torch.Tensor | None = None,
    accumulation_dtype: torch.dtype = torch.float64,
) -> SubsetGridResult:
    """Evaluate all low-order truncations ``1 <= |S| <= r`` efficiently."""
    m = deltas.shape[1] - 1
    l = infer_l_from_residual_count(m)
    if tuple(orders_residual.shape) != (m,):
        raise ValueError(f"orders_residual must have shape {(m,)}")
    max_order = l if max_order is None else int(max_order)
    if max_order < 1 or max_order > l:
        raise ValueError(f"max_order must lie in [1, {l}]")
    if not accumulation_dtype.is_floating_point:
        raise TypeError("accumulation_dtype must be floating point")
    residual = deltas[:, 1:].to(accumulation_dtype)
    baseline = deltas[:, 0].to(accumulation_dtype)
    full = (
        deltas.to(accumulation_dtype).sum(dim=1)
        if full_output is None
        else full_output.to(device=residual.device, dtype=accumulation_dtype)
    )
    b = deltas.shape[0]
    running = baseline.clone()
    metrics_by_name: dict[str, list[torch.Tensor]] = {}
    sizes: list[int] = []
    if deltas.ndim == 3:
        top1 = full.argmax(dim=1)
        full_norm = torch.linalg.vector_norm(full, dim=1)
        full_margin = _full_margin(full, top1)
    if magnitudes is not None:
        magnitudes = magnitudes.to(device=residual.device, dtype=accumulation_dtype)
        if tuple(magnitudes.shape) != (b, m):
            raise ValueError(f"magnitudes must have shape {(b, m)}")
        total_mass = magnitudes.square().sum(dim=1)
        running_mass = torch.zeros(b, dtype=accumulation_dtype, device=residual.device)
    running_size = 0
    orders_device = orders_residual.to(residual.device)
    for order in range(1, max_order + 1):
        indices = torch.where(orders_device == order)[0]
        running = running + residual.index_select(1, indices).sum(dim=1)
        running_size += indices.numel()
        sizes.append(running_size)
        if deltas.ndim == 2:
            point_metrics = {
                "err_v": (full - running).abs() / (full.abs() + EPS),
            }
        else:
            point_metrics = {
                name: value[:, 0]
                for name, value in _vector_metrics(
                    running[:, None, :], full, top1, full_norm, full_margin
                ).items()
            }
        if magnitudes is not None:
            running_mass = running_mass + magnitudes.index_select(1, indices).square().sum(dim=1)
            point_metrics["captured_mass"] = running_mass / (total_mass + EPS)
        for name, value in point_metrics.items():
            metrics_by_name.setdefault(name, []).append(value.detach().cpu())
    return SubsetGridResult(
        grid=torch.arange(1, max_order + 1, dtype=torch.long),
        subset_sizes=torch.tensor(sizes, dtype=torch.long),
        metrics={
            name: torch.stack(values, dim=1) for name, values in metrics_by_name.items()
        },
    )
