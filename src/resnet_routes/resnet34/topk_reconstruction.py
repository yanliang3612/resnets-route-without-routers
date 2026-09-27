"""Exact ResNet-34 Experiment 3 evaluator.

This entry point evaluates all 65,536 block-gate masks for one image at a
time, computes the FP64 Möbius decomposition, performs signed magnitude Top-K
reconstruction, and evaluates the three preregistered baselines.  Large
per-image curves are reduced online; only reporting-grid values and effective
K summaries are retained per image.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import ResNet34_Weights, resnet34

from .data import (
    DEFAULT_SPLITS,
    ManifestParquetDataset,
    PERMUTATION_SEED,
    SampleRecord,
    load_permutation_manifest,
    select_records,
)
from .model import GATE_NAMES, LSB_GATE_MAPPING, build_gated_resnet34
from .io_utils import (
    atomic_save_checkpoint,
    atomic_write_json,
    canonical_json_sha256,
    load_torch_checkpoint,
    sha256_file,
)
from .mask_evaluator import evaluate_all_masks
from .mobius import full_reconstruction, mobius_coefficients
from .online_stats import OnlineScalarDistribution, OnlineStats
from .streaming_topk import (
    descending_order,
    evaluate_low_order,
    evaluate_scalar_topk,
    evaluate_subset_grid,
    evaluate_vector_topk,
    order_matched_indices,
    random_permutations,
    residual_orders,
    scalar_magnitudes,
    vector_magnitudes,
)


SCHEMA_VERSION = 1
L = len(GATE_NAMES)
NUM_MASKS = 1 << L
M = NUM_MASKS - 1
EPS = 1e-12
OFFICIAL_SAMPLE_INDEX_SHA256 = (
    "39fe1032a3a84de70434f403afadd51f13c0156a15d2c3a839c1af9d9d3949c0"
)
OFFICIAL_SELECTION_MANIFEST_SHA256 = (
    "9e00c1a6f008f73d99650fe69a79ad49e40e12c7796152e0fad2cb77aae7d276"
)

ABSOLUTE_K_GRID = tuple([1 << exponent for exponent in range(16)] + [M])
P_GRID = (0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
ERROR_THRESHOLDS = (0.10, 0.05)
MASS_THRESHOLDS = (0.90, 0.95, 0.99)
DEFAULT_BASELINE_SEEDS = (12345, 12346, 12347, 12348, 12349)

# Files that can change the numerical result of the evaluator.  Their hashes
# are part of the checkpoint fingerprint so an uncommitted (or wholly
# untracked) implementation cannot silently resume a checkpoint produced by
# different code.
NUMERICAL_IMPLEMENTATION_FILES = (
    "data.py",
    "topk_reconstruction.py",
    "model.py",
    "io_utils.py",
    "mask_evaluator.py",
    "mobius.py",
    "online_stats.py",
    "streaming_topk.py",
)

CURVE_METRICS = {
    "curve.magnitude.scalar.captured_mass": ("scalar", "captured_mass"),
    "curve.magnitude.scalar.relative_error": ("scalar", "err_v"),
    "curve.magnitude.vector.captured_mass": ("vector", "captured_mass"),
    "curve.magnitude.vector.relative_error": ("vector", "err_h"),
    "curve.magnitude.vector.cosine": ("vector", "cosine"),
    "curve.magnitude.vector.agreement": ("vector", "agreement"),
    "curve.magnitude.vector.margin": ("vector", "margin"),
    "curve.magnitude.vector.margin_ratio": ("vector", "margin_ratio"),
}

VECTOR_NAME_MAP = {
    "captured_mass": "captured_mass",
    "err_h": "relative_error",
    "cosine": "cosine",
    "agreement": "agreement",
    "margin": "margin",
    "margin_ratio": "margin_ratio",
}
SCALAR_NAME_MAP = {
    "captured_mass": "captured_mass",
    "err_v": "relative_error",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _threshold_key(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def make_reporting_k_grid() -> tuple[int, ...]:
    proportional = [int(math.ceil(proportion * M)) for proportion in P_GRID]
    return tuple(sorted(set((*ABSOLUTE_K_GRID, *proportional))))


def low_order_sizes() -> tuple[int, ...]:
    running = 0
    sizes: list[int] = []
    for order in range(1, L + 1):
        running += math.comb(L, order)
        sizes.append(running)
    return tuple(sizes)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-index", type=Path,
        default=Path("data/indices/imagenet_test_10000.json"),
    )
    parser.add_argument(
        "--selection-manifest", type=Path,
        default=Path("data/indices/resnet34_test_splits_seed3403.json"),
    )
    parser.add_argument(
        "--parquet-dir", type=Path, default=Path("imagenet-1k/data"),
        help="directory containing ImageNet parquet shards",
    )
    parser.add_argument(
        "--split", default="main",
        choices=(
            "smoke", "benchmark", "pilot", "main", "extension",
            "extension_append",
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mask-chunk", type=int, default=1024)
    parser.add_argument("--k-chunk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-random-seeds", type=int, default=5)
    parser.add_argument(
        "--baseline-mode", choices=("none", "random", "all"), default="all",
        help="none for throughput benchmarks; random omits low/order-matched",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--stop-after-images",
        type=int,
        default=None,
        help="gracefully pause after this many newly processed images (resume test)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/topk_interaction_reconstruction/generated/resnet34"),
    )
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--run-note", default="")
    args = parser.parse_args(argv)
    if args.batch_size != 1:
        parser.error("the exact ResNet-34 evaluator requires --batch-size 1")
    if args.mask_chunk < 1 or args.k_chunk < 1:
        parser.error("mask/k chunks must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    if args.num_random_seeds < 1 and args.baseline_mode != "none":
        parser.error("baselines require at least one random seed")
    if args.num_random_seeds > len(DEFAULT_BASELINE_SEEDS):
        parser.error(f"at most {len(DEFAULT_BASELINE_SEEDS)} fixed seeds are defined")
    if args.checkpoint_every < 1 or args.log_every < 1:
        parser.error("checkpoint/log intervals must be positive")
    if args.stop_after_images is not None and args.stop_after_images < 1:
        parser.error("--stop-after-images must be positive")
    return args


def _validate_official_run_contract(
    args: argparse.Namespace,
    *,
    manifest: Any,
    manifest_sha256: str,
    source_sha256: str,
) -> None:
    """Reject configurations that would look official but violate the plan."""
    if int(manifest.seed) != PERMUTATION_SEED:
        raise ValueError(
            f"selection manifest seed is {manifest.seed}, expected {PERMUTATION_SEED}"
        )
    if manifest.count != 10_000:
        raise ValueError(
            f"selection manifest contains {manifest.count} samples, expected 10000"
        )
    if source_sha256 != OFFICIAL_SAMPLE_INDEX_SHA256:
        raise ValueError(
            "sample-index SHA256 does not match the frozen Experiment 3 source"
        )
    if manifest_sha256 != OFFICIAL_SELECTION_MANIFEST_SHA256:
        raise ValueError(
            "selection-manifest SHA256 does not match the frozen Experiment 3 manifest"
        )
    for name, expected in DEFAULT_SPLITS.items():
        actual = manifest.splits.get(name)
        normalized = (
            {"start": int(actual["start"]), "count": int(actual["count"])}
            if actual is not None
            else None
        )
        if normalized != expected:
            raise ValueError(
                f"selection split {name!r} is {normalized}, expected {expected}"
            )

    required = {
        "smoke": ("all", 1),
        "benchmark": ("none", 1),
        "pilot": ("all", 1),
        "main": ("all", 5),
        "extension": ("all", 5),
        "extension_append": ("all", 5),
    }
    expected_mode, expected_seeds = required[args.split]
    if args.baseline_mode != expected_mode:
        raise ValueError(
            f"split {args.split!r} requires --baseline-mode {expected_mode}"
        )
    if args.num_random_seeds != expected_seeds:
        raise ValueError(
            f"split {args.split!r} requires --num-random-seeds {expected_seeds}"
        )
    if args.seed != 0:
        raise ValueError("the frozen Experiment 3 protocol requires --seed 0")


def _implementation_hashes(repo_root: Path) -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    result: dict[str, str] = {}
    for name in NUMERICAL_IMPLEMENTATION_FILES:
        path = package_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing numerical implementation file: {path}")
        result[path.relative_to(repo_root).as_posix()] = sha256_file(path)
    return result


def _relocate_records(
    records: Sequence[SampleRecord], parquet_dir: Path
) -> list[SampleRecord]:
    """Relocate frozen shard paths by basename without changing sample IDs."""
    root = parquet_dir.expanduser().resolve()
    relocated: list[SampleRecord] = []
    for record in records:
        shard = root / Path(record.shard).name
        if not shard.is_file():
            raise FileNotFoundError(
                f"missing parquet shard {shard}; check --parquet-dir"
            )
        relocated.append(replace(record, shard=str(shard)))
    return relocated


def configure_determinism(seed: int, *, cudnn_benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    # Benchmarking is allowed to choose the fastest algorithm, but cuDNN is
    # constrained to deterministic implementations.
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def _git_metadata(root: Path) -> dict[str, Any]:
    def command(*parts: str) -> str:
        try:
            return subprocess.check_output(
                parts, cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    status = command("git", "status", "--short")
    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "dirty": bool(status and status != "unknown"),
        "status": status.splitlines(),
    }


def _device_metadata(device: torch.device) -> dict[str, Any]:
    data: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        data.update(
            {
                "gpu": props.name,
                "gpu_total_bytes": props.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return data


def _derived_seed(
    sample_id: str,
    family: str,
    base_seed: int,
    global_seed: int,
) -> int:
    payload = (
        f"resnet34-exp3|{global_seed}|{sample_id}|{family}|{base_seed}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _static_config(
    args: argparse.Namespace,
    *,
    logical_records: Sequence[SampleRecord],
    processing_records: Sequence[SampleRecord],
    manifest_sha256: str,
    source_sha256: str,
    repo_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    baseline_seeds = list(DEFAULT_BASELINE_SEEDS[: args.num_random_seeds])
    report_k = list(make_reporting_k_grid())
    device_metadata = _device_metadata(device)
    git_metadata = _git_metadata(repo_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "Experiment 3 exact block-level ResNet-34 only",
        "architecture": "torchvision.models.resnet34",
        "weights": "ResNet34_Weights.IMAGENET1K_V1",
        "weights_url": ResNet34_Weights.IMAGENET1K_V1.url,
        "gate_names": list(GATE_NAMES),
        "lsb_gate_mapping": [[bit, name] for bit, name in LSB_GATE_MAPPING],
        "bit_order": "LSB; bit l maps to gate_names[l]",
        "L": L,
        "num_masks": NUM_MASKS,
        "num_residual_terms": M,
        "num_classes": 1000,
        "sample_index": str(args.sample_index.resolve()),
        "sample_index_path": str(args.sample_index.resolve()),
        "sample_index_sha256": source_sha256,
        "parquet_dir": str(args.parquet_dir.expanduser().resolve()),
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_path": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": manifest_sha256,
        "selection_seed": PERMUTATION_SEED,
        "split": args.split,
        "num_images": len(logical_records),
        "logical_sample_ids": [record.sample_id for record in logical_records],
        "sample_ids": [record.sample_id for record in logical_records],
        "logical_selection_positions": [record.selection_position for record in logical_records],
        "permutation_positions": [
            record.selection_position for record in logical_records
        ],
        "processing_sample_ids": [record.sample_id for record in processing_records],
        "processing_order": "sorted by (shard,row) for row-group cache efficiency",
        "global_seed": args.seed,
        "baseline_seeds": baseline_seeds,
        "baseline_seed_derivation": (
            "sha256('resnet34-exp3|global_seed|sample_id|family|baseline_seed')"
        ),
        "baseline_mode": args.baseline_mode,
        "absolute_k_grid": list(ABSOLUTE_K_GRID),
        "p_grid": list(P_GRID),
        "proportion_grid": list(P_GRID),
        "reporting_k": report_k,
        "error_thresholds": list(ERROR_THRESHOLDS),
        "mass_thresholds": list(MASS_THRESHOLDS),
        "low_order_r": list(range(1, L + 1)),
        "low_order_k": list(low_order_sizes()),
        "batch_size": 1,
        "mask_chunk": args.mask_chunk,
        "k_chunk": args.k_chunk,
        "num_workers": args.num_workers,
        "forward_dtype": "float32",
        "raw_logits_dtype": "float32",
        "transform_dtype": "float64",
        "accumulation_dtype": "float64",
        "fp32_transform_sensitivity": {
            "enabled": args.split == "pilot",
            "selection_positions": list(range(5)) if args.split == "pilot" else [],
            "purpose": "diagnostic only; excluded from formal FP64 endpoints",
        },
        "epsilon": EPS,
        "tf32": False,
        "amp": False,
        "cudnn_benchmark": args.cudnn_benchmark,
        "cudnn_deterministic": True,
        "checkpoint_every": args.checkpoint_every,
        "run_note": args.run_note,
        "device": str(device),
        "GPU": device_metadata.get("gpu"),
        "CUDA": device_metadata.get("cuda"),
        "torch_version": device_metadata.get("torch"),
        "torchvision_version": device_metadata.get("torchvision"),
        "software_hardware": device_metadata,
        "numerical_implementation_sha256": _implementation_hashes(repo_root),
        "git_commit": git_metadata.get("commit"),
        "git": git_metadata,
    }


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash run-defining fields, using explicit source hashes for code identity."""
    payload = json.loads(json.dumps(config))
    # Git metadata is provenance, not identity: the working tree may become
    # dirty merely because output artifacts are created.  Numerical source
    # identity is instead enforced by numerical_implementation_sha256 above.
    payload.pop("git", None)
    return canonical_json_sha256(payload)


def _verify_reference_model(
    gated_model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    reference = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device).eval()
    gated_model.eval()
    with torch.inference_mode():
        expected = reference(x)
        actual = gated_model(x)
    difference = (actual - expected).abs()
    ref_parameters = dict(reference.named_parameters())
    gated_parameters = dict(gated_model.named_parameters())
    parameter_keys_match = tuple(ref_parameters) == tuple(gated_parameters)
    parameters_equal = parameter_keys_match and all(
        torch.equal(ref_parameters[name], gated_parameters[name])
        for name in ref_parameters
    )
    result = {
        "performed": True,
        "parameter_keys_match": parameter_keys_match,
        "parameters_equal": parameters_equal,
        "max_abs_logit_diff": float(difference.max().item()),
        "mean_abs_logit_diff": float(difference.mean().item()),
        "allclose_rtol_1e-5_atol_1e-6": bool(
            torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
        ),
        "top1_equal": bool(torch.equal(actual.argmax(1), expected.argmax(1))),
        "reference_top1": int(expected.argmax(1).item()),
        "gated_top1": int(actual.argmax(1).item()),
    }
    del reference, expected, actual, difference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not (
        result["parameters_equal"]
        and result["max_abs_logit_diff"] <= 1e-5
        and result["allclose_rtol_1e-5_atol_1e-6"]
        and result["top1_equal"]
    ):
        raise RuntimeError(f"all-open gated model verification failed: {result}")
    return result


def _point_vector_metrics(
    reconstruction: torch.Tensor,
    full: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if reconstruction.ndim != 2 or full.shape != reconstruction.shape:
        raise ValueError("point vector inputs must both have shape (B,C)")
    top1 = full.argmax(dim=1)
    full_norm = torch.linalg.vector_norm(full, dim=1)
    recon_norm = torch.linalg.vector_norm(reconstruction, dim=1)
    relative_error = torch.linalg.vector_norm(full - reconstruction, dim=1) / (
        full_norm + EPS
    )
    cosine = (reconstruction * full).sum(dim=1) / (recon_norm * full_norm + EPS)
    agreement = (reconstruction.argmax(dim=1) == top1).to(torch.float64)
    target = reconstruction.gather(1, top1[:, None]).squeeze(1)
    top2 = reconstruction.topk(2, dim=1).values
    predicted = reconstruction.argmax(dim=1)
    other = torch.where(predicted == top1, top2[:, 1], top2[:, 0])
    margin = target - other
    full_target = full.gather(1, top1[:, None]).squeeze(1)
    full_top2 = full.topk(2, dim=1).values
    full_margin = full_target - full_top2[:, 1]
    return {
        "relative_error": relative_error,
        "cosine": cosine,
        "agreement": agreement,
        "margin": margin,
        "margin_ratio": margin / (full_margin + EPS),
    }


def _average_metric_dicts(
    metric_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not metric_dicts:
        raise ValueError("at least one metric dictionary is required")
    keys = set(metric_dicts[0])
    if any(set(item) != keys for item in metric_dicts):
        raise ValueError("baseline metric dictionaries use different keys")
    return {
        key: torch.stack([item[key].to(torch.float64) for item in metric_dicts]).mean(0)
        for key in sorted(keys)
    }


def _prefixed_grid_metrics(
    prefix: str,
    metrics: Mapping[str, torch.Tensor],
    name_map: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.{name_map[name]}": value.squeeze(0).cpu()
        for name, value in metrics.items()
        if name in name_map
    }


def _per_image_list(metrics: Mapping[str, torch.Tensor], name: str) -> list[float]:
    value = metrics[name].squeeze(0).detach().cpu().to(torch.float64)
    return [float(item) for item in value.tolist()]


def _run_one_image(
    *,
    model: torch.nn.Module,
    gated_blocks: Sequence[torch.nn.Module],
    x: torch.Tensor,
    sample_id: str,
    mask_chunk: int,
    k_chunk: int,
    reporting_k: Sequence[int],
    orders: torch.Tensor,
    baseline_seeds: Sequence[int],
    global_seed: int,
    baseline_mode: str,
    compute_fp32_sensitivity: bool = False,
    progress_callback: Any = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any], dict[str, float]]:
    device = x.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings: dict[str, float] = {}

    start = time.perf_counter()
    h_masks = evaluate_all_masks(
        model, gated_blocks, x, mask_chunk=mask_chunk,
        progress_callback=progress_callback,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["mask_seconds"] = time.perf_counter() - start

    h_full = h_masks[:, -1, :].to(torch.float64).clone()
    # The all-open mask is evaluated as one member of a large mask batch.
    # Compare it to a true B=1 all-open forward so batch/chunk execution cannot
    # silently change the reference endpoint.
    with torch.inference_mode():
        h_full_single = model(x).to(torch.float64)
    chunk_all_open_difference = (h_full - h_full_single).abs()
    chunk_all_open_max_abs_diff = float(chunk_all_open_difference.max().item())
    chunk_all_open_mean_abs_diff = float(chunk_all_open_difference.mean().item())
    chunk_all_open_top1_equal = bool(
        torch.equal(h_full.argmax(dim=1), h_full_single.argmax(dim=1))
    )
    del h_full_single, chunk_all_open_difference
    v_top1 = h_full.argmax(dim=1)
    deltas_h_fp32 = (
        mobius_coefficients(h_masks, dtype=torch.float32)
        if compute_fp32_sensitivity
        else None
    )
    start = time.perf_counter()
    deltas_h = mobius_coefficients(h_masks, dtype=torch.float64)
    del h_masks
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["mobius_seconds"] = time.perf_counter() - start

    full_reconstructed = full_reconstruction(deltas_h)
    full_error = torch.linalg.vector_norm(full_reconstructed - h_full, dim=1) / (
        torch.linalg.vector_norm(h_full, dim=1) + EPS
    )
    full_cosine = torch.nn.functional.cosine_similarity(
        full_reconstructed, h_full, dim=1
    )
    full_agreement = (full_reconstructed.argmax(1) == v_top1).to(torch.float64)

    gather_index = v_top1[:, None, None].expand(-1, NUM_MASKS, 1)
    deltas_v = deltas_h.gather(2, gather_index).squeeze(2)
    v_full = h_full.gather(1, v_top1[:, None]).squeeze(1)

    magnitude_h = vector_magnitudes(deltas_h[:, 1:, :])
    sort_h = descending_order(magnitude_h)
    magnitude_v = scalar_magnitudes(deltas_v[:, 1:])
    sort_v = descending_order(magnitude_v)

    fp32_sensitivity: dict[str, Any] | None = None
    if deltas_h_fp32 is not None:
        fp32_full = full_reconstruction(deltas_h_fp32).to(torch.float64)
        fp32_coefficient_error = torch.linalg.vector_norm(
            deltas_h - deltas_h_fp32.to(torch.float64)
        ) / (torch.linalg.vector_norm(deltas_h) + EPS)
        fp32_full_error = torch.linalg.vector_norm(fp32_full - h_full) / (
            torch.linalg.vector_norm(h_full) + EPS
        )
        magnitude_h_fp32 = vector_magnitudes(deltas_h_fp32[:, 1:, :])
        sort_h_fp32 = descending_order(magnitude_h_fp32)
        inverse_rank_fp32 = torch.empty(M, dtype=torch.long, device=device)
        inverse_rank_fp32[sort_h_fp32[0]] = torch.arange(M, device=device)
        overlap_k = (int(math.ceil(0.01 * M)), int(math.ceil(0.10 * M)), int(math.ceil(0.50 * M)))
        rank_overlap = {
            str(k): float(
                (inverse_rank_fp32[sort_h[0, :k]] < k)
                .to(torch.float64)
                .mean()
                .item()
            )
            for k in overlap_k
        }
        fp32_result = evaluate_vector_topk(
            deltas_h_fp32,
            full_output=h_full,
            sort_idx=sort_h_fp32,
            magnitudes=magnitude_h_fp32,
            k_chunk=k_chunk,
            k_grid=reporting_k,
            orders_residual=orders,
            error_thresholds=(),
            mass_thresholds=(),
            return_curves=False,
            accumulation_dtype=torch.float32,
        )
        fp32_sensitivity = {
            "coefficient_relative_l2": float(fp32_coefficient_error.item()),
            "full_reconstruction_relative_error": float(fp32_full_error.item()),
            "full_reconstruction_max_abs_error": float(
                (fp32_full - h_full).abs().max().item()
            ),
            "max_reporting_error_absolute_difference": 0.0,
            "rank_overlap": rank_overlap,
        }
        # The grid-wise comparison is filled after the primary FP64 result is
        # evaluated below.  Keep only compact diagnostics, never FP32 curves.
        fp32_sensitivity["_fp32_result"] = fp32_result
        del fp32_full, fp32_coefficient_error, fp32_full_error
        del magnitude_h_fp32, sort_h_fp32, inverse_rank_fp32, deltas_h_fp32

    start = time.perf_counter()
    vector_result = evaluate_vector_topk(
        deltas_h,
        full_output=h_full,
        sort_idx=sort_h,
        magnitudes=magnitude_h,
        k_chunk=k_chunk,
        k_grid=reporting_k,
        orders_residual=orders,
        error_thresholds=ERROR_THRESHOLDS,
        mass_thresholds=MASS_THRESHOLDS,
        return_curves=True,
        accumulation_dtype=torch.float64,
    )
    scalar_result = evaluate_scalar_topk(
        deltas_v,
        full_output=v_full,
        sort_idx=sort_v,
        magnitudes=magnitude_v,
        k_chunk=max(k_chunk, 4096),
        k_grid=reporting_k,
        error_thresholds=(),
        mass_thresholds=MASS_THRESHOLDS,
        return_curves=True,
        accumulation_dtype=torch.float64,
    )
    if fp32_sensitivity is not None:
        fp32_result = fp32_sensitivity.pop("_fp32_result")
        fp32_sensitivity["max_reporting_error_absolute_difference"] = float(
            (
                fp32_result.at_k["err_h"].to(torch.float64)
                - vector_result.at_k["err_h"].to(torch.float64)
            )
            .abs()
            .max()
            .item()
        )
        fp32_sensitivity["full_k_stream_relative_error"] = float(
            fp32_result.at_k["err_h"][0, -1].item()
        )
        del fp32_result
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["magnitude_topk_seconds"] = time.perf_counter() - start

    baseline_h = deltas_h[:, 0, :]
    baseline_metrics = _point_vector_metrics(baseline_h, h_full)
    baseline_v_error = (v_full - deltas_v[:, 0]).abs() / (v_full.abs() + EPS)

    random_average: dict[str, torch.Tensor] | None = None
    order_matched_average: dict[str, torch.Tensor] | None = None
    low_order_result = None
    start = time.perf_counter()
    if baseline_mode != "none":
        random_runs: list[Mapping[str, torch.Tensor]] = []
        for base_seed in baseline_seeds:
            seed = _derived_seed(sample_id, "random", base_seed, global_seed)
            permutation = random_permutations(
                1, M, [seed], device=device
            )[seed]
            result = evaluate_vector_topk(
                deltas_h,
                full_output=h_full,
                sort_idx=permutation,
                magnitudes=magnitude_h,
                k_chunk=k_chunk,
                k_grid=reporting_k,
                orders_residual=orders,
                error_thresholds=(),
                mass_thresholds=(),
                return_curves=False,
                accumulation_dtype=torch.float64,
            )
            random_runs.append(result.at_k)
        random_average = _average_metric_dicts(random_runs)

    if baseline_mode == "all":
        order_runs: list[Mapping[str, torch.Tensor]] = []
        for base_seed in baseline_seeds:
            seed = _derived_seed(
                sample_id, "order_matched", base_seed, global_seed
            )
            selected = order_matched_indices(
                sort_h, orders, reporting_k, seed=seed
            )
            result = evaluate_subset_grid(
                deltas_h,
                selected,
                full_output=h_full,
                magnitudes=magnitude_h,
                accumulation_dtype=torch.float64,
            )
            order_runs.append(result.metrics)
        order_matched_average = _average_metric_dicts(order_runs)
        low_order_result = evaluate_low_order(
            deltas_h,
            orders,
            max_order=L,
            full_output=h_full,
            magnitudes=magnitude_h,
            accumulation_dtype=torch.float64,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["baselines_seconds"] = time.perf_counter() - start

    values: dict[str, torch.Tensor] = {}
    for output_name, (family, input_name) in CURVE_METRICS.items():
        result = scalar_result if family == "scalar" else vector_result
        values[output_name] = result.curves[input_name].squeeze(0)
    values.update(
        _prefixed_grid_metrics(
            "magnitude.scalar", scalar_result.at_k, SCALAR_NAME_MAP
        )
    )
    values.update(
        _prefixed_grid_metrics(
            "magnitude.vector", vector_result.at_k, VECTOR_NAME_MAP
        )
    )
    values.update(
        {
            f"baseline.vector.{name}": value.squeeze(0).cpu()
            for name, value in baseline_metrics.items()
        }
    )
    values["baseline.scalar.relative_error"] = baseline_v_error.squeeze(0).cpu()
    assert vector_result.order_composition is not None
    assert vector_result.order_enrichment is not None
    values["order.composition"] = vector_result.order_composition.squeeze(0)
    values["order.enrichment"] = vector_result.order_enrichment.squeeze(0)

    for threshold in ERROR_THRESHOLDS:
        source_key = f"err_h_le_{threshold:g}"
        values[f"keff.error.{_threshold_key(threshold)}"] = (
            vector_result.k_effective[source_key].squeeze(0).to(torch.float64)
        )
    for threshold in MASS_THRESHOLDS:
        source_key = f"mass_ge_{threshold:g}"
        values[f"keff.mass.{_threshold_key(threshold)}"] = (
            vector_result.k_effective[source_key].squeeze(0).to(torch.float64)
        )

    if random_average is not None:
        values.update(
            _prefixed_grid_metrics("random.vector", random_average, VECTOR_NAME_MAP)
        )
    if order_matched_average is not None:
        values.update(
            _prefixed_grid_metrics(
                "order_matched.vector", order_matched_average, VECTOR_NAME_MAP
            )
        )
    if low_order_result is not None:
        values.update(
            _prefixed_grid_metrics(
                "low_order.vector", low_order_result.metrics, VECTOR_NAME_MAP
            )
        )

    for name, value in timings.items():
        values[f"runtime.{name}"] = torch.tensor(value, dtype=torch.float64)
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    values["runtime.peak_vram_bytes"] = torch.tensor(float(peak_bytes))

    gamma_h = vector_result.curves["captured_mass"][0]
    gamma_v = scalar_result.curves["captured_mass"][0]
    stream_full_k = {
        "vector_relative_error": float(vector_result.at_k["err_h"][0, -1].item()),
        "vector_cosine": float(vector_result.at_k["cosine"][0, -1].item()),
        "vector_agreement": float(vector_result.at_k["agreement"][0, -1].item()),
        "vector_captured_mass": float(
            vector_result.at_k["captured_mass"][0, -1].item()
        ),
        "scalar_relative_error": float(scalar_result.at_k["err_v"][0, -1].item()),
        "scalar_captured_mass": float(
            scalar_result.at_k["captured_mass"][0, -1].item()
        ),
    }

    baseline_full_k: dict[str, dict[str, float]] = {}
    for family, metric_values in (
        ("random", random_average),
        ("order_matched", order_matched_average),
        (
            "low_order",
            None if low_order_result is None else low_order_result.metrics,
        ),
    ):
        if metric_values is None:
            continue
        baseline_full_k[family] = {
            "relative_error": float(metric_values["err_h"][0, -1].item()),
            "cosine": float(metric_values["cosine"][0, -1].item()),
            "agreement": float(metric_values["agreement"][0, -1].item()),
            "captured_mass": float(
                metric_values["captured_mass"][0, -1].item()
            ),
        }

    stream_full_k_passed = (
        stream_full_k["vector_relative_error"] <= 1e-6
        and stream_full_k["vector_cosine"] >= 0.999999
        and stream_full_k["vector_agreement"] == 1.0
        and abs(stream_full_k["vector_captured_mass"] - 1.0) <= 1e-6
        and stream_full_k["scalar_relative_error"] <= 1e-6
        and abs(stream_full_k["scalar_captured_mass"] - 1.0) <= 1e-6
    )
    baseline_full_k_passed = all(
        endpoint["relative_error"] <= 1e-6
        and endpoint["cosine"] >= 0.999999
        and endpoint["agreement"] == 1.0
        and abs(endpoint["captured_mass"] - 1.0) <= 1e-6
        for endpoint in baseline_full_k.values()
    )
    finite = all(torch.isfinite(value).all().item() for value in values.values())
    verification = {
        "sample_id": sample_id,
        "full_reconstruction_relative_error": float(full_error.item()),
        "full_reconstruction_cosine": float(full_cosine.item()),
        "full_reconstruction_agreement": float(full_agreement.item()),
        "gamma_h_final": float(gamma_h[-1].item()),
        "gamma_v_final": float(gamma_v[-1].item()),
        "gamma_h_monotonic": bool((gamma_h[1:] >= gamma_h[:-1] - 1e-12).all()),
        "gamma_v_monotonic": bool((gamma_v[1:] >= gamma_v[:-1] - 1e-12).all()),
        "stream_full_k": stream_full_k,
        "stream_full_k_passed": bool(stream_full_k_passed),
        "baseline_full_k": baseline_full_k,
        "baseline_full_k_passed": bool(baseline_full_k_passed),
        "all_metrics_finite": bool(finite),
        "full_top1": int(v_top1.item()),
        "chunk_all_open_max_abs_diff": chunk_all_open_max_abs_diff,
        "chunk_all_open_mean_abs_diff": chunk_all_open_mean_abs_diff,
        "chunk_all_open_top1_equal": chunk_all_open_top1_equal,
        "peak_vram_bytes": peak_bytes,
    }
    if not (
        verification["full_reconstruction_relative_error"] <= 1e-4
        and verification["full_reconstruction_cosine"] >= 0.999999
        and verification["full_reconstruction_agreement"] == 1.0
        and abs(verification["gamma_h_final"] - 1.0) <= 1e-6
        and abs(verification["gamma_v_final"] - 1.0) <= 1e-6
        and verification["gamma_h_monotonic"]
        and verification["gamma_v_monotonic"]
        and verification["stream_full_k_passed"]
        and verification["baseline_full_k_passed"]
        and verification["chunk_all_open_max_abs_diff"] <= 1e-4
        and verification["chunk_all_open_top1_equal"]
        and verification["all_metrics_finite"]
    ):
        raise RuntimeError(f"per-image verification failed for {sample_id}: {verification}")

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "status": "success",
        "reporting_k": list(reporting_k),
        "low_order_r": list(range(1, L + 1)),
        "low_order_k": list(low_order_sizes()),
        "full_top1": int(v_top1.item()),
        "full_logit_norm": float(torch.linalg.vector_norm(h_full).item()),
        "full_reconstruction_relative_error": verification[
            "full_reconstruction_relative_error"
        ],
        "full_reconstruction_cosine": verification["full_reconstruction_cosine"],
        "full_reconstruction_agreement": verification["full_reconstruction_agreement"],
        "peak_vram_bytes": peak_bytes,
        "sensitivity__fp32__performed": fp32_sensitivity is not None,
        "sensitivity__fp32__coefficient_relative_l2": None,
        "sensitivity__fp32__full_reconstruction_relative_error": None,
        "sensitivity__fp32__full_reconstruction_max_abs_error": None,
        "sensitivity__fp32__max_reporting_error_absolute_difference": None,
        "sensitivity__fp32__full_k_stream_relative_error": None,
        "sensitivity__fp32__rank_overlap_k_1pct": None,
        "sensitivity__fp32__rank_overlap_k_10pct": None,
        "sensitivity__fp32__rank_overlap_k_50pct": None,
    }
    if fp32_sensitivity is not None:
        row.update(
            {
                "sensitivity__fp32__coefficient_relative_l2": fp32_sensitivity[
                    "coefficient_relative_l2"
                ],
                "sensitivity__fp32__full_reconstruction_relative_error": fp32_sensitivity[
                    "full_reconstruction_relative_error"
                ],
                "sensitivity__fp32__full_reconstruction_max_abs_error": fp32_sensitivity[
                    "full_reconstruction_max_abs_error"
                ],
                "sensitivity__fp32__max_reporting_error_absolute_difference": fp32_sensitivity[
                    "max_reporting_error_absolute_difference"
                ],
                "sensitivity__fp32__full_k_stream_relative_error": fp32_sensitivity[
                    "full_k_stream_relative_error"
                ],
                "sensitivity__fp32__rank_overlap_k_1pct": fp32_sensitivity[
                    "rank_overlap"
                ][str(int(math.ceil(0.01 * M)))],
                "sensitivity__fp32__rank_overlap_k_10pct": fp32_sensitivity[
                    "rank_overlap"
                ][str(int(math.ceil(0.10 * M)))],
                "sensitivity__fp32__rank_overlap_k_50pct": fp32_sensitivity[
                    "rank_overlap"
                ][str(int(math.ceil(0.50 * M)))],
            }
        )
    for source_name, output_name in SCALAR_NAME_MAP.items():
        row[f"magnitude__scalar__{output_name}"] = _per_image_list(
            scalar_result.at_k, source_name
        )
    for source_name, output_name in VECTOR_NAME_MAP.items():
        row[f"magnitude__vector__{output_name}"] = _per_image_list(
            vector_result.at_k, source_name
        )
    for threshold in ERROR_THRESHOLDS:
        key = _threshold_key(threshold)
        row[f"keff__error__{key}"] = int(values[f"keff.error.{key}"].item())
    for threshold in MASS_THRESHOLDS:
        key = _threshold_key(threshold)
        row[f"keff__mass__{key}"] = int(values[f"keff.mass.{key}"].item())
    row["order__composition"] = vector_result.order_composition.squeeze(0).tolist()
    row["order__enrichment"] = vector_result.order_enrichment.squeeze(0).tolist()
    if random_average is not None:
        for source_name, output_name in VECTOR_NAME_MAP.items():
            if source_name in random_average:
                row[f"random__vector__{output_name}"] = _per_image_list(
                    random_average, source_name
                )
    if order_matched_average is not None:
        for source_name, output_name in VECTOR_NAME_MAP.items():
            if source_name in order_matched_average:
                row[f"order_matched__vector__{output_name}"] = _per_image_list(
                    order_matched_average, source_name
                )
    if low_order_result is not None:
        for source_name, output_name in VECTOR_NAME_MAP.items():
            if source_name in low_order_result.metrics:
                row[f"low_order__vector__{output_name}"] = _per_image_list(
                    low_order_result.metrics, source_name
                )
    row.update({f"runtime__{name}": value for name, value in timings.items()})

    del deltas_h, deltas_v, h_full, full_reconstructed
    return values, row, verification, timings


def _checkpoint_state(
    *,
    config_fingerprint: str,
    stats: OnlineStats,
    distributions: Mapping[str, OnlineScalarDistribution],
    rows: Sequence[Mapping[str, Any]],
    verification_rows: Sequence[Mapping[str, Any]],
    model_verification: Mapping[str, Any] | None,
    resume_events: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint,
        "stats": stats.state_dict(),
        "distributions": {
            name: distribution.state_dict()
            for name, distribution in distributions.items()
        },
        "rows": list(rows),
        "verification_rows": list(verification_rows),
        "model_verification": dict(model_verification or {}),
        "resume_events": [dict(event) for event in resume_events],
        "elapsed_seconds": float(elapsed_seconds),
        "saved_at": _utc_now(),
    }


def _append_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tensor_json(value: torch.Tensor) -> Any:
    value = value.detach().cpu()
    return float(value.item()) if value.ndim == 0 else value.tolist()


def _moments_json(finalized: Any) -> dict[str, Any]:
    return {
        "count": finalized.count,
        "mean": _tensor_json(finalized.mean),
        "std": _tensor_json(finalized.std),
        "se": _tensor_json(finalized.se),
        "ci_low": _tensor_json(finalized.ci_low),
        "ci_high": _tensor_json(finalized.ci_high),
    }


def _scalar_summary(values: Sequence[float]) -> dict[str, Any]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        raise ValueError("cannot summarize an empty scalar sequence")
    mean = tensor.mean()
    if tensor.numel() == 1:
        std = torch.tensor(float("nan"), dtype=torch.float64)
        se = std.clone()
    else:
        std = tensor.std(unbiased=True)
        se = std / math.sqrt(tensor.numel())
    return {
        "count": int(tensor.numel()),
        "mean": float(mean.item()),
        "std": float(std.item()),
        "se": float(se.item()),
        "ci_low": float((mean - 1.959963984540054 * se).item()),
        "ci_high": float((mean + 1.959963984540054 * se).item()),
    }


def _set_nested(root: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_save_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verification_summary(
    model_verification: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    stats_count: int,
    expected_count: int,
    resume_events: Sequence[Mapping[str, Any]],
    require_resume: bool,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot summarize zero verification rows")
    errors = [float(row["full_reconstruction_relative_error"]) for row in rows]
    cosines = [float(row["full_reconstruction_cosine"]) for row in rows]
    agreements = [float(row["full_reconstruction_agreement"]) for row in rows]
    chunk_differences = [float(row["chunk_all_open_max_abs_diff"]) for row in rows]
    peak = max(int(row["peak_vram_bytes"]) for row in rows)
    model_all_open_passed = bool(
        model_verification.get("performed")
        and model_verification.get("parameter_keys_match")
        and model_verification.get("parameters_equal")
        and model_verification.get("allclose_rtol_1e-5_atol_1e-6")
        and model_verification.get("top1_equal")
        and float(model_verification.get("max_abs_logit_diff", math.inf)) <= 1e-5
    )
    unique_ids = len({str(row["sample_id"]) for row in rows}) == len(rows)
    resume_check = {
        "required_for_this_split": bool(require_resume),
        "exercised": bool(resume_events),
        "events": [dict(event) for event in resume_events],
        "no_duplicate_sample_ids": unique_ids,
        "stats_count_matches_rows": stats_count == len(rows),
        "complete_count_matches_selection": stats_count == expected_count,
    }
    resume_check["passed"] = bool(
        unique_ids
        and stats_count == len(rows)
        and stats_count == expected_count
        and (bool(resume_events) or not require_resume)
        and all(
            bool(event.get("checkpoint_fingerprint_match"))
            and bool(event.get("row_count_matches_stats"))
            and bool(event.get("verification_count_matches_stats"))
            and bool(event.get("unique_completed_sample_ids"))
            for event in resume_events
        )
    )
    block_count_check = len(GATE_NAMES) == 16
    gate_order_check = (
        len(GATE_NAMES) == 16
        and GATE_NAMES[0] == "layer1.0"
        and GATE_NAMES[-1] == "layer4.2"
    )
    mask_count_check = NUM_MASKS == 65536
    all_stream_full_k_passed = all(
        bool(row.get("stream_full_k_passed")) for row in rows
    )
    all_baseline_full_k_passed = all(
        bool(row.get("baseline_full_k_passed")) for row in rows
    )
    stream_endpoints = [row["stream_full_k"] for row in rows]
    stream_full_k_summary = {
        "max_vector_relative_error": max(
            float(endpoint["vector_relative_error"]) for endpoint in stream_endpoints
        ),
        "min_vector_cosine": min(
            float(endpoint["vector_cosine"]) for endpoint in stream_endpoints
        ),
        "all_vector_agreement": all(
            float(endpoint["vector_agreement"]) == 1.0
            for endpoint in stream_endpoints
        ),
        "max_abs_vector_mass_minus_one": max(
            abs(float(endpoint["vector_captured_mass"]) - 1.0)
            for endpoint in stream_endpoints
        ),
        "max_scalar_relative_error": max(
            float(endpoint["scalar_relative_error"]) for endpoint in stream_endpoints
        ),
        "max_abs_scalar_mass_minus_one": max(
            abs(float(endpoint["scalar_captured_mass"]) - 1.0)
            for endpoint in stream_endpoints
        ),
        "passed": all_stream_full_k_passed,
    }
    baseline_families = sorted(
        {
            family
            for row in rows
            for family in row.get("baseline_full_k", {}).keys()
        }
    )
    baseline_full_k_summary: dict[str, Any] = {
        "passed": all_baseline_full_k_passed,
        "families": {},
    }
    for family in baseline_families:
        endpoints = [row["baseline_full_k"][family] for row in rows]
        baseline_full_k_summary["families"][family] = {
            "max_relative_error": max(
                float(endpoint["relative_error"]) for endpoint in endpoints
            ),
            "min_cosine": min(float(endpoint["cosine"]) for endpoint in endpoints),
            "all_agreement": all(
                float(endpoint["agreement"]) == 1.0 for endpoint in endpoints
            ),
            "max_abs_mass_minus_one": max(
                abs(float(endpoint["captured_mass"]) - 1.0)
                for endpoint in endpoints
            ),
        }
    max_gamma_h_error = max(
        abs(float(row["gamma_h_final"]) - 1.0) for row in rows
    )
    max_gamma_v_error = max(
        abs(float(row["gamma_v_final"]) - 1.0) for row in rows
    )
    summary = {
        "model_all_open": dict(model_verification),
        "model_all_open_passed": model_all_open_passed,
        "block_count_check": block_count_check,
        "stage_counts": [3, 4, 6, 3],
        "gate_order_check": gate_order_check,
        "mask_count": NUM_MASKS,
        "mask_count_check": mask_count_check,
        "residual_term_count": M,
        "per_order_subset_counts": [math.comb(L, order) for order in range(L + 1)],
        "max_full_reconstruction_relative_error": max(errors),
        "min_full_reconstruction_cosine": min(cosines),
        "all_full_reconstruction_agreement": all(value == 1.0 for value in agreements),
        "max_chunk_all_open_abs_diff": max(chunk_differences),
        "all_chunk_all_open_top1_equal": all(
            bool(row["chunk_all_open_top1_equal"]) for row in rows
        ),
        "all_gamma_h_monotonic": all(bool(row["gamma_h_monotonic"]) for row in rows),
        "all_gamma_v_monotonic": all(bool(row["gamma_v_monotonic"]) for row in rows),
        "all_stream_full_k_passed": all_stream_full_k_passed,
        "all_baseline_full_k_passed": all_baseline_full_k_passed,
        "stream_full_k": stream_full_k_summary,
        "baseline_full_k": baseline_full_k_summary,
        "all_metrics_finite": all(bool(row["all_metrics_finite"]) for row in rows),
        "max_abs_gamma_h_final_minus_one": max_gamma_h_error,
        "max_abs_gamma_v_final_minus_one": max_gamma_v_error,
        "peak_vram_bytes": peak,
        "num_verified_images": len(rows),
        "resume_check": resume_check,
        # Stable aliases matching the experiment-plan result contract.
        "all_open_max_abs_diff": float(
            model_verification.get("max_abs_logit_diff", math.inf)
        ),
        "all_open_top1_agreement": bool(model_verification.get("top1_equal")),
        "full_reconstruction_error": max(errors),
        "full_reconstruction_cosine": min(cosines),
        "full_reconstruction_agreement": all(value == 1.0 for value in agreements),
        "gamma_monotonic_check": all(
            bool(row["gamma_h_monotonic"]) and bool(row["gamma_v_monotonic"])
            for row in rows
        ),
        "nan_inf_check": all(bool(row["all_metrics_finite"]) for row in rows),
    }
    summary["passed"] = bool(
            model_all_open_passed
            and block_count_check
            and gate_order_check
            and mask_count_check
            and max(errors) <= 1e-4
            and min(cosines) >= 0.999999
            and all(value == 1.0 for value in agreements)
            and max(chunk_differences) <= 1e-4
            and all(bool(row["chunk_all_open_top1_equal"]) for row in rows)
            and all(bool(row["gamma_h_monotonic"]) for row in rows)
            and all(bool(row["gamma_v_monotonic"]) for row in rows)
            and max_gamma_h_error <= 1e-6
            and max_gamma_v_error <= 1e-6
            and all_stream_full_k_passed
            and all_baseline_full_k_passed
            and all(bool(row["all_metrics_finite"]) for row in rows)
            and resume_check["passed"]
    )
    return summary


def _finalize_outputs(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    stats: OnlineStats,
    distributions: Mapping[str, OnlineScalarDistribution],
    rows: Sequence[Mapping[str, Any]],
    verification_rows: Sequence[Mapping[str, Any]],
    model_verification: Mapping[str, Any],
    started_at: str,
    elapsed_seconds: float,
    resume_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    finalized = stats.finalize(0.95)
    metrics: dict[str, Any] = {}
    runtime: dict[str, Any] = {
        "started_at": started_at,
        "ended_at": _utc_now(),
        "wall_seconds": elapsed_seconds,
        "num_images": stats.count,
    }
    curve_arrays: dict[str, np.ndarray] = {
        "k": np.arange(1, M + 1, dtype=np.int64),
        "reporting_k": np.asarray(make_reporting_k_grid(), dtype=np.int64),
        "low_order_r": np.arange(1, L + 1, dtype=np.int64),
        "low_order_k": np.asarray(low_order_sizes(), dtype=np.int64),
        "interaction_order": np.arange(0, L + 1, dtype=np.int64),
    }
    metric_shapes: dict[str, list[int]] = {}
    for name, value in finalized.items():
        serialized = _moments_json(value)
        metric_shapes[name] = list(value.mean.shape)
        if name.startswith("curve."):
            base = name[len("curve."):]
            for statistic in ("mean", "se", "ci_low", "ci_high"):
                array = getattr(value, statistic).detach().cpu().numpy()
                curve_arrays[f"{base}.{statistic}"] = array
        elif name.startswith("runtime."):
            _set_nested(runtime, name[len("runtime."):], serialized)
        else:
            _set_nested(metrics, name, serialized)

    for name, distribution in distributions.items():
        summary = distribution.finalize()
        target = metrics
        parts = name.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        existing = target.setdefault(parts[-1], {})
        existing.update(
            {
                "median": summary.median,
                "q25": summary.q25,
                "q75": summary.q75,
                "iqr": summary.iqr,
            }
        )

    mask_total = stats.count * NUM_MASKS
    mask_seconds = sum(float(row["runtime__mask_seconds"]) for row in rows)
    runtime["total_mask_image_forwards"] = mask_total
    runtime["aggregate_mask_seconds"] = mask_seconds
    runtime["mask_images_per_second"] = mask_total / max(mask_seconds, EPS)
    runtime["peak_vram_bytes"] = max(int(row["peak_vram_bytes"]) for row in rows)

    grids = {
        "absolute_k": list(ABSOLUTE_K_GRID),
        "p": list(P_GRID),
        "p_to_k": {
            f"{100 * proportion:g}%": int(math.ceil(proportion * M))
            for proportion in P_GRID
        },
        "reporting_k": list(make_reporting_k_grid()),
        "low_order_r": list(range(1, L + 1)),
        "low_order_k": list(low_order_sizes()),
        "error_thresholds": list(ERROR_THRESHOLDS),
        "mass_thresholds": list(MASS_THRESHOLDS),
        "orders": list(range(L + 1)),
    }
    verification = _verification_summary(
        model_verification,
        verification_rows,
        stats_count=stats.count,
        expected_count=int(config["num_images"]),
        resume_events=resume_events,
        require_resume=str(config.get("split")) == "smoke",
    )
    if not verification["passed"]:
        raise RuntimeError(f"aggregate verification failed: {verification}")

    sensitivity_rows = [
        row for row in rows if bool(row.get("sensitivity__fp32__performed"))
    ]
    sensitivity: dict[str, Any] = {
        "fp32_transform": {
            "performed": bool(sensitivity_rows),
            "num_images": len(sensitivity_rows),
            "selection_positions": [
                int(row["selection_position"]) for row in sensitivity_rows
            ],
            "diagnostic_only": True,
        }
    }
    if sensitivity_rows:
        prefix = "sensitivity__fp32__"
        for field in (
            "coefficient_relative_l2",
            "full_reconstruction_relative_error",
            "full_reconstruction_max_abs_error",
            "max_reporting_error_absolute_difference",
            "full_k_stream_relative_error",
            "rank_overlap_k_1pct",
            "rank_overlap_k_10pct",
            "rank_overlap_k_50pct",
        ):
            sensitivity["fp32_transform"][field] = _scalar_summary(
                [float(row[prefix + field]) for row in sensitivity_rows]
            )

    final_config = dict(config)
    final_config.update(
        {
            "started_at": started_at,
            "start_time": started_at,
            "ended_at": runtime["ended_at"],
            "end_time": runtime["ended_at"],
            "wall_seconds": float(elapsed_seconds),
        }
    )

    curves_path = output_dir / "curves.npz"
    per_image_path = output_dir / "per_image_summary.parquet"
    _atomic_save_npz(curves_path, curve_arrays)
    _atomic_save_parquet(per_image_path, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_images": stats.count,
        "config": final_config,
        "verification": verification,
        "grids": grids,
        "metric_axes": {
            "curves.npz:<metric>.<statistic>": ["k"],
            "metrics.magnitude.*.<metric>": ["reporting_k"],
            "metrics.random.vector.<metric>": ["reporting_k"],
            "metrics.order_matched.vector.<metric>": ["reporting_k"],
            "metrics.low_order.vector.<metric>": ["low_order_r"],
            "metrics.order.composition": ["reporting_k", "interaction_order"],
            "metrics.order.enrichment": ["reporting_k", "interaction_order"],
            "per_image_summary.parquet": ["image"],
        },
        "metric_shapes": metric_shapes,
        "npz_shapes": {
            name: list(array.shape) for name, array in curve_arrays.items()
        },
        "metrics": metrics,
        "runtime": runtime,
        "sensitivity": sensitivity,
        "artifacts": {
            "curves": curves_path.name,
            "per_image_summary": per_image_path.name,
            "config": "config.json",
            "checkpoint": "checkpoint.pt",
            "progress": "progress.jsonl",
            "run_log": "run.log",
            "table": "table_resnet34.md",
            "report": "REPORT.md",
            "checksums": "SHA256SUMS",
        },
    }
    atomic_write_json(summary, output_dir / "summary.json")
    atomic_write_json(final_config, output_dir / "config.json")
    atomic_write_json(verification, output_dir / "verification.json")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    progress_path = args.output_dir / "progress.jsonl"
    configure_determinism(args.seed, cudnn_benchmark=args.cudnn_benchmark)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    loaded_manifest = load_permutation_manifest(
        args.selection_manifest,
        sample_index_path=args.sample_index,
        verify_source=True,
    )
    manifest_sha256 = sha256_file(args.selection_manifest)
    _validate_official_run_contract(
        args,
        manifest=loaded_manifest,
        manifest_sha256=manifest_sha256,
        source_sha256=loaded_manifest.source_sha256,
    )
    logical_records = select_records(
        loaded_manifest, split=args.split, sort_for_io=False
    )
    processing_records = select_records(
        loaded_manifest, split=args.split, sort_for_io=True
    )
    processing_records = _relocate_records(processing_records, args.parquet_dir)
    static_config = _static_config(
        args,
        logical_records=logical_records,
        processing_records=processing_records,
        manifest_sha256=manifest_sha256,
        source_sha256=loaded_manifest.source_sha256,
        repo_root=repo_root,
        device=device,
    )
    config_fingerprint = _config_fingerprint(static_config)
    started_at = _utc_now()
    config = {**static_config, "config_fingerprint": config_fingerprint, "started_at": started_at}

    stats = OnlineStats(device="cpu")
    distributions = {
        f"keff.error.{_threshold_key(value)}": OnlineScalarDistribution()
        for value in ERROR_THRESHOLDS
    }
    distributions.update(
        {
            f"keff.mass.{_threshold_key(value)}": OnlineScalarDistribution()
            for value in MASS_THRESHOLDS
        }
    )
    rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    model_verification: dict[str, Any] = {}
    resume_events: list[dict[str, Any]] = []
    prior_elapsed = 0.0

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume requested but {checkpoint_path} does not exist")
        checkpoint = load_torch_checkpoint(checkpoint_path)
        fingerprint_matches = checkpoint.get("config_fingerprint") == config_fingerprint
        if not fingerprint_matches:
            raise ValueError("checkpoint configuration does not match this run")
        stats = OnlineStats.from_state_dict(checkpoint["stats"], device="cpu")
        distributions = {
            name: OnlineScalarDistribution.from_state_dict(state)
            for name, state in checkpoint["distributions"].items()
        }
        rows = list(checkpoint.get("rows", []))
        verification_rows = list(checkpoint.get("verification_rows", []))
        model_verification = dict(checkpoint.get("model_verification", {}))
        resume_events = [dict(event) for event in checkpoint.get("resume_events", [])]
        prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        started_at = str(checkpoint.get("started_at", config.get("started_at", started_at)))
        completed_ids = [str(row.get("sample_id")) for row in rows]
        row_count_matches = len(rows) == stats.count
        verification_count_matches = len(verification_rows) == stats.count
        unique_completed_ids = (
            len(completed_ids) == len(set(completed_ids)) == stats.count
            and set(completed_ids) == stats.seen_sample_ids
        )
        selected_ids = {record.sample_id for record in processing_records}
        if not (
            row_count_matches
            and verification_count_matches
            and unique_completed_ids
            and stats.seen_sample_ids.issubset(selected_ids)
            and stats.count <= len(processing_records)
        ):
            raise ValueError("checkpoint rows/sample IDs are inconsistent with this run")
        resume_events.append(
            {
                "timestamp": _utc_now(),
                "restored_count": stats.count,
                "expected_count": len(processing_records),
                "checkpoint_fingerprint_match": bool(fingerprint_matches),
                "row_count_matches_stats": row_count_matches,
                "verification_count_matches_stats": verification_count_matches,
                "unique_completed_sample_ids": unique_completed_ids,
            }
        )
        print(f"[resume] restored {stats.count}/{len(processing_records)} images")
    elif checkpoint_path.exists() or (args.output_dir / "summary.json").exists():
        raise FileExistsError(
            f"output {args.output_dir} already contains a run; use --resume or a new directory"
        )

    config = {**config, "started_at": started_at}
    atomic_write_json(config, args.output_dir / "config.json")
    remaining_records = [
        record for record in processing_records if not stats.has_sample(record.sample_id)
    ]
    dataset = ManifestParquetDataset(remaining_records)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    record_by_id = {record.sample_id: record for record in processing_records}

    model, gated_blocks, gate_names = build_gated_resnet34()
    if tuple(gate_names) != GATE_NAMES:
        raise RuntimeError("builder returned an unexpected gate order")
    model = model.to(device).eval()
    # This is a hard gate for every invocation, including resumes.  Always use
    # the first frozen processing record so the saved and resumed checks are
    # directly comparable.
    verification_dataset = ManifestParquetDataset([processing_records[0]])
    verification_x = verification_dataset[0][0].unsqueeze(0).to(device)
    current_model_verification = _verify_reference_model(model, verification_x, device)
    if model_verification:
        stable_keys = (
            "parameter_keys_match",
            "parameters_equal",
            "max_abs_logit_diff",
            "mean_abs_logit_diff",
            "allclose_rtol_1e-5_atol_1e-6",
            "top1_equal",
            "reference_top1",
            "gated_top1",
        )
        if any(
            current_model_verification.get(key) != model_verification.get(key)
            for key in stable_keys
        ):
            raise RuntimeError("resumed model verification differs from checkpoint")
        current_model_verification["resume_recheck_equal"] = True
    model_verification = current_model_verification
    del verification_x, verification_dataset
    print(
        "[verify] all-open max_abs_diff="
        f"{model_verification['max_abs_logit_diff']:.3g}, top1_equal=True"
    )
    orders = residual_orders(L, device="cpu")
    reporting_k = make_reporting_k_grid()
    baseline_seeds = DEFAULT_BASELINE_SEEDS[: args.num_random_seeds]
    run_start = time.perf_counter()
    iterator = iter(loader)
    local_done = 0

    while True:
        data_start = time.perf_counter()
        try:
            x, labels, master_indices, sample_ids = next(iterator)
        except StopIteration:
            break
        data_seconds = time.perf_counter() - data_start
        sample_id = str(sample_ids[0])
        record = record_by_id[sample_id]
        x = x.to(device, non_blocking=True)

        image_start = time.perf_counter()
        last_mask_log = [0]

        def mask_progress(done: int, total: int) -> None:
            step = max(args.mask_chunk * 16, 8192)
            if done == total or done - last_mask_log[0] >= step:
                print(
                    f"[mask] image={stats.count + 1}/{len(processing_records)} "
                    f"sample={sample_id} {done}/{total}",
                    flush=True,
                )
                last_mask_log[0] = done

        try:
            values, row, verification, timings = _run_one_image(
                model=model,
                gated_blocks=gated_blocks,
                x=x,
                sample_id=sample_id,
                mask_chunk=args.mask_chunk,
                k_chunk=args.k_chunk,
                reporting_k=reporting_k,
                orders=orders,
                baseline_seeds=baseline_seeds,
                global_seed=args.seed,
                baseline_mode=args.baseline_mode,
                compute_fp32_sensitivity=(
                    args.split == "pilot" and record.selection_position < 5
                ),
                progress_callback=mask_progress,
            )
        except Exception as error:
            _append_progress(
                progress_path,
                {
                    "timestamp": _utc_now(),
                    "sample_id": sample_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise

        timings["data_seconds"] = data_seconds
        timings["total_seconds"] = time.perf_counter() - image_start + data_seconds
        values["runtime.data_seconds"] = torch.tensor(data_seconds, dtype=torch.float64)
        values["runtime.total_seconds"] = torch.tensor(
            timings["total_seconds"], dtype=torch.float64
        )
        row["master_index"] = int(master_indices.item())
        row["selection_position"] = record.selection_position
        row["shard"] = record.shard
        row["row"] = record.row
        row["label"] = int(labels.item())
        row["runtime__data_seconds"] = data_seconds
        row["runtime__total_seconds"] = timings["total_seconds"]

        stats.update(values, sample_id=sample_id)
        for name, distribution in distributions.items():
            distribution.update(values[name])
        rows.append(row)
        verification_rows.append(verification)
        local_done += 1

        elapsed = prior_elapsed + (time.perf_counter() - run_start)
        pause_requested = bool(
            args.stop_after_images is not None
            and local_done >= args.stop_after_images
            and stats.count < len(processing_records)
        )
        should_checkpoint = (
            stats.count % args.checkpoint_every == 0
            or stats.count == len(processing_records)
            or pause_requested
        )
        if should_checkpoint:
            state = _checkpoint_state(
                config_fingerprint=config_fingerprint,
                stats=stats,
                distributions=distributions,
                rows=rows,
                verification_rows=verification_rows,
                model_verification=model_verification,
                resume_events=resume_events,
                elapsed_seconds=elapsed,
            )
            state["started_at"] = started_at
            atomic_save_checkpoint(state, checkpoint_path)
        _append_progress(
            progress_path,
            {
                "timestamp": _utc_now(),
                "sample_id": sample_id,
                "master_index": record.master_index,
                "selection_position": record.selection_position,
                "status": "success",
                "completed": stats.count,
                "total": len(processing_records),
                "checkpointed": should_checkpoint,
                "timings": timings,
                "verification": verification,
            },
        )
        if local_done % args.log_every == 0 or stats.count == len(processing_records):
            average = elapsed / max(stats.count, 1)
            eta = average * (len(processing_records) - stats.count)
            print(
                f"[done] {stats.count}/{len(processing_records)} sample={sample_id} "
                f"image={timings['total_seconds']:.1f}s avg={average:.1f}s "
                f"ETA={eta / 60:.1f}min peak={verification['peak_vram_bytes'] / 2**30:.2f}GiB",
                flush=True,
            )
        del x, values
        if pause_requested:
            print(
                f"[paused] checkpointed {stats.count}/{len(processing_records)} "
                "images; rerun with --resume",
                flush=True,
            )
            return

    total_elapsed = prior_elapsed + (time.perf_counter() - run_start)
    if stats.count != len(processing_records):
        raise RuntimeError(
            f"processed {stats.count} images, expected {len(processing_records)}"
        )
    summary = _finalize_outputs(
        output_dir=args.output_dir,
        config={**config, "started_at": started_at},
        stats=stats,
        distributions=distributions,
        rows=rows,
        verification_rows=verification_rows,
        model_verification=model_verification,
        started_at=started_at,
        elapsed_seconds=total_elapsed,
        resume_events=resume_events,
    )
    final_state = _checkpoint_state(
        config_fingerprint=config_fingerprint,
        stats=stats,
        distributions=distributions,
        rows=rows,
        verification_rows=verification_rows,
        model_verification=model_verification,
        resume_events=resume_events,
        elapsed_seconds=total_elapsed,
    )
    final_state["started_at"] = started_at
    final_state["complete"] = True
    atomic_save_checkpoint(final_state, checkpoint_path)
    print(
        f"[complete] {stats.count} images, wall={total_elapsed / 60:.1f}min, "
        f"masks/s={summary['runtime']['mask_images_per_second']:.1f}, "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
