"""Run label-aware ResNet-34 Experiments 4 and 5 in one exact mask sweep.

Experiment 4 stores compact nested Top-K rankings instead of dense signatures:
one ``N x max(K)`` index array is sufficient for every requested K.  Experiment
5 stores only its per-image scalar statistics.  Both analyses therefore share
the same 65,536-mask forward pass and FP64 Mobius coefficients.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .model import build_gated_resnet34
from .mask_evaluator import evaluate_all_masks
from .mobius import mobius_coefficients
from .streaming_topk import (
    descending_order,
    evaluate_vector_topk,
    residual_orders,
    scalar_magnitudes,
    vector_magnitudes,
)

from .common import (
    EPS,
    L,
    NUM_MASKS,
    NUM_RESIDUAL,
    atomic_save_checkpoint,
    atomic_save_parquet,
    atomic_write_json,
    build_loader,
    config_fingerprint,
    configure_determinism,
    implementation_hashes,
    load_index_selection,
    load_resume_checkpoint,
    sample_id_from_batch,
    software_hardware_metadata,
    write_result_checksums,
)
from .metrics import (
    complexity_metrics,
    difficulty_metrics,
    signature_k_grid,
)


SCHEMA_VERSION = 1
ERROR_THRESHOLDS = (0.10, 0.05)
MASS_THRESHOLDS = (0.90, 0.95, 0.99)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-index", type=Path,
        default=Path("data/indices/imagenet_val_balanced_10000.json"),
    )
    parser.add_argument("--parquet-dir", type=Path, default=Path("imagenet-1k/data"))
    parser.add_argument(
        "--per-class", type=int, default=10,
        help="balanced images per class; use 0 to evaluate the full sample index",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="unbalanced prefix selector, only allowed when --per-class=0",
    )
    parser.add_argument("--synthetic-images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-chunk", type=int, default=1024)
    parser.add_argument("--k-chunk", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low-order-cutoff", type=int, default=3)
    parser.add_argument(
        "--signature-fractions", type=float, nargs="+",
        default=(0.01, 0.03, 0.05, 0.10, 0.13, 0.25),
    )
    parser.add_argument(
        "--signature-fixed-k", type=int, nargs="*", default=(32,),
        help="absolute K values to retain in addition to the fractional grid",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-images", type=int, default=None)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--work-dir", type=Path,
        default=Path("runs/resnet34/dependence_and_difficulty"),
    )
    parser.add_argument(
        "--exp4-output-dir", type=Path,
        default=Path("experiments/input_dependent_expert_sets/generated/resnet34"),
    )
    parser.add_argument(
        "--exp5-output-dir", type=Path,
        default=Path("experiments/difficulty_interaction_complexity/generated/resnet34"),
    )
    args = parser.parse_args(argv)
    if args.per_class < 0:
        parser.error("per-class must be non-negative")
    if args.per_class and args.max_images is not None:
        parser.error("max-images is only valid with --per-class=0")
    if args.mask_chunk < 1 or args.k_chunk < 1:
        parser.error("mask/k chunks must be positive")
    if args.checkpoint_every < 1 or args.log_every < 1:
        parser.error("checkpoint/log intervals must be positive")
    if args.stop_after_images is not None and args.stop_after_images < 1:
        parser.error("stop-after-images must be positive")
    if args.low_order_cutoff != 3:
        parser.error(
            "Experiment 5 and its downstream schema define the low-order cutoff as 3"
        )
    return args


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _open_memmap(
    path: Path,
    *,
    dtype: np.dtype | str,
    shape: tuple[int, ...],
    resume: bool,
) -> np.memmap:
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"resume array missing: {path}")
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(
                f"resume array {path} is {array.shape}/{array.dtype}; "
                f"expected {shape}/{np.dtype(dtype)}"
            )
        return array
    if path.exists():
        raise FileExistsError(f"signature array already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _flush(arrays: Sequence[np.memmap]) -> None:
    for array in arrays:
        array.flush()


def _run_one(
    model: torch.nn.Module,
    gated_blocks: Sequence[torch.nn.Module],
    x: torch.Tensor,
    label: torch.Tensor,
    *,
    mask_chunk: int,
    k_chunk: int,
    max_signature_k: int,
    orders_residual: torch.Tensor,
    low_order_cutoff: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    with torch.inference_mode():
        direct_full = model(x).detach().to(torch.float64)
    h_masks = evaluate_all_masks(model, gated_blocks, x, mask_chunk=mask_chunk)
    h_full_fp32 = h_masks[:, -1, :]
    h_full = h_full_fp32.to(torch.float64)
    chunk_diff = (direct_full - h_full).abs()
    top1 = h_full.argmax(dim=-1)
    deltas = mobius_coefficients(h_masks, dtype=torch.float64)
    reconstructed = deltas.sum(dim=1)
    reconstruction_error = (
        torch.linalg.vector_norm(reconstructed - h_full, dim=-1)
        / (torch.linalg.vector_norm(h_full, dim=-1) + EPS)
    )

    residual_h = deltas[:, 1:, :]
    magnitudes_h = vector_magnitudes(residual_h)
    scalar_all = deltas.gather(
        2, top1.view(-1, 1, 1).expand(-1, NUM_MASKS, 1)
    ).squeeze(2)
    magnitudes_v = scalar_magnitudes(scalar_all[:, 1:])
    sort_h = descending_order(magnitudes_h)
    sort_v = descending_order(magnitudes_v)
    top_indices_h = sort_h[:, :max_signature_k]
    top_indices_v = sort_v[:, :max_signature_k]
    mass_weights_h = magnitudes_h.square() / (
        magnitudes_h.square().sum(dim=1, keepdim=True) + EPS
    )
    top_weights_h = torch.gather(mass_weights_h, 1, top_indices_h)

    row: dict[str, Any] = {
        # Difficulty values retain the original experiment's FP32 logits;
        # decomposition and reconstruction arithmetic remain formal FP64.
        **difficulty_metrics(h_full_fp32, label),
        **complexity_metrics(
            magnitudes_h, magnitudes_v, orders_residual,
            low_order_cutoff=low_order_cutoff,
        ),
        "labels": int(label.item()),
        "pseudo_labels": int(top1.item()),
    }
    topk_result = evaluate_vector_topk(
        deltas,
        full_output=h_full,
        sort_idx=sort_h,
        magnitudes=magnitudes_h,
        k_chunk=k_chunk,
        k_grid=(),
        orders_residual=orders_residual,
        error_thresholds=ERROR_THRESHOLDS,
        mass_thresholds=MASS_THRESHOLDS,
        return_curves=False,
        accumulation_dtype=torch.float64,
    )
    row.update(
        {
            "K_err_010": int(topk_result.k_effective["err_h_le_0.1"].item()),
            "K_err_005": int(topk_result.k_effective["err_h_le_0.05"].item()),
            "K_mass_090": int(topk_result.k_effective["mass_ge_0.9"].item()),
            "K_mass_095": int(topk_result.k_effective["mass_ge_0.95"].item()),
            "K_mass_099": int(topk_result.k_effective["mass_ge_0.99"].item()),
        }
    )
    tensors = {
        "vector_indices": top_indices_h[0].to(torch.int32).cpu(),
        "scalar_indices": top_indices_v[0].to(torch.int32).cpu(),
        "vector_weights": top_weights_h[0].to(torch.float32).cpu(),
        "full_mass_weights": mass_weights_h[0].to(torch.float64).cpu(),
    }
    finite = all(
        torch.isfinite(torch.as_tensor(value, dtype=torch.float64)).all().item()
        for value in row.values()
    ) and all(torch.isfinite(value).all().item() for value in tensors.values())
    verification = {
        "chunk_all_open_max_abs_diff": float(chunk_diff.max().item()),
        "chunk_all_open_top1_equal": bool(
            direct_full.argmax(dim=-1).eq(top1).all().item()
        ),
        "mobius_reconstruction_relative_error": float(reconstruction_error.item()),
        "mobius_top1_agreement": bool(
            reconstructed.argmax(dim=-1).eq(top1).all().item()
        ),
        "vector_mass_sum": float(mass_weights_h.sum().item()),
        "all_metrics_finite": bool(finite),
    }
    verification["passed"] = bool(
        verification["chunk_all_open_max_abs_diff"] <= 1e-4
        and verification["chunk_all_open_top1_equal"]
        and verification["mobius_reconstruction_relative_error"] <= 1e-10
        and verification["mobius_top1_agreement"]
        and abs(verification["vector_mass_sum"] - 1.0) <= 1e-10
        and verification["all_metrics_finite"]
    )
    if not verification["passed"]:
        raise RuntimeError(f"per-image verification failed: {verification}")
    del h_masks, deltas, residual_h, scalar_all, topk_result
    return row, tensors, verification


def _save_state(
    path: Path,
    *,
    fingerprint: str,
    rows: Mapping[int, Mapping[str, Any]],
    verification_rows: Mapping[int, Mapping[str, Any]],
    global_mass_sum: torch.Tensor,
    elapsed_seconds: float,
) -> None:
    atomic_save_checkpoint(
        {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": fingerprint,
            "rows": {int(key): dict(value) for key, value in rows.items()},
            "verification_rows": {
                int(key): dict(value) for key, value in verification_rows.items()
            },
            "global_mass_sum": global_mass_sum.to(torch.float64).cpu(),
            "elapsed_seconds": float(elapsed_seconds),
            "saved_at": _utc_now(),
        },
        path,
    )


def _write_npy(path: Path, values: np.ndarray) -> None:
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=values.dtype, shape=values.shape
    )
    output[...] = values
    output.flush()
    del output


def _verification_summary(
    rows: Sequence[Mapping[str, Any]], expected: int
) -> dict[str, Any]:
    result = {
        "num_verified_images": len(rows),
        "expected_images": expected,
        "all_passed": len(rows) == expected and all(bool(row["passed"]) for row in rows),
        "max_chunk_all_open_abs_diff": max(
            float(row["chunk_all_open_max_abs_diff"]) for row in rows
        ),
        "max_mobius_reconstruction_relative_error": max(
            float(row["mobius_reconstruction_relative_error"]) for row in rows
        ),
        "max_abs_vector_mass_minus_one": max(
            abs(float(row["vector_mass_sum"]) - 1.0) for row in rows
        ),
        "all_metrics_finite": all(bool(row["all_metrics_finite"]) for row in rows),
        "gate_count": L,
        "mask_count": NUM_MASKS,
    }
    result["passed"] = bool(
        result["all_passed"]
        and result["max_chunk_all_open_abs_diff"] <= 1e-4
        and result["max_mobius_reconstruction_relative_error"] <= 1e-10
        and result["max_abs_vector_mass_minus_one"] <= 1e-10
        and result["all_metrics_finite"]
    )
    return result


def _finalize(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    rows_by_slot: Mapping[int, Mapping[str, Any]],
    verification_by_slot: Mapping[int, Mapping[str, Any]],
    global_mass_sum: torch.Tensor,
    *,
    k_grid: Sequence[int],
    signature_arrays: Sequence[np.memmap],
    wall_seconds: float,
) -> None:
    ordered = [dict(rows_by_slot[index]) for index in range(len(rows_by_slot))]
    verification_rows = [
        dict(verification_by_slot[index]) for index in range(len(verification_by_slot))
    ]
    verification = _verification_summary(verification_rows, len(ordered))
    if not verification["passed"]:
        raise RuntimeError(f"final verification failed: {verification}")
    _flush(signature_arrays)

    labels = np.asarray([row["labels"] for row in ordered], dtype=np.int64)
    pseudo = np.asarray([row["pseudo_labels"] for row in ordered], dtype=np.int64)
    confidence = np.asarray([row["confidence"] for row in ordered], dtype=np.float32)
    pred_margin = np.asarray([row["pred_margin"] for row in ordered], dtype=np.float32)
    for name, values in (
        ("labels.npy", labels), ("pseudo_labels.npy", pseudo),
        ("confidence.npy", confidence), ("pred_margin.npy", pred_margin),
        (
            "global_mass_mean.npy",
            (global_mass_sum / len(ordered)).numpy().astype(np.float64),
        ),
    ):
        _write_npy(args.exp4_output_dir / name, values)

    signature_metadata = {
        "schema_version": SCHEMA_VERSION,
        "format": "nested_topk_indices_v1",
        "num_images": len(ordered),
        "num_residual_subsets": NUM_RESIDUAL,
        "K_list": list(k_grid),
        "max_K": int(max(k_grid)),
        "arrays": {
            "vector_indices": "vector_indices.npy",
            "scalar_indices": "scalar_indices.npy",
            "vector_weights": "vector_weights.npy",
            "labels": "labels.npy",
            "pseudo_labels": "pseudo_labels.npy",
            "confidence": "confidence.npy",
            "pred_margin": "pred_margin.npy",
            "global_mass_mean": "global_mass_mean.npy",
        },
        "config": {**dict(config), "wall_seconds": wall_seconds, "status": "complete"},
    }
    atomic_write_json(signature_metadata, args.exp4_output_dir / "signatures.json")
    atomic_write_json(verification, args.exp4_output_dir / "verification.json")
    atomic_save_parquet(ordered, args.exp4_output_dir / "per_image_summary.parquet")
    write_result_checksums(
        args.exp4_output_dir,
        (
            "signatures.json", "vector_indices.npy", "scalar_indices.npy",
            "vector_weights.npy", "labels.npy", "pseudo_labels.npy",
            "confidence.npy", "pred_margin.npy", "global_mass_mean.npy",
            "verification.json", "per_image_summary.parquet",
        ),
    )

    # Emit the exact key schema consumed by experiment_5.step_2_analyze.
    exp5_keys = (
        "loss", "wrong", "true_margin", "confidence", "pred_margin",
        "labels", "pseudo_labels",
        "h/N_eff", "h/N_ent", "h/kbar", "h/kappa", "h/C_le_3", "h/T_gt_3",
        "v/N_eff", "v/N_ent", "v/kbar", "v/kappa", "v/C_le_3", "v/T_gt_3",
        "K_err_010", "K_err_005", "K_mass_090", "K_mass_095", "K_mass_099",
    )
    per_image: dict[str, torch.Tensor] = {}
    integer_keys = {
        "labels", "pseudo_labels", "K_err_010", "K_err_005",
        "K_mass_090", "K_mass_095", "K_mass_099",
    }
    for key in exp5_keys:
        dtype = torch.long if key in integer_keys else torch.float64
        per_image[key] = torch.tensor([row[key] for row in ordered], dtype=dtype)
    args.exp5_output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_checkpoint(per_image, args.exp5_output_dir / "per_image.pt")
    exp5_config = {
        **dict(config),
        "num_images": len(ordered),
        "thresholds_err": list(ERROR_THRESHOLDS),
        "thresholds_mass": list(MASS_THRESHOLDS),
        "low_order_cutoff": args.low_order_cutoff,
        "wall_seconds": wall_seconds,
        "status": "complete",
    }
    atomic_write_json(exp5_config, args.exp5_output_dir / "config.json")
    atomic_write_json(verification, args.exp5_output_dir / "verification.json")
    atomic_save_parquet(ordered, args.exp5_output_dir / "per_image_summary.parquet")
    write_result_checksums(
        args.exp5_output_dir,
        ("per_image.pt", "config.json", "verification.json", "per_image_summary.parquet"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_determinism(args.seed, cudnn_benchmark=args.cudnn_benchmark)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.exp4_output_dir.mkdir(parents=True, exist_ok=True)
    args.exp5_output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.work_dir / "checkpoint.pt"
    progress_path = args.work_dir / "progress.jsonl"

    selection = None
    if args.synthetic_images is None:
        selection = load_index_selection(
            args.sample_index,
            parquet_dir=args.parquet_dir,
            per_class=(args.per_class or None),
            max_images=args.max_images,
            seed=args.seed,
        )
    loader, record_lookup, total = build_loader(
        selection,
        synthetic_images=args.synthetic_images,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    k_grid = signature_k_grid(
        NUM_RESIDUAL, args.signature_fractions, args.signature_fixed_k
    )
    max_k = max(k_grid)
    repo_root = Path(__file__).resolve().parents[1]
    numerical_files = [
        Path(__file__), Path(__file__).with_name("common.py"),
        Path(__file__).with_name("metrics.py"),
        repo_root / "resnet34/model.py",
        repo_root / "resnet34/mask_evaluator.py",
        repo_root / "resnet34/mobius.py",
        repo_root / "resnet34/streaming_topk.py",
        repo_root / "resnet18/complexity.py",
        repo_root / "resnet18/difficulty.py",
    ]
    device = torch.device(args.device)
    runtime_metadata = software_hardware_metadata(device)
    config = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "torchvision.models.resnet34",
        "weights": "ResNet34_Weights.IMAGENET1K_V1",
        "L": L,
        "num_masks": NUM_MASKS,
        "num_residual_subsets": NUM_RESIDUAL,
        "forward_dtype": "float32",
        "mobius_dtype": "float64",
        "sample_index": None if selection is None else str(args.sample_index.resolve()),
        "sample_index_sha256": None if selection is None else selection.source_sha256,
        "selector": {"synthetic_images": args.synthetic_images} if selection is None else selection.selector,
        "parquet_dir": None if selection is None else str(args.parquet_dir.resolve()),
        "signature_fractions": [float(value) for value in args.signature_fractions],
        "signature_fixed_k": [int(value) for value in args.signature_fixed_k],
        "K_list": list(k_grid),
        "low_order_cutoff": args.low_order_cutoff,
        "mask_chunk": args.mask_chunk,
        "k_chunk": args.k_chunk,
        "seed": args.seed,
        "device": args.device,
        "software_hardware": runtime_metadata,
        "cudnn_benchmark": args.cudnn_benchmark,
        "cudnn_deterministic": True,
        "tf32": False,
        "implementation_sha256": implementation_hashes(numerical_files, repo_root),
    }
    fingerprint = config_fingerprint(config)
    array_specs = (
        (args.exp4_output_dir / "vector_indices.npy", np.int32, (total, max_k)),
        (args.exp4_output_dir / "scalar_indices.npy", np.int32, (total, max_k)),
        (args.exp4_output_dir / "vector_weights.npy", np.float32, (total, max_k)),
    )
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"--resume requested but {state_path} is missing")
        state = load_resume_checkpoint(state_path, fingerprint)
        rows_by_slot = {int(key): dict(value) for key, value in state["rows"].items()}
        verification_by_slot = {
            int(key): dict(value) for key, value in state["verification_rows"].items()
        }
        global_mass_sum = state["global_mass_sum"].to(torch.float64)
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
    else:
        if state_path.exists():
            raise FileExistsError(
                f"{state_path} already exists; use --resume or a new --work-dir"
            )
        rows_by_slot: dict[int, dict[str, Any]] = {}
        verification_by_slot: dict[int, dict[str, Any]] = {}
        global_mass_sum = torch.zeros(NUM_RESIDUAL, dtype=torch.float64)
        elapsed_before = 0.0
    arrays = [
        _open_memmap(path, dtype=dtype, shape=shape, resume=args.resume)
        for path, dtype, shape in array_specs
    ]
    vector_indices, scalar_indices, vector_weights = arrays
    if not args.resume:
        # The signature arrays are materialized before the first image.  Seal
        # an empty matching state immediately so even an early interruption is
        # resumable instead of leaving otherwise-orphaned memmaps.
        _flush(arrays)
        _save_state(
            state_path,
            fingerprint=fingerprint,
            rows=rows_by_slot,
            verification_rows=verification_by_slot,
            global_mass_sum=global_mass_sum,
            elapsed_seconds=0.0,
        )

    orders = residual_orders(L, device=device)
    model, gated_blocks, _ = build_gated_resnet34()
    model.to(device).eval()
    start = time.time()
    newly_processed = 0
    for x, label, master_index, sample_id_batch in loader:
        sample_id = sample_id_from_batch(sample_id_batch)
        record = record_lookup[sample_id]
        slot = int(record.selection_position)
        if slot in rows_by_slot:
            continue
        x = x.to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)
        image_start = time.time()
        row_values, tensors, verification = _run_one(
            model, gated_blocks, x, label,
            mask_chunk=args.mask_chunk,
            k_chunk=args.k_chunk,
            max_signature_k=max_k,
            orders_residual=orders,
            low_order_cutoff=args.low_order_cutoff,
        )
        vector_indices[slot, :] = tensors["vector_indices"].numpy()
        scalar_indices[slot, :] = tensors["scalar_indices"].numpy()
        vector_weights[slot, :] = tensors["vector_weights"].numpy()
        global_mass_sum.add_(tensors["full_mass_weights"])
        row = {
            "sample_id": sample_id,
            "selection_position": slot,
            "master_index": int(record.master_index),
            "shard": record.shard,
            "row": int(record.row),
            **row_values,
            "runtime_seconds": float(time.time() - image_start),
        }
        rows_by_slot[slot] = row
        verification_by_slot[slot] = {"sample_id": sample_id, **verification}
        newly_processed += 1
        elapsed = elapsed_before + time.time() - start
        _append_progress(
            progress_path,
            {
                "event": "image_complete", "sample_id": sample_id,
                "selection_position": slot, "completed": len(rows_by_slot),
                "total": total, "elapsed_seconds": elapsed,
            },
        )
        if len(rows_by_slot) % args.checkpoint_every == 0:
            _flush(arrays)
            _save_state(
                state_path, fingerprint=fingerprint, rows=rows_by_slot,
                verification_rows=verification_by_slot,
                global_mass_sum=global_mass_sum, elapsed_seconds=elapsed,
            )
        if newly_processed % args.log_every == 0 or len(rows_by_slot) == total:
            print(
                f"[exp4+5] {len(rows_by_slot)}/{total} sample={sample_id} "
                f"N_eff_h={row_values['h/N_eff']:.1f} "
                f"Kerr.10={row_values['K_err_010']} elapsed={elapsed:.1f}s",
                flush=True,
            )
        if args.stop_after_images is not None and newly_processed >= args.stop_after_images:
            _flush(arrays)
            _save_state(
                state_path, fingerprint=fingerprint, rows=rows_by_slot,
                verification_rows=verification_by_slot,
                global_mass_sum=global_mass_sum, elapsed_seconds=elapsed,
            )
            print(f"paused safely after {newly_processed} new images")
            return 0

    wall_seconds = elapsed_before + time.time() - start
    _flush(arrays)
    _save_state(
        state_path, fingerprint=fingerprint, rows=rows_by_slot,
        verification_rows=verification_by_slot,
        global_mass_sum=global_mass_sum, elapsed_seconds=wall_seconds,
    )
    if set(rows_by_slot) != set(range(total)):
        raise RuntimeError("completed slots do not exactly cover the selected dataset")
    if set(verification_by_slot) != set(range(total)):
        raise RuntimeError("verification slots do not exactly cover the selected dataset")
    completed_ids = [str(row["sample_id"]) for row in rows_by_slot.values()]
    if len(set(completed_ids)) != total:
        raise RuntimeError("completed sample IDs are not unique")
    _finalize(
        args, config, rows_by_slot, verification_by_slot, global_mass_sum,
        k_grid=k_grid, signature_arrays=arrays, wall_seconds=wall_seconds,
    )
    print(
        f"completed ResNet-34 Experiments 4 and 5: N={total}, "
        f"wall={wall_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
