"""Run ResNet-34 Experiments 1 and 2 in one exact mask sweep.

For every selected image this evaluator runs all 65,536 residual masks once,
performs the FP64 Mobius transform, and emits the original Experiment-1 and
Experiment-2 schemas for downstream tabular analysis.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from resnet_routes.resnet18.evaluation import render_table as render_exp1_table

from .model import build_gated_resnet34
from .mask_evaluator import evaluate_all_masks
from .mobius import mobius_coefficients

from .common import (
    L,
    NUM_MASKS,
    atomic_save_checkpoint,
    atomic_save_parquet,
    atomic_write_json,
    build_loader,
    config_fingerprint,
    configure_determinism,
    coordinate_summary,
    implementation_hashes,
    load_manifest_selection,
    load_resume_checkpoint,
    mean_se,
    sample_id_from_batch,
    software_hardware_metadata,
    write_result_checksums,
)
from .metrics import experiment1_metrics, experiment2_metrics, render_spectrum_table


SCHEMA_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=Path,
                        default=Path("data/indices/imagenet_test_10000.json"))
    parser.add_argument(
        "--selection-manifest", type=Path,
        default=Path("data/indices/resnet34_test_splits_seed3403.json"),
    )
    parser.add_argument(
        "--split", default="extension",
        choices=("smoke", "benchmark", "pilot", "main", "extension", "extension_append"),
    )
    parser.add_argument("--parquet-dir", type=Path, default=Path("imagenet-1k/data"))
    parser.add_argument("--synthetic-images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-chunk", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-images", type=int, default=None)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--work-dir", type=Path,
        default=Path("runs/resnet34/reconstruction_and_spectrum"),
    )
    parser.add_argument(
        "--exp1-output-dir", type=Path,
        default=Path("experiments/naive_additive_reconstruction/generated/resnet34"),
    )
    parser.add_argument(
        "--exp2-output-dir", type=Path,
        default=Path("experiments/interaction_order_spectrum/generated/resnet34"),
    )
    args = parser.parse_args(argv)
    if args.mask_chunk < 1 or args.num_workers < 0:
        parser.error("mask-chunk must be positive and num-workers non-negative")
    if args.checkpoint_every < 1 or args.log_every < 1:
        parser.error("checkpoint/log intervals must be positive")
    if args.stop_after_images is not None and args.stop_after_images < 1:
        parser.error("stop-after-images must be positive")
    return args


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_one(
    model: torch.nn.Module,
    gated_blocks: Sequence[torch.nn.Module],
    x: torch.Tensor,
    *,
    mask_chunk: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with torch.inference_mode():
        direct_full = model(x).detach().to(torch.float64)
    h_masks = evaluate_all_masks(model, gated_blocks, x, mask_chunk=mask_chunk)
    chunk_full = h_masks[:, -1, :].to(torch.float64)
    chunk_diff = (direct_full - chunk_full).abs()

    deltas = mobius_coefficients(h_masks, dtype=torch.float64)
    exp1 = experiment1_metrics(h_masks, deltas)
    exp2 = experiment2_metrics(deltas, chunk_full)
    finite = all(
        torch.isfinite(torch.as_tensor(value, dtype=torch.float64)).all().item()
        for value in list(exp1.values()) + list(exp2.values())
    )
    verification = {
        "chunk_all_open_max_abs_diff": float(chunk_diff.max().item()),
        "chunk_all_open_mean_abs_diff": float(chunk_diff.mean().item()),
        "chunk_all_open_top1_equal": bool(
            direct_full.argmax(dim=-1).eq(chunk_full.argmax(dim=-1)).all().item()
        ),
        "mobius_reconstruction_relative_error": exp1["mobius_recon_err"],
        "mobius_top1_agreement": exp1["mob/top1_agree"],
        "all_metrics_finite": bool(finite),
    }
    passed = (
        verification["chunk_all_open_max_abs_diff"] <= 1e-4
        and verification["chunk_all_open_top1_equal"]
        and verification["mobius_reconstruction_relative_error"] <= 1e-10
        and verification["mobius_top1_agreement"] == 1.0
        and verification["all_metrics_finite"]
    )
    verification["passed"] = bool(passed)
    if not passed:
        raise RuntimeError(f"per-image verification failed: {verification}")
    del deltas, h_masks
    return {"experiment1": exp1, "experiment2": exp2}, verification


def _save_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    rows: Sequence[Mapping[str, Any]],
    verification_rows: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> None:
    atomic_save_checkpoint(
        {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": fingerprint,
            "rows": list(rows),
            "verification_rows": list(verification_rows),
            "elapsed_seconds": float(elapsed_seconds),
            "saved_at": _utc_now(),
        },
        path,
    )


def _verification_summary(
    rows: Sequence[Mapping[str, Any]], expected: int
) -> dict[str, Any]:
    if len(rows) != expected:
        raise RuntimeError(f"completed {len(rows)} images, expected {expected}")
    unique = len({str(row["sample_id"]) for row in rows}) == len(rows)
    result = {
        "num_verified_images": len(rows),
        "expected_images": expected,
        "unique_sample_ids": unique,
        "all_passed": all(bool(row["passed"]) for row in rows),
        "max_chunk_all_open_abs_diff": max(
            float(row["chunk_all_open_max_abs_diff"]) for row in rows
        ),
        "max_mobius_reconstruction_relative_error": max(
            float(row["mobius_reconstruction_relative_error"]) for row in rows
        ),
        "all_mobius_top1_agreement": all(
            float(row["mobius_top1_agreement"]) == 1.0 for row in rows
        ),
        "all_metrics_finite": all(bool(row["all_metrics_finite"]) for row in rows),
        "gate_count": L,
        "mask_count": NUM_MASKS,
    }
    result["passed"] = bool(
        unique
        and result["all_passed"]
        and result["all_mobius_top1_agreement"]
        and result["all_metrics_finite"]
        and result["max_chunk_all_open_abs_diff"] <= 1e-4
        and result["max_mobius_reconstruction_relative_error"] <= 1e-10
    )
    return result


def _finalize(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    verification_rows: Sequence[Mapping[str, Any]],
    *,
    expected_positions: Sequence[int],
    wall_seconds: float,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["selection_position"]))
    normalized_expected = sorted(int(value) for value in expected_positions)
    expected_images = len(normalized_expected)
    if len(set(normalized_expected)) != expected_images:
        raise RuntimeError("selected manifest positions are not unique")
    if len(ordered) != expected_images:
        raise RuntimeError(
            f"completed {len(ordered)} result rows, expected {expected_images}"
        )
    positions = [int(row["selection_position"]) for row in ordered]
    if positions != normalized_expected:
        raise RuntimeError("result rows do not exactly cover the selected manifest positions")
    verification = _verification_summary(verification_rows, expected_images)
    if not verification["passed"]:
        raise RuntimeError(f"final verification failed: {verification}")

    common_config = {
        **dict(config),
        "num_images": len(ordered),
        "wall_seconds": float(wall_seconds),
        "status": "complete",
    }

    # Experiment 1: preserve the original summary schema and raw per-image logs.
    raw_logs: dict[str, list[float]] = {}
    for row in ordered:
        for name, value in row["experiment1"].items():
            raw_logs.setdefault(name, []).append(float(value))
    exp1_metrics = {name: mean_se(values) for name, values in raw_logs.items()}
    exp1_dir = args.exp1_output_dir
    exp1_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {"config": common_config, "metrics": exp1_metrics, "raw_logs": raw_logs},
        exp1_dir / "summary.json",
    )
    (exp1_dir / "table_1.md").write_text(render_exp1_table(exp1_metrics) + "\n")
    atomic_write_json(verification, exp1_dir / "verification.json")
    atomic_save_parquet(ordered, exp1_dir / "per_image_summary.parquet")
    write_result_checksums(
        exp1_dir,
        ("summary.json", "table_1.md", "verification.json", "per_image_summary.parquet"),
    )

    # Experiment 2: coordinate-wise mean/SE for every order statistic.
    exp2_keys = tuple(ordered[0]["experiment2"].keys())
    exp2_metrics: dict[str, Any] = {}
    for name in exp2_keys:
        values = [row["experiment2"][name] for row in ordered]
        if name.endswith("/kappa"):
            exp2_metrics[name] = mean_se([float(value) for value in values])
        else:
            exp2_metrics[name] = coordinate_summary(values)
    exp2_dir = args.exp2_output_dir
    exp2_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {"config": common_config, "metrics": exp2_metrics},
        exp2_dir / "summary.json",
    )
    (exp2_dir / "table_2.md").write_text(render_spectrum_table(exp2_metrics, L))
    atomic_write_json(verification, exp2_dir / "verification.json")
    atomic_save_parquet(ordered, exp2_dir / "per_image_summary.parquet")
    write_result_checksums(
        exp2_dir,
        ("summary.json", "table_2.md", "verification.json", "per_image_summary.parquet"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_determinism(args.seed, cudnn_benchmark=args.cudnn_benchmark)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.work_dir / "checkpoint.pt"
    progress_path = args.work_dir / "progress.jsonl"

    selection = None
    if args.synthetic_images is None:
        selection = load_manifest_selection(
            args.sample_index,
            args.selection_manifest,
            split=args.split,
            parquet_dir=args.parquet_dir,
        )
    loader, record_lookup, total = build_loader(
        selection,
        synthetic_images=args.synthetic_images,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    repo_root = Path(__file__).resolve().parents[1]
    numerical_files = [
        Path(__file__), Path(__file__).with_name("common.py"),
        Path(__file__).with_name("metrics.py"),
        repo_root / "resnet18/spectrum.py",
        repo_root / "resnet34/model.py",
        repo_root / "resnet34/mask_evaluator.py",
        repo_root / "resnet34/mobius.py",
    ]
    device = torch.device(args.device)
    runtime_metadata = software_hardware_metadata(device)
    config = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "torchvision.models.resnet34",
        "weights": "ResNet34_Weights.IMAGENET1K_V1",
        "L": L,
        "num_masks": NUM_MASKS,
        "forward_dtype": "float32",
        "mobius_dtype": "float64",
        "sample_index": None if args.synthetic_images else str(args.sample_index.resolve()),
        "sample_index_sha256": None if selection is None else selection.source_sha256,
        "selection_manifest": None if args.synthetic_images else str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": None if selection is None else selection.manifest_sha256,
        "selector": {"synthetic_images": args.synthetic_images} if selection is None else selection.selector,
        "parquet_dir": None if args.synthetic_images else str(args.parquet_dir.resolve()),
        "mask_chunk": args.mask_chunk,
        "seed": args.seed,
        "device": args.device,
        "software_hardware": runtime_metadata,
        "cudnn_benchmark": args.cudnn_benchmark,
        "cudnn_deterministic": True,
        "tf32": False,
        "implementation_sha256": implementation_hashes(numerical_files, repo_root),
    }
    fingerprint = config_fingerprint(config)
    rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but {checkpoint_path} is missing")
        state = load_resume_checkpoint(checkpoint_path, fingerprint)
        rows = [dict(row) for row in state["rows"]]
        verification_rows = [dict(row) for row in state["verification_rows"]]
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
    elif checkpoint_path.exists():
        raise FileExistsError(
            f"{checkpoint_path} already exists; use --resume or a new --work-dir"
        )

    completed = {str(row["sample_id"]) for row in rows}
    model, gated_blocks, _ = build_gated_resnet34()
    model.to(device).eval()
    start = time.time()
    newly_processed = 0
    for x, label, master_index, sample_id_batch in loader:
        sample_id = sample_id_from_batch(sample_id_batch)
        if sample_id in completed:
            continue
        record = record_lookup[sample_id]
        x = x.to(device, non_blocking=True)
        image_start = time.time()
        values, verification = _run_one(
            model, gated_blocks, x, mask_chunk=args.mask_chunk
        )
        row = {
            "sample_id": sample_id,
            "selection_position": int(record.selection_position),
            "master_index": int(record.master_index),
            "shard": record.shard,
            "row": int(record.row),
            "label": int(label.item()),
            **values,
            "runtime_seconds": float(time.time() - image_start),
        }
        verification_row = {"sample_id": sample_id, **verification}
        rows.append(row)
        verification_rows.append(verification_row)
        completed.add(sample_id)
        newly_processed += 1
        elapsed = elapsed_before + time.time() - start
        _append_progress(
            progress_path,
            {
                "event": "image_complete", "sample_id": sample_id,
                "selection_position": record.selection_position,
                "completed": len(rows), "total": total,
                "elapsed_seconds": elapsed,
            },
        )
        if len(rows) % args.checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_path, fingerprint=fingerprint, rows=rows,
                verification_rows=verification_rows, elapsed_seconds=elapsed,
            )
        if newly_processed % args.log_every == 0 or len(rows) == total:
            print(
                f"[exp1+2] {len(rows)}/{total} sample={sample_id} "
                f"mob_err={values['experiment1']['mobius_recon_err']:.2e} "
                f"kappa_h={values['experiment2']['h/kappa']:.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if args.stop_after_images is not None and newly_processed >= args.stop_after_images:
            _save_checkpoint(
                checkpoint_path, fingerprint=fingerprint, rows=rows,
                verification_rows=verification_rows, elapsed_seconds=elapsed,
            )
            print(f"paused safely after {newly_processed} new images")
            return 0

    wall_seconds = elapsed_before + time.time() - start
    expected_ids = set(record_lookup)
    if completed != expected_ids or len(rows) != total:
        raise RuntimeError("completed samples do not exactly match the selected dataset")
    _save_checkpoint(
        checkpoint_path, fingerprint=fingerprint, rows=rows,
        verification_rows=verification_rows, elapsed_seconds=wall_seconds,
    )
    _finalize(
        args, config, rows, verification_rows,
        expected_positions=[
            record.selection_position for record in record_lookup.values()
        ],
        wall_seconds=wall_seconds,
    )
    print(
        f"completed ResNet-34 Experiments 1 and 2: N={len(rows)}, "
        f"wall={wall_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
