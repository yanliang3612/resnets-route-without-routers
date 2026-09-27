"""Exact inference-time residual-scaling sweep for ImageNet ResNet-34.

This runner extends the audited ResNet-34 Experiment-2 implementation without
modifying its gated model, binary-mask evaluator, or metric code.  For a
binary subset mask ``b`` and scale ``lambda``, it evaluates the continuous
gate vector ``lambda * b``.  Every image is still evaluated once with all
gates open, independently of the requested sweep values, so top-1 agreement
always uses the genuine ``lambda=1`` full-model prediction as its reference.

The default data selection is the same frozen ``extension`` split (N=1,000)
used by the existing ResNet-34 Experiments 1/2.  The output
``summary_all.json`` intentionally preserves the ResNet-18 lambda-sweep schema.

Examples (from the repository root)::

    # Small smoke test.
    python -m resnet34_experiments.evaluate_lambda_sweep \
        --synthetic-images 2 --lambdas 1,.5 --device cuda \
        --work-dir /tmp/r34-lambda-smoke-work \
        --output-dir /tmp/r34-lambda-smoke

    # Full default nine-point sweep on the frozen extension split.
    python -m resnet34_experiments.evaluate_lambda_sweep

    # One-lambda job (suitable for one-GPU-per-lambda execution).
    python -m resnet34_experiments.evaluate_lambda_sweep \
        --lambda .5 \
        --work-dir runs/resnet34_lambda/lambda_0.5 \
        --output-dir experiment_2/results_resnet34_lambda_shards/lambda_0.5

    # Merge independently completed lambda jobs without running a model.
    python -m resnet34_experiments.evaluate_lambda_sweep \
        --merge-inputs experiment_2/results_resnet34_lambda_shards/lambda_* \
        --output-dir experiment_2/results_resnet34_lambda

    # Reuse the audited lambda=1 Experiment-2 aggregate instead of enumerating
    # its 65,536 masks again.  The direct lambda=1 reference prediction is
    # nevertheless computed for every image.
    python -m resnet34_experiments.evaluate_lambda_sweep \
        --skip-lambda-one-enumeration
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

from .model import (
    GATE_NAMES,
    build_gated_resnet34,
    masks_from_integers,
    reset_masks,
    set_per_sample_masks,
)
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
from .metrics import experiment2_metrics


SCHEMA_VERSION = 1
DEFAULT_LAMBDAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.5, 0.25, 0.1)
METRIC_KEYS = (
    "v/E", "h/E",
    "v/E_tilde", "h/E_tilde",
    "v/E_bar", "h/E_bar",
    "v/M", "h/M",
    "v/cum", "h/cum",
    "v/tail", "h/tail",
    "v/kappa", "h/kappa",
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without relying on common.py."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--merge-inputs", type=Path, nargs="+", default=None,
        help="merge completed one-lambda output directories; no model is run",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--selection-manifest", type=Path,
        default=Path("data/indices/resnet34_test_splits_seed3403.json"),
    )
    source.add_argument(
        "--synthetic-images", type=int, default=None,
        help="deterministic random images for smoke testing",
    )
    parser.add_argument(
        "--sample-index", type=Path,
        default=Path("data/indices/imagenet_test_10000.json")
    )
    parser.add_argument(
        "--split", default="extension",
        choices=("smoke", "benchmark", "pilot", "main", "extension", "extension_append"),
    )
    parser.add_argument("--parquet-dir", type=Path, default=Path("imagenet-1k/data"))
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument("--lambda", dest="single_lambda", type=float, default=None)
    sweep.add_argument(
        "--lambdas", type=str, default=None,
        help="comma-separated values (default: 1,.95,.9,.85,.8,.75,.5,.25,.1)",
    )
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
        "--skip-lambda-one-enumeration", action="store_true",
        help="inject lambda=1 aggregate metrics from --lambda-one-summary",
    )
    parser.add_argument(
        "--lambda-one-summary", type=Path,
        default=Path("experiments/interaction_order_spectrum/generated/resnet34/summary.json"),
    )
    parser.add_argument(
        "--merge-lambda-one", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "when merging real extension-split shards, inject lambda=1 from "
            "--lambda-one-summary if no input shard contains it"
        ),
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("runs/resnet34/residual_scaling_sweep")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/residual_scaling_sweep/generated/resnet34"),
    )
    args = parser.parse_args(argv)

    if args.mask_chunk < 1 or args.num_workers < 0:
        parser.error("mask-chunk must be positive and num-workers non-negative")
    if args.checkpoint_every < 1 or args.log_every < 1:
        parser.error("checkpoint/log intervals must be positive")
    if args.stop_after_images is not None and args.stop_after_images < 1:
        parser.error("stop-after-images must be positive")
    if args.synthetic_images is not None and args.synthetic_images < 1:
        parser.error("synthetic-images must be positive")
    if args.merge_inputs is not None:
        if args.resume or args.stop_after_images is not None:
            parser.error("resume/stop-after-images do not apply in merge mode")
        return args
    args.resolved_lambdas = parse_lambdas(args.single_lambda, args.lambdas)
    if args.skip_lambda_one_enumeration and not _contains_lambda(args.resolved_lambdas, 1.0):
        parser.error("--skip-lambda-one-enumeration requires lambda=1 in the requested sweep")
    if args.skip_lambda_one_enumeration and len(args.resolved_lambdas) == 1:
        parser.error(
            "lambda=1-only injection has no per-image sweep rows; use the existing "
            "Experiment-2 result directly"
        )
    if args.skip_lambda_one_enumeration and args.synthetic_images is not None:
        parser.error("lambda=1 injection is only valid for the frozen real-data extension split")
    if args.skip_lambda_one_enumeration and args.split != "extension":
        parser.error("lambda=1 injection is only valid with --split extension")
    return args


def parse_lambdas(single: float | None, specification: str | None) -> list[float]:
    if single is not None:
        values = [single]
    elif specification is None:
        values = list(DEFAULT_LAMBDAS)
    else:
        values = [float(token.strip()) for token in specification.split(",") if token.strip()]
    if not values:
        raise ValueError("at least one lambda is required")
    normalized: list[float] = []
    for value in values:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"lambda values must be finite and positive, got {value}")
        if value not in normalized:
            normalized.append(value)
    return sorted(normalized, reverse=True)


def _contains_lambda(values: Sequence[float], target: float) -> bool:
    return any(float(value) == float(target) for value in values)


def _lambda_key(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _append_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_all_scaled_masks(
    model: torch.nn.Module,
    gated_blocks: Sequence[torch.nn.Module],
    x: torch.Tensor,
    *,
    residual_scale: float,
    mask_chunk: int,
) -> torch.Tensor:
    """Evaluate all binary subsets after multiplying every open bit by lambda.

    This is deliberately implemented here rather than changing the audited
    binary evaluator.  Mask ordering remains integer/LSB-first, so the existing
    view-based Möbius transform can be reused unchanged.
    """
    if x.ndim < 1 or x.shape[0] != 1:
        raise ValueError(f"exact ResNet-34 enumeration requires B=1, got {tuple(x.shape)}")
    if len(gated_blocks) != L or L != len(GATE_NAMES):
        raise ValueError(f"expected {L} gated blocks, got {len(gated_blocks)}")
    if mask_chunk < 1:
        raise ValueError("mask_chunk must be positive")
    if not math.isfinite(residual_scale) or residual_scale < 0.0:
        raise ValueError("residual_scale must be finite and non-negative")

    output: torch.Tensor | None = None
    try:
        with torch.inference_mode():
            for start in range(0, NUM_MASKS, mask_chunk):
                stop = min(start + mask_chunk, NUM_MASKS)
                mask_ids = torch.arange(start, stop, dtype=torch.int64, device=x.device)
                masks = masks_from_integers(mask_ids, device=x.device)
                masks.mul_(float(residual_scale))
                set_per_sample_masks(gated_blocks, masks)
                logits = model(x.expand(stop - start, *x.shape[1:]))
                if logits.ndim != 2 or logits.shape[0] != stop - start:
                    raise ValueError(f"model output has invalid shape {tuple(logits.shape)}")
                if not torch.isfinite(logits).all().item():
                    raise FloatingPointError(
                        f"non-finite logits at lambda={residual_scale}, masks [{start},{stop})"
                    )
                if output is None:
                    output = torch.empty(
                        (NUM_MASKS, logits.shape[1]), dtype=torch.float32, device=x.device
                    )
                elif logits.shape[1] != output.shape[1]:
                    raise ValueError("model output width changed during mask enumeration")
                output[start:stop].copy_(logits.detach().to(torch.float32))
    finally:
        reset_masks(gated_blocks)
    if output is None:
        raise RuntimeError("no masks were evaluated")
    return output.unsqueeze(0).contiguous()


def _direct_scaled_full(
    model: torch.nn.Module,
    gated_blocks: Sequence[torch.nn.Module],
    x: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    full_gate = torch.full(
        (1, L), float(residual_scale), dtype=torch.float32, device=x.device
    )
    try:
        with torch.inference_mode():
            set_per_sample_masks(gated_blocks, full_gate)
            output = model(x).detach().to(torch.float64)
            if not torch.isfinite(output).all().item():
                raise FloatingPointError(
                    f"non-finite direct full output at lambda={residual_scale}"
                )
            return output
    finally:
        reset_masks(gated_blocks)


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    if set(metrics) != set(METRIC_KEYS):
        raise RuntimeError(f"metric keys differ: got {sorted(metrics)}")
    for key in METRIC_KEYS:
        value = torch.as_tensor(metrics[key], dtype=torch.float64)
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"non-finite metric {key}")
        expected = None
        if key.endswith(("/E", "/E_bar", "/M")):
            expected = L + 1
        elif key.endswith(("/E_tilde", "/cum", "/tail")):
            expected = L
        elif key.endswith("/kappa"):
            if value.ndim != 0 or not (1.0 - 1e-8 <= value.item() <= L + 1e-8):
                raise RuntimeError(f"invalid effective order {key}={value}")
        if expected is not None and tuple(value.shape) != (expected,):
            raise RuntimeError(f"{key} has shape {tuple(value.shape)}, expected {(expected,)}")
    for readout in ("v", "h"):
        spectrum = torch.as_tensor(metrics[f"{readout}/E_tilde"], dtype=torch.float64)
        cumulative = torch.as_tensor(metrics[f"{readout}/cum"], dtype=torch.float64)
        tail = torch.as_tensor(metrics[f"{readout}/tail"], dtype=torch.float64)
        if not torch.isclose(spectrum.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-8):
            raise RuntimeError(f"{readout} normalized residual spectrum does not sum to one")
        if not torch.all(cumulative[1:] + 1e-10 >= cumulative[:-1]):
            raise RuntimeError(f"{readout} cumulative spectrum is not monotone")
        if not torch.allclose(cumulative + tail, torch.ones_like(cumulative), atol=1e-10):
            raise RuntimeError(f"{readout} cumulative/tail identity failed")


def _run_one_lambda(
    model: torch.nn.Module,
    gated_blocks: Sequence[torch.nn.Module],
    x: torch.Tensor,
    *,
    residual_scale: float,
    mask_chunk: int,
    top1_reference: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask_logits = evaluate_all_scaled_masks(
        model, gated_blocks, x,
        residual_scale=residual_scale, mask_chunk=mask_chunk,
    )
    full_from_enumeration = mask_logits[:, -1, :].to(torch.float64)
    direct_scaled = _direct_scaled_full(model, gated_blocks, x, residual_scale)
    full_difference = (direct_scaled - full_from_enumeration).abs()
    coefficients = mobius_coefficients(mask_logits, dtype=torch.float64)
    reconstruction = coefficients.sum(dim=1)
    reconstruction_error = (
        torch.linalg.vector_norm(reconstruction - full_from_enumeration, dim=-1)
        / (torch.linalg.vector_norm(full_from_enumeration, dim=-1) + 1e-12)
    )
    metrics = experiment2_metrics(coefficients, full_from_enumeration)
    _validate_metrics(metrics)
    top1_scaled = int(full_from_enumeration.argmax(dim=-1).item())
    agreement = float(top1_scaled == int(top1_reference))
    verification = {
        "lambda": float(residual_scale),
        "full_chunk_direct_max_abs_diff": float(full_difference.max().item()),
        "full_chunk_direct_top1_equal": bool(
            direct_scaled.argmax(dim=-1).eq(full_from_enumeration.argmax(dim=-1)).all().item()
        ),
        "mobius_reconstruction_relative_error": float(reconstruction_error.item()),
        "all_metrics_finite": True,
    }
    verification["passed"] = bool(
        verification["full_chunk_direct_max_abs_diff"] <= 1e-4
        and verification["full_chunk_direct_top1_equal"]
        and verification["mobius_reconstruction_relative_error"] <= 1e-10
    )
    if not verification["passed"]:
        raise RuntimeError(f"lambda={residual_scale} verification failed: {verification}")
    del coefficients, mask_logits
    return {
        "metrics": metrics,
        "top1_scaled": top1_scaled,
        "top1_agree": agreement,
    }, verification


def _save_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    rows: Sequence[Mapping[str, Any]],
    verification_rows: Sequence[Mapping[str, Any]],
    completed_sample_ids: Sequence[str],
    elapsed_seconds: float,
) -> None:
    atomic_save_checkpoint(
        {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": fingerprint,
            "rows": list(rows),
            "verification_rows": list(verification_rows),
            "completed_sample_ids": sorted(str(value) for value in completed_sample_ids),
            "elapsed_seconds": float(elapsed_seconds),
            "saved_at": _utc_now(),
        },
        path,
    )


def _summary_stat(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize richer R34 summaries to the R18 mean/stderr/n contract."""
    return {
        "mean": value["mean"],
        "stderr": value["stderr"],
        "n": int(value["n"]),
    }


def _verify_result_checksums(directory: Path) -> None:
    """Verify every artifact listed in a shard's SHA256SUMS manifest."""
    manifest = directory / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing checksum manifest: {manifest}")
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"malformed {manifest}:{line_number}")
        expected, raw_name = fields
        name = raw_name.lstrip("* ")
        if Path(name).name != name:
            raise ValueError(f"unsafe checksum entry {name!r} in {manifest}")
        artifact = directory / name
        if not artifact.is_file():
            raise FileNotFoundError(f"checksum artifact is missing: {artifact}")
        actual = sha256_file(artifact)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {artifact}: expected {expected}, got {actual}"
            )
        seen.add(name)
    required = {
        "summary_all.json", "per_image_rows.parquet", "verification.json",
        "table_3.md", "latex.tex",
    }
    if not required.issubset(seen):
        raise ValueError(
            f"{manifest} does not cover required artifacts: {sorted(required - seen)}"
        )


def _load_lambda_one_injection(
    path: Path,
    *,
    expected_images: int,
    expected_source_sha256: str | None,
    expected_manifest_sha256: str | None,
    expected_implementation_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    config = blob.get("config", {})
    metrics = blob.get("metrics", {})
    if config.get("status") != "complete":
        raise ValueError("lambda=1 injection summary is not marked complete")
    if config.get("architecture") != "torchvision.models.resnet34":
        raise ValueError("lambda=1 injection uses a different architecture")
    if config.get("weights") != "ResNet34_Weights.IMAGENET1K_V1":
        raise ValueError("lambda=1 injection uses different pretrained weights")
    if config.get("forward_dtype") != "float32":
        raise ValueError("lambda=1 injection uses a different forward dtype")
    if config.get("mobius_dtype") != "float64":
        raise ValueError("lambda=1 injection uses a different Mobius dtype")
    if int(config.get("L", -1)) != L or int(config.get("num_masks", -1)) != NUM_MASKS:
        raise ValueError("lambda=1 injection is not a 16-gate exact ResNet-34 summary")
    if int(config.get("num_images", -1)) != expected_images:
        raise ValueError("lambda=1 summary uses a different number of images")
    if expected_source_sha256 is not None and config.get("sample_index_sha256") != expected_source_sha256:
        raise ValueError("lambda=1 summary uses a different sample index")
    if expected_manifest_sha256 is not None and config.get("selection_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("lambda=1 summary uses a different selection manifest")
    selector = config.get("selector", {})
    if selector.get("kind") != "manifest_split" or selector.get("split") != "extension":
        raise ValueError("lambda=1 injection must come from the extension split")
    if expected_implementation_sha256 is not None:
        injected_hashes = config.get("implementation_sha256", {})
        shared = set(injected_hashes).intersection(expected_implementation_sha256)
        required_shared = {
            "resnet18/spectrum.py",
            "resnet34/model.py",
            "resnet34/mobius.py",
            "resnet34/common.py",
            "resnet34/metrics.py",
        }
        if not required_shared.issubset(shared):
            raise ValueError("lambda=1 injection lacks shared implementation hashes")
        mismatched = [
            name for name in required_shared
            if injected_hashes[name] != expected_implementation_sha256[name]
        ]
        if mismatched:
            raise ValueError(
                f"lambda=1 injection implementation hashes differ: {mismatched}"
            )
    output: dict[str, Any] = {}
    for key in METRIC_KEYS:
        if key not in metrics:
            raise ValueError(f"lambda=1 summary is missing {key}")
        output[key] = _summary_stat(metrics[key])
        if int(output[key]["n"]) != expected_images:
            raise ValueError(f"lambda=1 summary {key} has inconsistent n")
        mean = torch.as_tensor(output[key]["mean"], dtype=torch.float64)
        stderr = torch.as_tensor(output[key]["stderr"], dtype=torch.float64)
        if mean.shape != stderr.shape:
            raise ValueError(f"lambda=1 summary {key} mean/stderr shapes differ")
        if not torch.isfinite(mean).all().item() or not torch.isfinite(stderr).all().item():
            raise ValueError(f"lambda=1 summary {key} contains non-finite values")
        if bool((stderr < 0).any().item()):
            raise ValueError(f"lambda=1 summary {key} has a negative standard error")
    _validate_metrics({key: output[key]["mean"] for key in METRIC_KEYS})
    output["top1_agree"] = {"mean": 1.0, "stderr": 0.0, "n": expected_images}
    return output


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    lambdas: Sequence[float],
    *,
    injected_lambda_one: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for residual_scale in lambdas:
        key = _lambda_key(residual_scale)
        if residual_scale == 1.0 and injected_lambda_one is not None:
            result[key] = dict(injected_lambda_one)
            continue
        selected = [row for row in rows if _lambda_key(float(row["lambda"])) == key]
        if not selected:
            raise RuntimeError(f"no per-image rows for lambda={residual_scale}")
        metrics: dict[str, Any] = {}
        for metric_key in METRIC_KEYS:
            values = [row["metrics"][metric_key] for row in selected]
            if metric_key.endswith("/kappa"):
                metrics[metric_key] = _summary_stat(mean_se([float(value) for value in values]))
            else:
                metrics[metric_key] = _summary_stat(coordinate_summary(values))
        metrics["top1_agree"] = _summary_stat(
            mean_se([float(row["top1_agree"]) for row in selected])
        )
        result[key] = metrics
    return result


def _render_markdown(per_lambda: Mapping[str, Mapping[str, Any]]) -> str:
    blocks = ["# ResNet-34 inference-time residual scaling sweep", ""]
    ordered = sorted((float(key), value) for key, value in per_lambda.items())
    for readout, label in (("v", "predicted-class scalar"), ("h", "logit vector")):
        blocks.extend([
            f"## Readout: {label}", "",
            "| $\\lambda$ | $\\kappa$ | $C_{\\le 3}$ | $T_{>3}$ | top-1 agreement |",
            "| -: | -: | -: | -: | -: |",
        ])
        for residual_scale, metrics in sorted(ordered, reverse=True):
            kappa = metrics[f"{readout}/kappa"]
            cumulative = metrics[f"{readout}/cum"]
            tail = metrics[f"{readout}/tail"]
            agreement = metrics["top1_agree"]
            blocks.append(
                f"| {residual_scale:.3f} | {kappa['mean']:.4f} +/- {kappa['stderr']:.4f} | "
                f"{cumulative['mean'][2]:.4f} +/- {cumulative['stderr'][2]:.4f} | "
                f"{tail['mean'][2]:.4f} +/- {tail['stderr'][2]:.4f} | "
                f"{agreement['mean']:.4f} +/- {agreement['stderr']:.4f} |"
            )
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def _render_latex(per_lambda: Mapping[str, Mapping[str, Any]]) -> str:
    tables: list[str] = []
    ordered = sorted(
        ((float(key), value) for key, value in per_lambda.items()), reverse=True
    )
    for readout, name, label in (
        ("v", "predicted-class scalar", "tab:exp3-r34-scalar"),
        ("h", "vector-logit", "tab:exp3-r34-vector"),
    ):
        lines = [
            "\\begin{table}[h]",
            "  \\centering",
            f"  \\caption{{ResNet-34 inference-time residual scaling sweep for the {name} readout.}}",
            f"  \\label{{{label}}}",
            "  \\small",
            "  \\setlength{\\tabcolsep}{6pt}",
            "  \\renewcommand{\\arraystretch}{1.08}",
            "  \\begin{tabular}{@{}rccc@{}}",
            "    \\toprule[1.2pt]",
            "    $\\lambda$ & $\\kappa$ & $C_{\\le 3}$ & Top-1 agreement with $\\lambda=1$ \\\\",
            "    \\midrule",
        ]
        for residual_scale, metrics in ordered:
            lines.append(
                f"    ${residual_scale:.3f}$ & ${metrics[f'{readout}/kappa']['mean']:.3f}$ & "
                f"${metrics[f'{readout}/cum']['mean'][2]:.3f}$ & "
                f"${metrics['top1_agree']['mean']:.3f}$ \\\\"
            )
        lines.extend([
            "    \\bottomrule[1.2pt]",
            "  \\end{tabular}",
            "\\end{table}",
        ])
        tables.append("\n".join(lines))
    return "\n\n".join(tables) + "\n"


def _final_verification(
    rows: Sequence[Mapping[str, Any]],
    verification_rows: Sequence[Mapping[str, Any]],
    *,
    expected_sample_ids: set[str],
    enumerated_lambdas: Sequence[float],
) -> dict[str, Any]:
    expected_pairs = {
        (sample_id, _lambda_key(value))
        for sample_id in expected_sample_ids for value in enumerated_lambdas
    }
    actual_pairs = {
        (str(row["sample_id"]), _lambda_key(float(row["lambda"]))) for row in rows
    }
    if len(actual_pairs) != len(rows) or actual_pairs != expected_pairs:
        raise RuntimeError("per-image rows do not exactly cover sample x enumerated-lambda pairs")
    if len(verification_rows) != len(rows):
        raise RuntimeError("verification row count differs from result row count")
    verification_pairs = {
        (str(row["sample_id"]), _lambda_key(float(row["lambda"])))
        for row in verification_rows
    }
    if len(verification_pairs) != len(verification_rows) or verification_pairs != expected_pairs:
        raise RuntimeError("verification rows do not cover the expected sample/lambda pairs")
    all_passed = all(bool(row["passed"]) for row in verification_rows)
    result = {
        "expected_images": len(expected_sample_ids),
        "enumerated_lambdas": list(enumerated_lambdas),
        "expected_per_image_rows": len(expected_pairs),
        "actual_per_image_rows": len(rows),
        "unique_sample_lambda_pairs": len(actual_pairs) == len(rows),
        "all_passed": all_passed,
        "max_full_chunk_direct_abs_diff": max(
            (float(row["full_chunk_direct_max_abs_diff"]) for row in verification_rows),
            default=0.0,
        ),
        "max_mobius_reconstruction_relative_error": max(
            (float(row["mobius_reconstruction_relative_error"]) for row in verification_rows),
            default=0.0,
        ),
        "passed": bool(all_passed and actual_pairs == expected_pairs),
    }
    return result


def _write_outputs(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    per_lambda: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {"config": dict(config), "per_lambda": dict(per_lambda)},
        output_dir / "summary_all.json",
    )
    atomic_save_parquet(rows, output_dir / "per_image_rows.parquet")
    atomic_write_json(dict(verification), output_dir / "verification.json")
    _atomic_write_text(output_dir / "table_3.md", _render_markdown(per_lambda))
    _atomic_write_text(output_dir / "latex.tex", _render_latex(per_lambda))
    write_result_checksums(
        output_dir,
        ("summary_all.json", "per_image_rows.parquet", "verification.json", "table_3.md", "latex.tex"),
    )


def _merge_outputs(args: argparse.Namespace) -> int:
    summaries: list[tuple[Path, dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    per_lambda: dict[str, Any] = {}
    summary_only_lambda_keys: set[str] = set()
    inherited_lambda_one_sources: list[tuple[str | None, str | None]] = []
    reference_signature: tuple[Any, ...] | None = None
    for directory in args.merge_inputs:
        summary_path = directory / "summary_all.json"
        parquet_path = directory / "per_image_rows.parquet"
        verification_path = directory / "verification.json"
        _verify_result_checksums(directory)
        blob = json.loads(summary_path.read_text(encoding="utf-8"))
        config = blob["config"]
        verification_blob = json.loads(verification_path.read_text(encoding="utf-8"))
        if config.get("status") != "complete":
            raise ValueError(f"merge input is not marked complete: {directory}")
        if not bool(verification_blob.get("passed")):
            raise ValueError(f"merge input verification failed: {directory}")
        if not parquet_path.is_file():
            raise FileNotFoundError(f"merge input has no per-image rows: {parquet_path}")
        signature = (
            config.get("architecture"), config.get("weights"), config.get("L"),
            config.get("num_masks"), config.get("num_images"),
            config.get("sample_index_sha256"), config.get("selection_manifest_sha256"),
            json.dumps(config.get("selector"), sort_keys=True), config.get("seed"),
            config.get("continuous_mask_definition"),
            json.dumps(config.get("implementation_sha256"), sort_keys=True),
        )
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(f"incompatible merge input {directory}")
        for key, metrics in blob["per_lambda"].items():
            if key in per_lambda and per_lambda[key] != metrics:
                raise ValueError(f"conflicting summaries for lambda={key}")
            per_lambda[key] = metrics
        if config.get("lambda_one_injected_summary") and "1" in blob["per_lambda"]:
            summary_only_lambda_keys.add("1")
            inherited_lambda_one_sources.append((
                config.get("lambda_one_injected_summary"),
                config.get("lambda_one_injected_summary_sha256"),
            ))
        import pyarrow.parquet as pq

        for row in pq.read_table(parquet_path).to_pylist():
            pair = (str(row["sample_id"]), _lambda_key(float(row["lambda"])))
            if pair in seen_pairs:
                raise ValueError(f"duplicate per-image pair {pair}")
            seen_pairs.add(pair)
            rows.append(row)
        summaries.append((directory, blob))
    if not summaries:
        raise ValueError("merge mode needs at least one input")
    base = dict(summaries[0][1]["config"])
    expected_images = int(base["num_images"])
    selector = base.get("selector", {})
    lambda_one_injected = False
    lambda_one_source_path: str | None = None
    lambda_one_source_sha256: str | None = None
    if (
        args.merge_lambda_one
        and "1" not in per_lambda
        and selector.get("kind") == "manifest_split"
        and selector.get("split") == "extension"
    ):
        per_lambda["1"] = _load_lambda_one_injection(
            args.lambda_one_summary,
            expected_images=expected_images,
            expected_source_sha256=base.get("sample_index_sha256"),
            expected_manifest_sha256=base.get("selection_manifest_sha256"),
            expected_implementation_sha256=base.get("implementation_sha256"),
        )
        summary_only_lambda_keys.add("1")
        lambda_one_injected = True
        lambda_one_source_path = str(args.lambda_one_summary.resolve())
        lambda_one_source_sha256 = sha256_file(args.lambda_one_summary)
    elif "1" in summary_only_lambda_keys:
        lambda_one_injected = True
        source_shas = {sha for _path, sha in inherited_lambda_one_sources}
        if None in source_shas or len(source_shas) != 1:
            raise ValueError("input shards disagree on inherited lambda=1 source provenance")
        lambda_one_source_path, lambda_one_source_sha256 = inherited_lambda_one_sources[0]
    lambdas = sorted((float(key) for key in per_lambda), reverse=True)
    for lambda_key, metrics in per_lambda.items():
        required = set(METRIC_KEYS) | {"top1_agree"}
        if set(metrics) != required:
            raise ValueError(
                f"lambda={lambda_key} has incompatible summary keys: {sorted(metrics)}"
            )
        if any(int(metrics[key]["n"]) != expected_images for key in required):
            raise ValueError(f"lambda={lambda_key} summary uses a different sample count")
        _validate_metrics({key: metrics[key]["mean"] for key in METRIC_KEYS})
    row_counts: dict[str, int] = {}
    coverage_by_lambda: dict[str, set[tuple[str, int]]] = {}
    reference_by_sample: dict[str, tuple[int, int]] = {}
    for row in rows:
        key = _lambda_key(float(row["lambda"]))
        row_counts[key] = row_counts.get(key, 0) + 1
        sample_id = str(row["sample_id"])
        position = int(row["selection_position"])
        coverage_by_lambda.setdefault(key, set()).add((sample_id, position))
        reference = (position, int(row["top1_reference"]))
        if sample_id in reference_by_sample and reference_by_sample[sample_id] != reference:
            raise RuntimeError(
                f"inconsistent selection position or lambda=1 reference for {sample_id}"
            )
        reference_by_sample[sample_id] = reference
    enumerated_keys = sorted(row_counts, key=float, reverse=True)
    expected_row_keys = set(per_lambda) - summary_only_lambda_keys
    if set(row_counts) != expected_row_keys:
        raise RuntimeError(
            "merged per-image lambda coverage does not match non-injected summaries: "
            f"rows={sorted(row_counts)}, expected={sorted(expected_row_keys)}"
        )
    if any(count != expected_images for count in row_counts.values()):
        raise RuntimeError(f"merged per-image coverage is incomplete: {row_counts}")
    if coverage_by_lambda:
        first_key = enumerated_keys[0]
        expected_coverage = coverage_by_lambda[first_key]
        if len(expected_coverage) != expected_images:
            raise RuntimeError(f"lambda={first_key} does not contain unique sample/position rows")
        for key, coverage in coverage_by_lambda.items():
            if coverage != expected_coverage:
                raise RuntimeError(f"lambda={key} covers a different sample/position set")
    base.update({
        "lambdas": lambdas,
        "enumerated_lambdas": sorted((float(key) for key in row_counts), reverse=True),
        "lambda_one_injected_summary": (
            lambda_one_source_path if lambda_one_injected else None
        ),
        "lambda_one_injected_summary_sha256": (
            lambda_one_source_sha256 if lambda_one_injected else None
        ),
        "merge_inputs": [str(path.resolve()) for path, _ in summaries],
        "wall_seconds": max(float(blob["config"].get("wall_seconds", 0.0)) for _, blob in summaries),
        "sum_job_seconds": sum(
            float(blob["config"].get("wall_seconds", 0.0)) for _, blob in summaries
        ),
        "status": "complete_merged",
    })
    verification = {
        "mode": "merged",
        "merge_inputs": len(summaries),
        "lambdas": lambdas,
        "enumerated_lambdas_with_per_image_rows": [float(key) for key in enumerated_keys],
        "lambda_one_injected": lambda_one_injected,
        "per_lambda_row_counts": row_counts,
        "sample_position_coverage_identical": True,
        "lambda_one_references_consistent": True,
        "actual_per_image_rows": len(rows),
        "passed": True,
    }
    rows.sort(key=lambda row: (int(row["selection_position"]), -float(row["lambda"])))
    _write_outputs(
        args.output_dir, config=base, per_lambda=per_lambda,
        rows=rows, verification=verification,
    )
    print(f"Merged {len(summaries)} result directories into {args.output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.merge_inputs is not None:
        return _merge_outputs(args)

    lambdas: list[float] = args.resolved_lambdas
    enumerated_lambdas = [
        value for value in lambdas
        if not (args.skip_lambda_one_enumeration and value == 1.0)
    ]
    configure_determinism(args.seed, cudnn_benchmark=args.cudnn_benchmark)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.work_dir / "checkpoint.pt"
    progress_path = args.work_dir / "progress.jsonl"

    selection = None
    if args.synthetic_images is None:
        selection = load_manifest_selection(
            args.sample_index, args.selection_manifest,
            split=args.split, parquet_dir=args.parquet_dir,
        )
    loader, record_lookup, total = build_loader(
        selection, synthetic_images=args.synthetic_images,
        seed=args.seed, num_workers=args.num_workers,
    )
    if selection is not None and args.split == "extension" and total != 1000:
        raise RuntimeError(
            f"frozen ResNet-34 extension split must contain N=1000 images, got {total}"
        )
    repo_root = Path(__file__).resolve().parents[1]
    numerical_files = [
        Path(__file__),
        repo_root / "resnet18/spectrum.py",
        repo_root / "resnet34/model.py",
        repo_root / "resnet34/mobius.py",
        repo_root / "resnet34/common.py",
        repo_root / "resnet34/metrics.py",
    ]
    device = torch.device(args.device)
    config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "torchvision.models.resnet34",
        "weights": "ResNet34_Weights.IMAGENET1K_V1",
        "L": L,
        "num_masks": NUM_MASKS,
        "num_images": total,
        "synthetic": selection is None,
        "batch_size": 1,
        "forward_dtype": "float32",
        "mobius_dtype": "float64",
        "continuous_mask_definition": "binary_mask * lambda",
        "sample_index": None if selection is None else str(args.sample_index.resolve()),
        "sample_index_sha256": None if selection is None else selection.source_sha256,
        "selection_manifest": None if selection is None else str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": None if selection is None else selection.manifest_sha256,
        "selector": {"synthetic_images": args.synthetic_images} if selection is None else selection.selector,
        "parquet_dir": None if selection is None else str(args.parquet_dir.resolve()),
        "mask_chunk": args.mask_chunk,
        "seed": args.seed,
        "device": args.device,
        "lambdas": lambdas,
        "enumerated_lambdas": enumerated_lambdas,
        "lambda_one_reference": "direct all-open forward per image",
        "lambda_one_injected_summary": (
            str(args.lambda_one_summary.resolve()) if args.skip_lambda_one_enumeration else None
        ),
        "lambda_one_injected_summary_sha256": (
            sha256_file(args.lambda_one_summary) if args.skip_lambda_one_enumeration else None
        ),
        "software_hardware": software_hardware_metadata(device),
        "cudnn_benchmark": args.cudnn_benchmark,
        "cudnn_deterministic": True,
        "tf32": False,
        "implementation_sha256": implementation_hashes(numerical_files, repo_root),
    }
    fingerprint = config_fingerprint(config)
    rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    completed_samples: set[str] = set()
    elapsed_before = 0.0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but {checkpoint_path} is missing")
        state = load_resume_checkpoint(checkpoint_path, fingerprint)
        rows = [dict(row) for row in state["rows"]]
        verification_rows = [dict(row) for row in state["verification_rows"]]
        completed_samples = {str(value) for value in state.get("completed_sample_ids", [])}
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
    elif checkpoint_path.exists():
        raise FileExistsError(f"{checkpoint_path} exists; use --resume or a new --work-dir")

    # The checkpoint is written only after all requested lambdas for an image.
    # Validate exact pair-level coverage before doing any expensive work: a
    # group with the right count but a duplicate/wrong lambda is still corrupt.
    requested_keys = {_lambda_key(value) for value in enumerated_lambdas}
    checkpoint_pairs: set[tuple[str, str]] = set()
    keys_by_sample: dict[str, set[str]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        key = _lambda_key(float(row["lambda"]))
        pair = (sample_id, key)
        if pair in checkpoint_pairs:
            raise RuntimeError(f"checkpoint contains duplicate pair {pair}")
        checkpoint_pairs.add(pair)
        keys_by_sample.setdefault(sample_id, set()).add(key)
    if any(keys != requested_keys for keys in keys_by_sample.values()):
        raise RuntimeError("checkpoint contains a partial or wrong per-image lambda group")
    verification_pairs = {
        (str(row["sample_id"]), _lambda_key(float(row["lambda"])))
        for row in verification_rows
    }
    if len(verification_pairs) != len(verification_rows) or verification_pairs != checkpoint_pairs:
        raise RuntimeError("checkpoint verification pairs differ from result pairs")
    inferred_completed = set(keys_by_sample)
    if completed_samples and inferred_completed and completed_samples != inferred_completed:
        raise RuntimeError("checkpoint completed-sample ids disagree with result rows")
    if not completed_samples:
        completed_samples = inferred_completed

    # Validate the expensive lambda=1 injection before starting any mask
    # enumeration.  This avoids discovering a split/provenance mismatch after
    # an otherwise successful multi-hour run.
    injected = None
    if args.skip_lambda_one_enumeration:
        injected = _load_lambda_one_injection(
            args.lambda_one_summary,
            expected_images=total,
            expected_source_sha256=None if selection is None else selection.source_sha256,
            expected_manifest_sha256=None if selection is None else selection.manifest_sha256,
            expected_implementation_sha256=config.get("implementation_sha256"),
        )

    model, gated_blocks, _ = build_gated_resnet34()
    model.to(device).eval()
    start = time.time()
    newly_processed = 0
    for x, label, _master_index, sample_id_batch in loader:
        sample_id = sample_id_from_batch(sample_id_batch)
        if sample_id in completed_samples:
            continue
        record = record_lookup[sample_id]
        x = x.to(device, non_blocking=True)
        image_start = time.time()
        with torch.inference_mode():
            # reset_masks is guaranteed by each enumerator/direct-scaled helper.
            reference_logits = model(x)
            if not torch.isfinite(reference_logits).all().item():
                raise FloatingPointError(f"non-finite lambda=1 reference for {sample_id}")
            top1_reference = int(reference_logits.argmax(dim=-1).item())
        image_rows: list[dict[str, Any]] = []
        image_verification: list[dict[str, Any]] = []
        for residual_scale in enumerated_lambdas:
            lambda_start = time.time()
            result, verification = _run_one_lambda(
                model, gated_blocks, x,
                residual_scale=residual_scale,
                mask_chunk=args.mask_chunk,
                top1_reference=top1_reference,
            )
            image_rows.append({
                "sample_id": sample_id,
                "selection_position": int(record.selection_position),
                "master_index": int(record.master_index),
                "shard": record.shard,
                "row": int(record.row),
                "label": int(label.item()),
                "lambda": float(residual_scale),
                "top1_reference": top1_reference,
                **result,
                "runtime_seconds": float(time.time() - lambda_start),
            })
            image_verification.append({"sample_id": sample_id, **verification})
        rows.extend(image_rows)
        verification_rows.extend(image_verification)
        completed_samples.add(sample_id)
        newly_processed += 1
        elapsed = elapsed_before + time.time() - start
        _append_progress(progress_path, {
            "event": "image_complete", "sample_id": sample_id,
            "selection_position": int(record.selection_position),
            "completed_images": len(completed_samples), "total_images": total,
            "enumerated_lambdas": enumerated_lambdas,
            "image_seconds": time.time() - image_start,
            "elapsed_seconds": elapsed,
        })
        if len(completed_samples) % args.checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_path, fingerprint=fingerprint, rows=rows,
                verification_rows=verification_rows,
                completed_sample_ids=completed_samples, elapsed_seconds=elapsed,
            )
        if newly_processed % args.log_every == 0 or len(completed_samples) == total:
            latest = image_rows[-1] if image_rows else None
            suffix = "lambda=1 injected" if latest is None else (
                f"lambda={latest['lambda']:g} kappa_h={latest['metrics']['h/kappa']:.3f} "
                f"agree={latest['top1_agree']:.0f}"
            )
            print(
                f"[r34-lambda] {len(completed_samples)}/{total} sample={sample_id} "
                f"{suffix} elapsed={elapsed:.1f}s", flush=True,
            )
        if args.stop_after_images is not None and newly_processed >= args.stop_after_images:
            _save_checkpoint(
                checkpoint_path, fingerprint=fingerprint, rows=rows,
                verification_rows=verification_rows,
                completed_sample_ids=completed_samples, elapsed_seconds=elapsed,
            )
            print(f"paused safely after {newly_processed} new images")
            return 0

    wall_seconds = elapsed_before + time.time() - start
    expected_sample_ids = set(record_lookup)
    if completed_samples != expected_sample_ids:
        raise RuntimeError("completed samples do not exactly match the selected dataset")
    _save_checkpoint(
        checkpoint_path, fingerprint=fingerprint, rows=rows,
        verification_rows=verification_rows,
        completed_sample_ids=completed_samples, elapsed_seconds=wall_seconds,
    )
    verification = _final_verification(
        rows, verification_rows,
        expected_sample_ids=expected_sample_ids,
        enumerated_lambdas=enumerated_lambdas,
    )
    if not verification["passed"]:
        raise RuntimeError(f"final verification failed: {verification}")
    per_lambda = _aggregate_rows(rows, lambdas, injected_lambda_one=injected)
    config.update({"wall_seconds": wall_seconds, "status": "complete"})
    rows.sort(key=lambda row: (int(row["selection_position"]), -float(row["lambda"])))
    _write_outputs(
        args.output_dir, config=config, per_lambda=per_lambda,
        rows=rows, verification=verification,
    )
    print(f"Saved ResNet-34 lambda sweep to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
