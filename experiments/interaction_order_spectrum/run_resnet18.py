"""Step 1 -- Möbius interaction order spectrum sweep.

Reuses ``experiment_1`` for the gated ResNet-18, 256-mask enumeration, and
parquet data path.  Per image, computes:

* scalar Möbius coefficients of the predicted-class logit ``v_x(m) = h_{x, hat_y(x)}(m)``;
* vector Möbius coefficients of the full logit ``h_x(m) in R^1000``;
* per-order energies E_k^v, E_k^h (k = 0..L);
* the residual-only / baseline-inclusive normalized spectra, average
  per-interaction energy M_k, cumulative C_<=K, effective order kappa,
  and high-order tail T_>K.

Aggregates mean +/- standard error across N images and writes
``results/summary.json`` plus a markdown summary ``results/table_2.md``.

Usage:
    python -m experiment_2.step_1_evaluate \\
        --sample-index experiment_1/sample_index.json \\
        --batch-size 8 --mask-chunk 256 --num-workers 8 --seed 0 \\
        --output-dir experiment_2/results

    python -m experiment_2.step_1_evaluate --synthetic --num-images 8  # smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from resnet_routes.resnet18.evaluation import evaluate_all_masks
from resnet_routes.resnet18.data import build_loader
from resnet_routes.resnet18.model import build_gated_resnet18
from resnet_routes.resnet18.mobius import enumerate_masks, full_index, mobius_coefficients
from resnet_routes.resnet18.spectrum import (
    derived_metrics,
    order_energies_scalar,
    order_energies_vector,
    order_indices,
    scalar_mobius_predicted,
)


L = 8
NUM_MASKS = 1 << L


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample-index", type=Path,
                     help="JSON written by experiment_1.step_2_sample")
    src.add_argument("--synthetic", action="store_true",
                     help="run on deterministic random tensors (smoke test)")
    p.add_argument("--num-images", type=int, default=64,
                   help="(synthetic only) number of random images")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mask-chunk", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--parquet-dir", type=Path, default=None,
                   help="directory containing ImageNet parquet shards")
    p.add_argument("--no-pretrained", action="store_true",
                   help="do not download/load ImageNet weights (smoke tests only)")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "generated/resnet18")
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mean_se(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().to(torch.float64).reshape(-1)
    n = max(t.numel(), 1)
    mean = t.mean().item()
    se = (t.std(unbiased=True).item() / (n ** 0.5)) if n > 1 else 0.0
    return {"mean": mean, "stderr": se, "n": n}


def _stack_mean_se(samples: List[torch.Tensor]) -> Dict[str, list]:
    """Stack per-batch tensors of shape (B, K) into one (N, K) and reduce per K."""
    big = torch.cat(samples, dim=0).to(torch.float64)
    n = big.shape[0]
    means = big.mean(dim=0).tolist()
    if n > 1:
        se = (big.std(dim=0, unbiased=True) / (n ** 0.5)).tolist()
    else:
        se = [0.0] * big.shape[1]
    return {"mean": means, "stderr": se, "n": n}


def render_table(summary_metrics: Dict[str, dict], readout: str) -> str:
    """Markdown table -- order-spectrum highlights for one readout (v or h)."""
    E_tilde = summary_metrics[f"{readout}/E_tilde"]
    cum = summary_metrics[f"{readout}/cum"]
    tail = summary_metrics[f"{readout}/tail"]
    kappa = summary_metrics[f"{readout}/kappa"]
    M = summary_metrics[f"{readout}/M"]

    rows = [
        f"### Readout: {'predicted-class scalar' if readout == 'v' else 'logit vector'}",
        "",
        "| k | $\\widetilde{E}_k$ | $M_k$ (avg per-interaction energy) | $C_{\\le k}$ | $T_{>k}$ |",
        "| -: | :--- | :--- | :--- | :--- |",
    ]
    for i in range(L):
        k = i + 1
        rows.append(
            f"| {k} | {E_tilde['mean'][i]:.4f} ± {E_tilde['stderr'][i]:.4f} | "
            f"{M['mean'][k]:.4g} ± {M['stderr'][k]:.4g} | "
            f"{cum['mean'][i]:.4f} ± {cum['stderr'][i]:.4f} | "
            f"{tail['mean'][i]:.4f} ± {tail['stderr'][i]:.4f} |"
        )
    rows += [
        "",
        f"Effective interaction order  $\\kappa = {kappa['mean']:.4f} \\pm {kappa['stderr']:.4f}$",
        f"$M_0$ (shortcut baseline energy)  $= {M['mean'][0]:.4g} \\pm {M['stderr'][0]:.4g}$",
    ]
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    model, gated_blocks = build_gated_resnet18(pretrained=not args.no_pretrained)
    model.to(device).eval()
    masks = enumerate_masks(L, device=device)
    orders = order_indices(L, device=device)
    full_idx = full_index(L)

    loader, total = build_loader(
        sample_index=str(args.sample_index) if args.sample_index else None,
        synthetic_n=args.num_images if args.synthetic else None,
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
        parquet_dir=args.parquet_dir,
    )
    print(f"[step_1] device={device} images={total} masks/image={NUM_MASKS} "
          f"mask_chunk={args.mask_chunk} batch_size={args.batch_size}")

    # Per-image accumulators -- each batch produces (B, ...) tensors that we
    # concatenate across the dataset and reduce at the end.
    buf: Dict[str, List[torch.Tensor]] = {
        "v/E": [], "h/E": [],
        "v/E_tilde": [], "h/E_tilde": [],
        "v/E_bar": [], "h/E_bar": [],
        "v/M": [], "h/M": [],
        "v/cum": [], "h/cum": [],
        "v/tail": [], "h/tail": [],
        "v/kappa": [], "h/kappa": [],
    }

    n_done = 0
    t0 = time.time()
    for batch_idx, (x, _y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        h_masks = evaluate_all_masks(model, gated_blocks, x, masks, args.mask_chunk)
        h_full = h_masks[:, full_idx, :]
        top1_full = h_full.argmax(dim=-1)

        deltas_h = mobius_coefficients(h_masks)              # (B, M, C)
        deltas_v = scalar_mobius_predicted(h_masks, top1_full)  # (B, M)

        E_v = order_energies_scalar(deltas_v, orders)        # (B, L+1)
        E_h = order_energies_vector(deltas_h, orders)        # (B, L+1)

        for tag, E in (("v", E_v), ("h", E_h)):
            d = derived_metrics(E)
            buf[f"{tag}/E"].append(E.cpu())
            buf[f"{tag}/E_tilde"].append(d["E_tilde"].cpu())
            buf[f"{tag}/E_bar"].append(d["E_bar"].cpu())
            buf[f"{tag}/M"].append(d["M"].cpu())
            buf[f"{tag}/cum"].append(d["cum"].cpu())
            buf[f"{tag}/tail"].append(d["tail"].cpu())
            buf[f"{tag}/kappa"].append(d["kappa"].cpu())

        n_done += x.shape[0]
        if (batch_idx + 1) % args.log_every == 0 or n_done == total:
            print(f"[step_1] {n_done}/{total}  "
                  f"kappa_v_last={buf['v/kappa'][-1].mean().item():.3f}  "
                  f"kappa_h_last={buf['h/kappa'][-1].mean().item():.3f}  "
                  f"({time.time() - t0:.1f}s)")

    # Reduce: stacked tensors -> mean +/- standard error per coordinate.
    metrics: Dict[str, dict] = {}
    for key, samples in buf.items():
        if key.endswith("/kappa"):
            metrics[key] = _mean_se(torch.cat(samples, dim=0))
        else:
            metrics[key] = _stack_mean_se(samples)

    summary = {
        "config": {
            "sample_index": str(args.sample_index) if args.sample_index else None,
            "synthetic": args.synthetic,
            "num_images": total,
            "L": L,
            "num_masks": NUM_MASKS,
            "batch_size": args.batch_size,
            "mask_chunk": args.mask_chunk,
            "seed": args.seed,
            "device": str(device),
            "wall_seconds": round(time.time() - t0, 2),
        },
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    table = (
        "# Experiment 2 -- Möbius interaction order spectrum\n\n"
        + render_table(metrics, "v") + "\n\n"
        + render_table(metrics, "h") + "\n"
    )
    (args.output_dir / "table_2.md").write_text(table)

    print("\n=== Table 2 (preview) ===")
    print(table)
    print(f"\nSaved summary.json + table_2.md to {args.output_dir}/")


if __name__ == "__main__":
    main()
