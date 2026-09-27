"""Step 3 -- run the 256-mask sweep on the sampled images.

Reads ``sample_index.json`` produced by step_2 (or runs against synthetic
tensors), evaluates every binary mask over the 8 BasicBlock residual
branches of a pretrained ResNet-18, computes the four reconstructions
(raw / mean / centered / Möbius) and per-image metrics, and writes a
``summary.json`` consumed by step_4.

Usage:
    python -m experiment_1.step_3_evaluate \\
        --sample-index experiment_1/sample_index.json \\
        --batch-size 4 --mask-chunk 64 \\
        --output-dir experiment_1/results

    python -m experiment_1.step_3_evaluate --synthetic --num-images 8  # smoke
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

from resnet_routes.resnet18.evaluation import aggregate, evaluate_all_masks, render_table
from resnet_routes.resnet18.data import build_loader
from resnet_routes.resnet18.model import build_gated_resnet18
from resnet_routes.resnet18.reconstruction import (
    mobius_reconstruction_error,
    per_image_metrics,
    reconstruction_centered,
    reconstruction_mean,
    reconstruction_mobius,
    reconstruction_raw,
)
from resnet_routes.resnet18.mobius import enumerate_masks, full_index, mobius_coefficients


L = 8
NUM_MASKS = 1 << L


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample-index", type=Path,
                     help="JSON written by step_2_sample.py")
    src.add_argument("--synthetic", action="store_true",
                     help="run on deterministic random tensors instead")
    p.add_argument("--num-images", type=int, default=64,
                   help="(synthetic only) number of random images to fabricate")
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
    p.add_argument("--save-logits", action="store_true",
                   help="persist (B, 256, 1000) logits and Möbius coeffs")
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    full_idx = full_index(L)

    loader, total = build_loader(
        sample_index=str(args.sample_index) if args.sample_index else None,
        synthetic_n=args.num_images if args.synthetic else None,
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
        parquet_dir=args.parquet_dir,
    )
    print(f"[step_3] device={device} images={total} masks/image={NUM_MASKS} "
          f"mask_chunk={args.mask_chunk} batch_size={args.batch_size}")

    metric_logs: Dict[str, List[float]] = {}
    saved_logits = []
    n_done = 0
    t0 = time.time()
    for batch_idx, (x, _y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        h_masks = evaluate_all_masks(model, gated_blocks, x, masks, args.mask_chunk)
        h_full = h_masks[:, full_idx, :]
        top1_full = h_full.argmax(dim=-1)

        deltas = mobius_coefficients(h_masks)
        recons = {
            "raw": reconstruction_raw(h_masks),
            "mean": reconstruction_mean(h_masks),
            "ctr": reconstruction_centered(h_masks),
            "mob": reconstruction_mobius(deltas),
        }
        for key, A in recons.items():
            for name, vals in per_image_metrics(A, h_full, top1_full).items():
                metric_logs.setdefault(f"{key}/{name}", []).extend(vals.tolist())
        mob_err = mobius_reconstruction_error(h_full, recons["mob"])
        metric_logs.setdefault("mobius_recon_err", []).extend(mob_err.tolist())

        if args.save_logits:
            saved_logits.append({"h_masks": h_masks.cpu(), "deltas": deltas.cpu()})

        n_done += x.shape[0]
        if (batch_idx + 1) % args.log_every == 0 or n_done == total:
            print(f"[step_3] {n_done}/{total}  "
                  f"raw_norm_ratio={metric_logs['raw/norm_ratio'][-1]:.3g}  "
                  f"mob_err={metric_logs['mobius_recon_err'][-1]:.2e}  "
                  f"({time.time() - t0:.1f}s)")

    agg = aggregate(metric_logs)

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
        },
        "metrics": agg,
        "raw_logs": metric_logs,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "table_1.md").write_text(render_table(agg) + "\n")
    if args.save_logits:
        torch.save(saved_logits, args.output_dir / "logits.pt")

    print("\n=== Table 1 (preview) ===")
    print(render_table(agg))
    print(f"\nSaved summary.json + table_1.md to {args.output_dir}/")


if __name__ == "__main__":
    main()
