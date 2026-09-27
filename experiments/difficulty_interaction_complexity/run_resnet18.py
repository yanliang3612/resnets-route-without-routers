"""Step 1 -- per-image difficulty + complexity dump.

Per image:
  * 256-mask sweep -> deltas_h (B, 256, C), deltas_v (B, 256).
  * Difficulty scores from the full-mask logits (label-aware: loss / wrong
    / true-class margin; label-free: confidence / pred margin).
  * Vector and scalar magnitudes over the 255 non-empty residual subsets.
  * The six complexity measures of the plan plus the cumulative top-K
    reconstruction curve (so K_tau / K_eta can be derived).

Outputs ``per_image.pt`` (per-image tensors only -- not the full 256-mask
logits, so this stays small).  Plus ``config.json`` for reproducibility.

Usage:
    python -m experiment_5.step_1_evaluate \\
        --sample-index experiment_1/sample_index_val.json \\
        --batch-size 8 --mask-chunk 256 --num-workers 8 --seed 0 \\
        --output-dir experiment_5/results

    python -m experiment_5.step_1_evaluate --synthetic --num-images 8   # smoke
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
from resnet_routes.resnet18.spectrum import order_indices, scalar_mobius_predicted
from resnet_routes.resnet18.topk import (
    captured_mass,
    cumulative_topk_reconstruction,
    descending_order,
    scalar_magnitudes,
    vector_magnitudes,
)

from resnet_routes.resnet18.complexity import (
    all_complexity_measures,
    k_eff_thresholds,
)
from resnet_routes.resnet18.difficulty import all_difficulty_scores


L = 8
NUM_MASKS = 1 << L
NUM_RES = NUM_MASKS - 1                 # 255

THRESHOLDS_ERR = (0.10, 0.05)           # K_tau on Err_K^h
THRESHOLDS_MASS = (0.90, 0.95, 0.99)    # K_eta on Gamma_K^h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample-index", type=Path,
                     help="JSON written by experiment_1.step_2_sample (val per-class)")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--num-images", type=int, default=64)
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
    p.add_argument("--log-every", type=int, default=50)
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
    orders_full = order_indices(L, device=device)
    orders_residual = orders_full[1:].long()             # (255,)

    loader, total = build_loader(
        sample_index=str(args.sample_index) if args.sample_index else None,
        synthetic_n=args.num_images if args.synthetic else None,
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
        parquet_dir=args.parquet_dir,
    )
    print(f"[step_1] device={device} images={total} masks/image={NUM_MASKS} "
          f"batch_size={args.batch_size} mask_chunk={args.mask_chunk}")

    buf: Dict[str, List[torch.Tensor]] = {
        # difficulty
        "loss": [], "wrong": [], "true_margin": [],
        "confidence": [], "pred_margin": [],
        "labels": [], "pseudo_labels": [],
        # complexity (vector ranking)
        "h/N_eff": [], "h/N_ent": [], "h/kbar": [], "h/kappa": [],
        "h/C_le_3": [], "h/T_gt_3": [],
        # complexity (scalar ranking)
        "v/N_eff": [], "v/N_ent": [], "v/kbar": [], "v/kappa": [],
        "v/C_le_3": [], "v/T_gt_3": [],
        # K_eff thresholds (vector ranking only -- err & mass)
        "K_err_010": [], "K_err_005": [],
        "K_mass_090": [], "K_mass_095": [], "K_mass_099": [],
    }

    n_done = 0
    t0 = time.time()
    has_labels = args.sample_index is not None  # synthetic still has fake labels but we treat them as available
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.long().to(device, non_blocking=True)
        h_masks = evaluate_all_masks(model, gated_blocks, x, masks, args.mask_chunk)
        h_full = h_masks[:, full_idx, :]
        top1_full = h_full.argmax(dim=-1)

        labels_for_diff = y if has_labels else None
        diff_scores = all_difficulty_scores(h_full, labels_for_diff)
        for k, v in diff_scores.items():
            buf[k].append(v.cpu())
        buf["labels"].append(y.cpu())
        buf["pseudo_labels"].append(top1_full.cpu())

        deltas_h = mobius_coefficients(h_masks)              # (B, 256, C)
        deltas_v = scalar_mobius_predicted(h_masks, top1_full)  # (B, 256)
        residual_h = deltas_h[:, 1:, :]
        residual_v = deltas_v[:, 1:]

        mags_h = vector_magnitudes(deltas_h)                  # (B, 255)
        mags_v = scalar_magnitudes(deltas_v)                  # (B, 255)

        comp_h = all_complexity_measures(mags_h, orders_residual)
        comp_v = all_complexity_measures(mags_v, orders_residual)
        for k, v in comp_h.items():
            buf[f"h/{k}".replace("_le_K", "_le_3").replace("_gt_K", "_gt_3")].append(v.cpu())
        for k, v in comp_v.items():
            buf[f"v/{k}".replace("_le_K", "_le_3").replace("_gt_K", "_gt_3")].append(v.cpu())

        # Cumulative reconstructions for K_eff thresholds (vector ranking)
        sort_h = descending_order(mags_h)                     # (B, 255)
        baseline_h = deltas_h[:, 0, :]
        recon_h_curve = cumulative_topk_reconstruction(residual_h, sort_h, baseline_h)
        diff_curve = h_full.unsqueeze(1) - recon_h_curve
        err_curve = torch.linalg.vector_norm(diff_curve, dim=-1) / (
            torch.linalg.vector_norm(h_full, dim=-1).unsqueeze(-1) + 1e-12
        )
        Gamma_h = captured_mass(mags_h, sort_h)
        for thr, key in zip(THRESHOLDS_ERR, ("K_err_010", "K_err_005")):
            buf[key].append(k_eff_thresholds(err_curve, thr, "err").cpu())
        for thr, key in zip(THRESHOLDS_MASS,
                            ("K_mass_090", "K_mass_095", "K_mass_099")):
            buf[key].append(k_eff_thresholds(Gamma_h, thr, "mass").cpu())

        n_done += x.shape[0]
        if (batch_idx + 1) % args.log_every == 0 or n_done == total:
            print(f"[step_1] {n_done}/{total}  ({time.time() - t0:.1f}s)")

    out: Dict[str, torch.Tensor] = {k: torch.cat(v, dim=0) for k, v in buf.items() if v}
    sig_path = args.output_dir / "per_image.pt"
    torch.save(out, sig_path)

    cfg = {
        "sample_index": str(args.sample_index) if args.sample_index else None,
        "synthetic": args.synthetic,
        "num_images": total,
        "L": L,
        "num_residual_subsets": NUM_RES,
        "thresholds_err": list(THRESHOLDS_ERR),
        "thresholds_mass": list(THRESHOLDS_MASS),
        "K_low_default": 3,
        "batch_size": args.batch_size,
        "mask_chunk": args.mask_chunk,
        "seed": args.seed,
        "device": str(device),
        "wall_seconds": round(time.time() - t0, 2),
    }
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    full_acc = (out["pseudo_labels"] == out["labels"]).float().mean().item()
    print(f"\nSaved per_image.pt to {sig_path}  (full-mask top-1 = {full_acc:.4f})")
    print("Run experiment_5.step_2_analyze to compute correlations and gaps.")


if __name__ == "__main__":
    main()
