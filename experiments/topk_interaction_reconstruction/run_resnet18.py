"""Step 1 -- top-K sparse residual-expert reconstruction sweep.

Per image:
  * 256-mask sweep -> deltas_h (B, 256, 1000), deltas_v (B, 256).
  * Magnitude rankings: scalar |Delta_S v_x|, vector ||Delta_S h_x||_2,
    over the 255 non-empty residual subsets.
  * Cumulative top-K reconstruction for K = 1..255 using the sorted axis
    (memory: B * 255 * 1000 floats per metric).
  * Baselines on the same K grid:
      - random-K reconstructions averaged over multiple seeds;
      - low-order truncation S: |S| <= r for r = 1..L (point estimates);
      - order-matched random matched to the top-K composition (point
        estimates at a sparse K grid).
  * Effective sparse expert counts K_tau (Err) and K_eta (mass).
  * Order composition Pi_K(k) of the selected magnitude top-K experts.

Aggregates mean +/- standard error across N images and writes
``results/summary.json`` plus ``results/table_3.md``.

Usage:
    python -m experiment_3.step_1_evaluate \\
        --sample-index experiment_1/sample_index.json \\
        --batch-size 8 --mask-chunk 256 --num-workers 8 --seed 0 \\
        --num-random-seeds 5 \\
        --output-dir experiment_3/results

    python -m experiment_3.step_1_evaluate --synthetic --num-images 4   # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
    gather_along,
    k_effective,
    low_order_indices_per_r,
    order_composition_curve,
    order_matched_indices,
    random_permutation,
    residual_indices,
    scalar_magnitudes,
    scalar_metrics,
    vector_magnitudes,
    vector_metrics,
)


L = 8
NUM_MASKS = 1 << L
NUM_RES = NUM_MASKS - 1                    # 255

# K grid for sparse-K reporting: powers of two plus the full set.
K_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 255)
P_GRID = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)

THRESHOLDS_ERR = (0.10, 0.05)
THRESHOLDS_MASS = (0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample-index", type=Path,
                     help="JSON written by experiment_1.step_2_sample")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mask-chunk", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-random-seeds", type=int, default=5,
                   help="number of random-K baselines to average (>=1)")
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


def _mean_se(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().to(torch.float64).reshape(-1)
    n = max(t.numel(), 1)
    mean = t.mean().item()
    se = (t.std(unbiased=True).item() / (n ** 0.5)) if n > 1 else 0.0
    return {"mean": mean, "stderr": se, "n": n}


def _stack_mean_se(samples: List[torch.Tensor]) -> Dict[str, list]:
    big = torch.cat(samples, dim=0).to(torch.float64)
    n = big.shape[0]
    means = big.mean(dim=0).tolist()
    if n > 1:
        se = (big.std(dim=0, unbiased=True) / (n ** 0.5)).tolist()
    else:
        se = [0.0] * big.shape[1]
    return {"mean": means, "stderr": se, "n": n}


def _empty_buf() -> Dict[str, List[torch.Tensor]]:
    return {
        "mag_v/Gamma": [], "mag_v/err_v": [],
        "mag_h/Gamma": [], "mag_h/err_h": [],
        "mag_h/cos": [], "mag_h/agree": [],
        "mag_h/margin": [], "mag_h/margin_ratio": [],
        "rand/err_h": [], "rand/cos": [], "rand/agree": [],
        "lowr/err_h": [], "lowr/cos": [], "lowr/agree": [],
        "ordmatch/err_h": [], "ordmatch/cos": [], "ordmatch/agree": [],
        "K_err": [], "K_mass": [],          # (B, len(THRESHOLDS_*))
        "Pi_K": [],                         # (B, 255, L+1) order composition
        "baseline/err_h": [],               # (B,) reconstruction with K=0
        "baseline/cos": [], "baseline/agree": [],
    }


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
    orders_residual = orders_full[1:].cpu()        # (255,)
    res_idx = residual_indices(L)                  # (255,) -> mask indices 1..255

    low_r = low_order_indices_per_r(orders_residual, L)

    loader, total = build_loader(
        sample_index=str(args.sample_index) if args.sample_index else None,
        synthetic_n=args.num_images if args.synthetic else None,
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
        parquet_dir=args.parquet_dir,
    )
    print(f"[step_1] device={device} images={total} masks/image={NUM_MASKS} "
          f"random_seeds={args.num_random_seeds} K_grid={K_GRID}")

    rand_gen = torch.Generator().manual_seed(args.seed + 12345)
    buf = _empty_buf()

    n_done = 0
    t0 = time.time()
    for batch_idx, (x, _y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        h_masks = evaluate_all_masks(model, gated_blocks, x, masks, args.mask_chunk)
        h_full = h_masks[:, full_idx, :]
        top1_full = h_full.argmax(dim=-1)

        deltas_h = mobius_coefficients(h_masks)               # (B, 256, C)
        deltas_v = scalar_mobius_predicted(h_masks, top1_full)  # (B, 256)

        baseline_h = deltas_h[:, 0, :]                        # (B, C)
        baseline_v = deltas_v[:, 0]                           # (B,)
        residual_h = deltas_h[:, 1:, :]                       # (B, 255, C)
        residual_v = deltas_v[:, 1:]                          # (B, 255)

        # Magnitudes & orderings
        mags_v = scalar_magnitudes(deltas_v)                  # (B, 255)
        mags_h = vector_magnitudes(deltas_h)                  # (B, 255)
        sort_v = descending_order(mags_v)                     # (B, 255)
        sort_h = descending_order(mags_h)

        # Captured mass curves
        Gamma_v = captured_mass(mags_v, sort_v)               # (B, 255)
        Gamma_h = captured_mass(mags_h, sort_h)
        buf["mag_v/Gamma"].append(Gamma_v.cpu())
        buf["mag_h/Gamma"].append(Gamma_h.cpu())

        # Cumulative reconstructions for the magnitude rankings (full curves)
        recon_v_curve = cumulative_topk_reconstruction(
            residual_v, sort_v, baseline_v
        )                                                      # (B, 255)
        recon_h_curve = cumulative_topk_reconstruction(
            residual_h, sort_h, baseline_h
        )                                                      # (B, 255, C)
        v_full = recon_v_curve.new_tensor(0)  # placeholder; overwritten below
        # Recover v_full from full mask: v_x(1_[L]) = h_full[:, top1_full]
        v_full = torch.gather(h_full, dim=-1, index=top1_full.unsqueeze(-1)).squeeze(-1)
        sm = scalar_metrics(recon_v_curve, v_full)
        vm = vector_metrics(recon_h_curve, h_full, top1_full)
        buf["mag_v/err_v"].append(sm["err_v"].cpu())
        for k in ("err_h", "cos", "agree", "margin", "margin_ratio"):
            buf[f"mag_h/{k}"].append(vm[k].cpu())

        # K_eff
        K_err = torch.stack(
            [k_effective(vm["err_h"], thr, "err") for thr in THRESHOLDS_ERR],
            dim=-1,
        )
        K_mass = torch.stack(
            [k_effective(Gamma_h, thr, "mass") for thr in THRESHOLDS_MASS],
            dim=-1,
        )
        buf["K_err"].append(K_err.cpu())
        buf["K_mass"].append(K_mass.cpu())

        # Pi_K(k) -- order composition of top-K vector ranking
        pi = order_composition_curve(sort_h, orders_residual, L)   # (B, 255, L+1)
        buf["Pi_K"].append(pi.cpu())

        # Baseline-only (K=0) reconstruction reference
        diff0 = h_full - baseline_h
        err0 = torch.linalg.vector_norm(diff0, dim=-1) / (
            torch.linalg.vector_norm(h_full, dim=-1) + 1e-12
        )
        cos0 = (baseline_h * h_full).sum(dim=-1) / (
            torch.linalg.vector_norm(baseline_h, dim=-1)
            * torch.linalg.vector_norm(h_full, dim=-1) + 1e-12
        )
        agree0 = (baseline_h.argmax(dim=-1) == top1_full).float()
        buf["baseline/err_h"].append(err0.cpu())
        buf["baseline/cos"].append(cos0.cpu())
        buf["baseline/agree"].append(agree0.cpu())

        # ----- Random-K baseline (full curves, averaged over seeds) -----
        rand_err_acc = torch.zeros_like(vm["err_h"])
        rand_cos_acc = torch.zeros_like(vm["cos"])
        rand_agree_acc = torch.zeros_like(vm["agree"])
        for _ in range(max(args.num_random_seeds, 1)):
            perm = random_permutation(x.shape[0], NUM_RES, rand_gen).to(device)
            rec = cumulative_topk_reconstruction(residual_h, perm, baseline_h)
            mr = vector_metrics(rec, h_full, top1_full)
            rand_err_acc += mr["err_h"]
            rand_cos_acc += mr["cos"]
            rand_agree_acc += mr["agree"]
        denom = max(args.num_random_seeds, 1)
        buf["rand/err_h"].append((rand_err_acc / denom).cpu())
        buf["rand/cos"].append((rand_cos_acc / denom).cpu())
        buf["rand/agree"].append((rand_agree_acc / denom).cpu())

        # ----- Low-order truncation baseline (point estimates per r) -----
        lowr_err = torch.zeros(x.shape[0], L, device=device)
        lowr_cos = torch.zeros_like(lowr_err)
        lowr_agree = torch.zeros_like(lowr_err)
        for r in range(1, L + 1):
            res_idx_r = low_r[r].to(device)             # 1D long tensor
            sub = residual_h.index_select(1, res_idx_r)
            recon = baseline_h + sub.sum(dim=1)
            diff = h_full - recon
            err = torch.linalg.vector_norm(diff, dim=-1) / (
                torch.linalg.vector_norm(h_full, dim=-1) + 1e-12
            )
            inner = (recon * h_full).sum(dim=-1)
            cos = inner / (
                torch.linalg.vector_norm(recon, dim=-1)
                * torch.linalg.vector_norm(h_full, dim=-1) + 1e-12
            )
            agree = (recon.argmax(dim=-1) == top1_full).float()
            lowr_err[:, r - 1] = err
            lowr_cos[:, r - 1] = cos
            lowr_agree[:, r - 1] = agree
        buf["lowr/err_h"].append(lowr_err.cpu())
        buf["lowr/cos"].append(lowr_cos.cpu())
        buf["lowr/agree"].append(lowr_agree.cpu())

        # ----- Order-matched random baseline (point estimates at K_GRID) -----
        sort_h_cpu = sort_h.cpu()
        ordmatch_idx = order_matched_indices(
            sort_h_cpu, orders_residual, K_GRID, rand_gen, L
        )
        ordm_err = torch.zeros(x.shape[0], len(K_GRID), device=device)
        ordm_cos = torch.zeros_like(ordm_err)
        ordm_agree = torch.zeros_like(ordm_err)
        for ki, K in enumerate(K_GRID):
            idx = ordmatch_idx[K].to(device)
            idx_b = idx.unsqueeze(-1).expand(-1, -1, residual_h.shape[-1])
            sub = torch.gather(residual_h, dim=1, index=idx_b)
            recon = baseline_h + sub.sum(dim=1)
            diff = h_full - recon
            err = torch.linalg.vector_norm(diff, dim=-1) / (
                torch.linalg.vector_norm(h_full, dim=-1) + 1e-12
            )
            inner = (recon * h_full).sum(dim=-1)
            cos = inner / (
                torch.linalg.vector_norm(recon, dim=-1)
                * torch.linalg.vector_norm(h_full, dim=-1) + 1e-12
            )
            agree = (recon.argmax(dim=-1) == top1_full).float()
            ordm_err[:, ki] = err
            ordm_cos[:, ki] = cos
            ordm_agree[:, ki] = agree
        buf["ordmatch/err_h"].append(ordm_err.cpu())
        buf["ordmatch/cos"].append(ordm_cos.cpu())
        buf["ordmatch/agree"].append(ordm_agree.cpu())

        n_done += x.shape[0]
        if (batch_idx + 1) % args.log_every == 0 or n_done == total:
            agree_at_K = vm["agree"]                          # (B, 255)
            agree_at_8 = agree_at_K[:, 7].mean().item()
            agree_at_32 = agree_at_K[:, 31].mean().item()
            print(f"[step_1] {n_done}/{total}  "
                  f"agree@K=8={agree_at_8:.3f}  agree@K=32={agree_at_32:.3f}  "
                  f"({time.time() - t0:.1f}s)")

    # -------------------- reduce --------------------
    metrics: Dict[str, dict] = {}
    for key, samples in buf.items():
        if key in ("baseline/err_h", "baseline/cos", "baseline/agree"):
            metrics[key] = _mean_se(torch.cat(samples, dim=0))
            continue
        big = torch.cat(samples, dim=0).to(torch.float64)
        n = big.shape[0]
        if big.dim() == 2:
            mean = big.mean(dim=0).tolist()
            if n > 1:
                se = (big.std(dim=0, unbiased=True) / (n ** 0.5)).tolist()
            else:
                se = [0.0] * big.shape[1]
            metrics[key] = {"mean": mean, "stderr": se, "n": n}
        elif big.dim() == 3:
            mean = big.mean(dim=0).tolist()
            if n > 1:
                se = (big.std(dim=0, unbiased=True) / (n ** 0.5)).tolist()
            else:
                se = [[0.0] * big.shape[2] for _ in range(big.shape[1])]
            metrics[key] = {"mean": mean, "stderr": se, "n": n}
        else:
            metrics[key] = _mean_se(big)

    summary = {
        "config": {
            "sample_index": str(args.sample_index) if args.sample_index else None,
            "synthetic": args.synthetic,
            "num_images": total,
            "L": L,
            "num_residual_subsets": NUM_RES,
            "K_grid": list(K_GRID),
            "p_grid": list(P_GRID),
            "thresholds_err": list(THRESHOLDS_ERR),
            "thresholds_mass": list(THRESHOLDS_MASS),
            "num_random_seeds": args.num_random_seeds,
            "batch_size": args.batch_size,
            "mask_chunk": args.mask_chunk,
            "seed": args.seed,
            "device": str(device),
            "wall_seconds": round(time.time() - t0, 2),
        },
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    table = render_table(metrics, total)
    (args.output_dir / "table_3.md").write_text(table)
    print("\n=== Table 3 (preview) ===\n" + table)
    print(f"\nSaved summary.json + table_3.md to {args.output_dir}/")


def render_table(metrics: Dict[str, dict], n: int) -> str:
    rows = [
        f"# Experiment 3 -- top-K sparse residual-expert reconstruction (N = {n})",
        "",
        "## Magnitude top-K (vector ranking)",
        "",
        "| K | $\\Gamma_K^h$ | $\\mathrm{Err}_K^h$ | cos | top-1 agree | margin ratio |",
        "| -: | :-- | :-- | :-- | :-- | :-- |",
    ]
    g_means = metrics["mag_h/Gamma"]["mean"]
    g_se = metrics["mag_h/Gamma"]["stderr"]
    e_means = metrics["mag_h/err_h"]["mean"]
    e_se = metrics["mag_h/err_h"]["stderr"]
    c_means = metrics["mag_h/cos"]["mean"]
    a_means = metrics["mag_h/agree"]["mean"]
    a_se = metrics["mag_h/agree"]["stderr"]
    mr_means = metrics["mag_h/margin_ratio"]["mean"]
    for K in K_GRID:
        i = K - 1
        rows.append(
            f"| {K} | {g_means[i]:.4f} ± {g_se[i]:.4f} | "
            f"{e_means[i]:.4g} ± {e_se[i]:.2g} | {c_means[i]:.4f} | "
            f"{a_means[i]:.4f} ± {a_se[i]:.4f} | {mr_means[i]:.4f} |"
        )

    rows += [
        "",
        "## Effective sparse expert counts (mean over images)",
        "",
        "| threshold | mean K | stderr |",
        "| :-- | :-- | :-- |",
    ]
    K_err_m = metrics["K_err"]["mean"]
    K_err_s = metrics["K_err"]["stderr"]
    for ti, thr in enumerate(THRESHOLDS_ERR):
        rows.append(f"| Err <= {thr} | {K_err_m[ti]:.2f} | {K_err_s[ti]:.2f} |")
    K_mass_m = metrics["K_mass"]["mean"]
    K_mass_s = metrics["K_mass"]["stderr"]
    for ti, thr in enumerate(THRESHOLDS_MASS):
        rows.append(f"| mass >= {thr} | {K_mass_m[ti]:.2f} | {K_mass_s[ti]:.2f} |")

    rows += [
        "",
        "## Baseline reference (K = 0, shortcut only)",
        "",
        f"Err = {metrics['baseline/err_h']['mean']:.4g} ± {metrics['baseline/err_h']['stderr']:.2g}, "
        f"cos = {metrics['baseline/cos']['mean']:.4f}, "
        f"top-1 agree = {metrics['baseline/agree']['mean']:.4f}",
        "",
        "## Comparison vs random / low-order / order-matched at K in K_GRID",
        "",
        "| K | top-1 (mag) | top-1 (random) | top-1 (order-match) |",
        "| -: | :-- | :-- | :-- |",
    ]
    rand_a = metrics["rand/agree"]["mean"]
    ordm_a = metrics["ordmatch/agree"]["mean"]
    for ki, K in enumerate(K_GRID):
        rows.append(
            f"| {K} | {a_means[K - 1]:.4f} | {rand_a[K - 1]:.4f} | {ordm_a[ki]:.4f} |"
        )

    rows += [
        "",
        "## Low-order truncation baseline (S: |S| <= r)",
        "",
        "| r | Err | cos | top-1 agree |",
        "| -: | :-- | :-- | :-- |",
    ]
    le = metrics["lowr/err_h"]["mean"]
    lc = metrics["lowr/cos"]["mean"]
    la = metrics["lowr/agree"]["mean"]
    for r in range(1, L + 1):
        rows.append(f"| {r} | {le[r - 1]:.4g} | {lc[r - 1]:.4f} | {la[r - 1]:.4f} |")

    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
