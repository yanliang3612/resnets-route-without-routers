"""Step 3 -- inference-time residual scaling sweep (Experiments 3A + 3B).

Scales every residual branch by a fixed factor ``lambda`` at inference time
and re-measures the Möbius interaction order spectrum.  Combines the broad
sweep of plan 3A and the performance-controlled narrow sweep of 3B into a
single pass through the dataset (data IO would otherwise dominate).

For each ``lambda``:
  * The 256 binary masks are scaled component-wise by ``lambda`` so that
    ``h_x^{(lambda)}(1_S) = forward(x; gate_l = lambda for l in S, 0 else)``.
  * Scalar (predicted-class) and vector (logit) Möbius coefficients are
    recomputed over the new (B, 256, C) tensor.
  * All plan metrics are recomputed: E_k, E_tilde, E_bar, M_k, C_<=K, T_>K, kappa.
  * The top-1 prediction at the full mask under ``lambda`` is compared against
    the lambda=1.0 full-mask prediction (per-image agreement, plan eq. 32).

Usage:
    python -m experiment_2.step_3_lambda_sweep \\
        --sample-index experiment_1/sample_index.json \\
        --batch-size 8 --mask-chunk 256 --num-workers 8 --seed 0 \\
        --output-dir experiment_2/results_lambda

    python -m experiment_2.step_3_lambda_sweep --synthetic --num-images 4  # smoke
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

# Broad (3A) + narrow (3B), deduplicated and sorted descending.
DEFAULT_LAMBDAS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.5, 0.25, 0.1]


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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--parquet-dir", type=Path, default=None,
                   help="directory containing ImageNet parquet shards")
    p.add_argument("--no-pretrained", action="store_true",
                   help="do not download/load ImageNet weights (smoke tests only)")
    p.add_argument("--lambdas", type=str, default=",".join(str(x) for x in DEFAULT_LAMBDAS),
                   help="comma-separated list; 1.0 must be present (used as reference).")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "generated/resnet18")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def parse_lambdas(spec: str) -> List[float]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    if 1.0 not in out:
        raise ValueError("--lambdas must include 1.0 as the reference run")
    # Largest first so lambda=1.0 is computed first in each batch -- top-1 at
    # lambda=1.0 is the reference for top1_agree at smaller lambdas.
    out = sorted(set(out), reverse=True)
    return out


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
        "v/E": [], "h/E": [],
        "v/E_tilde": [], "h/E_tilde": [],
        "v/E_bar": [], "h/E_bar": [],
        "v/M": [], "h/M": [],
        "v/cum": [], "h/cum": [],
        "v/tail": [], "h/tail": [],
        "v/kappa": [], "h/kappa": [],
        "top1_agree": [],   # vs lambda=1.0 full prediction
    }


def reduce_buf(buf: Dict[str, List[torch.Tensor]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for key, samples in buf.items():
        if key.endswith("/kappa") or key == "top1_agree":
            out[key] = _mean_se(torch.cat(samples, dim=0))
        else:
            out[key] = _stack_mean_se(samples)
    return out


def render_table(per_lambda: Dict[float, Dict[str, dict]], readout: str) -> str:
    rows = [
        f"### Readout: {'predicted-class scalar' if readout == 'v' else 'logit vector'}",
        "",
        "| $\\lambda$ | $\\kappa$ | $C_{\\le 3}$ | $T_{>3}$ | top-1 agree (vs $\\lambda=1$) |",
        "| -: | :-- | :-- | :-- | :-- |",
    ]
    for lam in sorted(per_lambda.keys(), reverse=True):
        m = per_lambda[lam]
        kappa = m[f"{readout}/kappa"]
        cum = m[f"{readout}/cum"]
        tail = m[f"{readout}/tail"]
        agree = m["top1_agree"]
        rows.append(
            f"| {lam:.2f} | {kappa['mean']:.4f} ± {kappa['stderr']:.4f} | "
            f"{cum['mean'][2]:.4f} ± {cum['stderr'][2]:.4f} | "
            f"{tail['mean'][2]:.4f} ± {tail['stderr'][2]:.4f} | "
            f"{agree['mean']:.4f} ± {agree['stderr']:.4f} |"
        )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    lambdas = parse_lambdas(args.lambdas)
    device = torch.device(args.device)
    model, gated_blocks = build_gated_resnet18(pretrained=not args.no_pretrained)
    model.to(device).eval()
    masks_binary = enumerate_masks(L, device=device)            # (256, 8) in {0, 1}
    orders = order_indices(L, device=device)
    full_idx = full_index(L)

    loader, total = build_loader(
        sample_index=str(args.sample_index) if args.sample_index else None,
        synthetic_n=args.num_images if args.synthetic else None,
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
        parquet_dir=args.parquet_dir,
    )
    print(f"[step_3] device={device} images={total} masks/image={NUM_MASKS} "
          f"lambdas={lambdas} batch_size={args.batch_size} mask_chunk={args.mask_chunk}")

    buffers: Dict[float, Dict[str, List[torch.Tensor]]] = {
        lam: _empty_buf() for lam in lambdas
    }

    n_done = 0
    t0 = time.time()
    for batch_idx, (x, _y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        # Cache lambda=1 full prediction first for top1_agree across all lambdas.
        top1_ref = None
        for lam in lambdas:
            scaled_masks = masks_binary * float(lam)
            h_masks = evaluate_all_masks(model, gated_blocks, x, scaled_masks,
                                         args.mask_chunk)
            h_full = h_masks[:, full_idx, :]
            top1 = h_full.argmax(dim=-1)
            if lam == 1.0:
                top1_ref = top1
            agree = (top1 == top1_ref).float() if top1_ref is not None else torch.ones_like(top1, dtype=torch.float32)

            deltas_h = mobius_coefficients(h_masks)
            deltas_v = scalar_mobius_predicted(h_masks, top1)  # readout follows the lambda's own prediction

            E_v = order_energies_scalar(deltas_v, orders)
            E_h = order_energies_vector(deltas_h, orders)
            buf = buffers[lam]
            for tag, E in (("v", E_v), ("h", E_h)):
                d = derived_metrics(E)
                buf[f"{tag}/E"].append(E.cpu())
                buf[f"{tag}/E_tilde"].append(d["E_tilde"].cpu())
                buf[f"{tag}/E_bar"].append(d["E_bar"].cpu())
                buf[f"{tag}/M"].append(d["M"].cpu())
                buf[f"{tag}/cum"].append(d["cum"].cpu())
                buf[f"{tag}/tail"].append(d["tail"].cpu())
                buf[f"{tag}/kappa"].append(d["kappa"].cpu())
            buf["top1_agree"].append(agree.cpu())

        n_done += x.shape[0]
        if (batch_idx + 1) % args.log_every == 0 or n_done == total:
            last = lambdas[-1]
            print(f"[step_3] {n_done}/{total}  "
                  f"k_v(1.0)={buffers[1.0]['v/kappa'][-1].mean().item():.3f}  "
                  f"k_v({last})={buffers[last]['v/kappa'][-1].mean().item():.3f}  "
                  f"agree({last})={buffers[last]['top1_agree'][-1].mean().item():.3f}  "
                  f"({time.time() - t0:.1f}s)")

    per_lambda = {lam: reduce_buf(buf) for lam, buf in buffers.items()}

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
            "lambdas": lambdas,
            "wall_seconds": round(time.time() - t0, 2),
        },
        "per_lambda": {f"{lam:g}": per_lambda[lam] for lam in lambdas},
    }
    (args.output_dir / "summary_all.json").write_text(json.dumps(summary, indent=2))
    table = (
        "# Experiment 3 -- inference-time residual scaling sweep\n\n"
        + render_table(per_lambda, "v") + "\n\n"
        + render_table(per_lambda, "h") + "\n"
    )
    (args.output_dir / "table_3.md").write_text(table)

    print("\n=== Table 3 (preview) ===")
    print(table)
    print(f"\nSaved summary_all.json + table_3.md to {args.output_dir}/")


if __name__ == "__main__":
    main()
