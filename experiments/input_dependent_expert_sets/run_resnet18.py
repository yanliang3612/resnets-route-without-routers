"""Step 1 -- compute and dump top-$K$ expert signatures for the val subset.

Per image:
  * 256-mask sweep -> deltas_h (B, 256, C), deltas_v (B, 256).
  * Vector magnitudes a_S^h(x) and scalar magnitudes a_S^v(x) over the
    255 non-empty residual subsets.
  * Binary and weighted top-$K$ signatures for K in K_LIST (vector ranking).
  * Per-image metadata: ground-truth label, predicted label,
    full-mask softmax confidence, full-mask margin (predicted vs runner-up).

Outputs: ``signatures.pt`` -- dict of per-image tensors, plus
``config.json`` with run configuration.

Usage:
    python -m experiment_4.step_1_signatures \\
        --sample-index experiment_1/sample_index_val.json \\
        --batch-size 8 --mask-chunk 256 --num-workers 8 --seed 0 \\
        --output-dir experiment_4/results

    python -m experiment_4.step_1_signatures --synthetic --num-images 8   # smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from resnet_routes.resnet18.evaluation import evaluate_all_masks
from resnet_routes.resnet18.data import build_loader
from resnet_routes.resnet18.model import build_gated_resnet18
from resnet_routes.resnet18.mobius import enumerate_masks, full_index, mobius_coefficients
from resnet_routes.resnet18.spectrum import scalar_mobius_predicted
from resnet_routes.resnet18.signatures import make_topk_signatures


L = 8
NUM_MASKS = 1 << L
NUM_RES = NUM_MASKS - 1                            # 255
# Plan §"Choice of K": 3, 8, 13, 26, 32, 64 (~1%, ~3%, ~5%, ~10%, ~13%, ~25%).
K_LIST: Tuple[int, ...] = (3, 8, 13, 26, 32, 64)


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

    loader, total = build_loader(
        sample_index=str(args.sample_index) if args.sample_index else None,
        synthetic_n=args.num_images if args.synthetic else None,
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
        parquet_dir=args.parquet_dir,
    )
    print(f"[step_1] device={device} images={total} K_list={K_LIST} "
          f"batch_size={args.batch_size} mask_chunk={args.mask_chunk}")

    bin_h: List[torch.Tensor] = []
    w_topk_h: List[torch.Tensor] = []
    w_full_h: List[torch.Tensor] = []
    bin_v: List[torch.Tensor] = []         # scalar-ranking (decision-level robustness)
    label_buf: List[torch.Tensor] = []
    pseudo_buf: List[torch.Tensor] = []
    conf_buf: List[torch.Tensor] = []
    margin_buf: List[torch.Tensor] = []

    n_done = 0
    t0 = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        h_masks = evaluate_all_masks(model, gated_blocks, x, masks, args.mask_chunk)
        h_full = h_masks[:, full_idx, :]
        top1_full = h_full.argmax(dim=-1)
        softmax_full = h_full.softmax(dim=-1)
        conf, _ = softmax_full.max(dim=-1)
        full_y = torch.gather(h_full, -1, top1_full.unsqueeze(-1)).squeeze(-1)
        masked = h_full.scatter(-1, top1_full.unsqueeze(-1), float("-inf"))
        runner = masked.max(dim=-1).values
        margin = full_y - runner

        deltas_h = mobius_coefficients(h_masks)              # (B, 256, C)
        deltas_v = scalar_mobius_predicted(h_masks, top1_full)  # (B, 256)
        residual_h = deltas_h[:, 1:, :]
        residual_v = deltas_v[:, 1:]

        mags_h = residual_h.pow(2).sum(dim=-1).sqrt()         # (B, 255)
        mags_v = residual_v.abs()                             # (B, 255)

        bh, wh, fh = make_topk_signatures(mags_h, K_LIST)
        bv, _, _ = make_topk_signatures(mags_v, K_LIST)
        bin_h.append(bh); w_topk_h.append(wh); w_full_h.append(fh.cpu())
        bin_v.append(bv)

        label_buf.append(y.long())
        pseudo_buf.append(top1_full.cpu())
        conf_buf.append(conf.cpu())
        margin_buf.append(margin.cpu())

        n_done += x.shape[0]
        if (batch_idx + 1) % args.log_every == 0 or n_done == total:
            print(f"[step_1] {n_done}/{total}  ({time.time() - t0:.1f}s)")

    out = {
        "binary_h": torch.cat(bin_h, dim=0),                 # (N, K, R)
        "weighted_topk_h": torch.cat(w_topk_h, dim=0),       # (N, K, R)
        "weights_full_h": torch.cat(w_full_h, dim=0),        # (N, R)
        "binary_v": torch.cat(bin_v, dim=0),                 # (N, K, R)
        "labels": torch.cat(label_buf, dim=0),               # (N,)
        "pseudo_labels": torch.cat(pseudo_buf, dim=0),       # (N,)
        "confidence": torch.cat(conf_buf, dim=0),            # (N,)
        "margin": torch.cat(margin_buf, dim=0),              # (N,)
        "K_list": list(K_LIST),
    }
    sig_path = args.output_dir / "signatures.pt"
    torch.save(out, sig_path)
    cfg = {
        "sample_index": str(args.sample_index) if args.sample_index else None,
        "synthetic": args.synthetic,
        "num_images": total,
        "L": L,
        "num_residual_subsets": NUM_RES,
        "K_list": list(K_LIST),
        "batch_size": args.batch_size,
        "mask_chunk": args.mask_chunk,
        "seed": args.seed,
        "device": str(device),
        "wall_seconds": round(time.time() - t0, 2),
    }
    (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"\nSaved signatures to {sig_path} "
          f"(binary_h={tuple(out['binary_h'].shape)}, "
          f"acc/labeled={(out['pseudo_labels'] == out['labels']).float().mean().item():.4f})")
    print("Run experiment_4.step_2_overlap to compute overlap statistics.")


if __name__ == "__main__":
    main()
