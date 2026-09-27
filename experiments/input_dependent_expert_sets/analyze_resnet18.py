"""Step 2 -- overlap statistics from cached signatures.

Loads ``signatures.pt`` (output of step_1) and computes:

* same-class vs different-class Jaccard / weighted Jaccard at each K;
* overlap gap Delta_class(K) and lift Lift_class(K);
* random-set baseline (uniform K-subsets);
* global-expert baseline -- mean per-image overlap with G_K = TopK_S
  E_x[a_S^h(x)^2] and the corresponding pair overlap from same / different
  -class pairs;
* label-shuffle control Delta_shuffle(K);
* nearest-neighbor classification accuracy in signature space (binary
  Jaccard NN, K = 32);
* easy / hard split: pair overlap conditioned on full-mask confidence
  quantile.

Writes ``overlap.json`` and ``table_4.md``.

Usage:
    python -m experiment_4.step_2_overlap \\
        --signatures experiment_4/results/signatures.pt \\
        --output-dir experiment_4/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from resnet_routes.resnet18.signatures import (
    bootstrap_mean_ci,
    global_top_k,
    jaccard,
    jaccard_vs_global,
    nn_accuracy_binary,
    sample_different_class_pairs,
    sample_same_class_pairs,
    shuffle_labels,
    weighted_jaccard,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--signatures", type=Path,
                   default=Path(__file__).resolve().parent / "generated/resnet18/signatures.pt")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "generated/resnet18")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=200,
                   help="bootstrap resamples for the gap CI")
    p.add_argument("--max-pairs", type=int, default=45000,
                   help="cap on per-condition pair count "
                        "(45 same-class pairs per class * 1000 classes = 45000)")
    p.add_argument("--nn-K", type=int, default=32,
                   help="K used for the NN-accuracy diagnostic")
    return p.parse_args(argv)


def _pair_overlaps(
    binary: torch.Tensor, weights: torch.Tensor, pairs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (jaccard, weighted_jaccard) for a (P, 2) pair index tensor.

    binary / weights shapes: ``(N, R)`` for a fixed K.
    """
    a, b = pairs.unbind(dim=-1)
    j = jaccard(binary[a], binary[b])
    wj = weighted_jaccard(weights[a], weights[b])
    return j, wj


def _summary(samples: torch.Tensor, n_boot: int, gen: torch.Generator) -> Dict[str, float]:
    mean, (lo, hi) = bootstrap_mean_ci(samples, n_boot, alpha=0.05, generator=gen)
    n = samples.numel()
    se = (samples.float().std(unbiased=True).item() / max(n, 1) ** 0.5) if n > 1 else 0.0
    return {"mean": mean, "stderr": se, "ci_lo": lo, "ci_hi": hi, "n": int(n)}


def _random_set_baseline(
    R: int, K: int, n_pairs: int, gen: torch.Generator,
) -> torch.Tensor:
    """Per-pair Jaccard for two independent uniform K-subsets of [R]."""
    out = torch.empty(n_pairs)
    for p in range(n_pairs):
        a = torch.zeros(R, dtype=torch.bool)
        b = torch.zeros(R, dtype=torch.bool)
        ia = torch.randperm(R, generator=gen)[:K]
        ib = torch.randperm(R, generator=gen)[:K]
        a[ia] = True; b[ib] = True
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        out[p] = inter / max(union, 1)
    return out


def _label_shuffle_pairs(
    labels: torch.Tensor, n_pairs: int, gen: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same/different pairs after shuffling labels.  Returns (same, diff) pair indices."""
    sh = shuffle_labels(labels, gen)
    same = sample_same_class_pairs(sh, gen, max_pairs=n_pairs)
    diff = sample_different_class_pairs(sh, n_pairs=same.shape[0], generator=gen)
    return same, diff


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sig = torch.load(args.signatures, map_location="cpu")
    binary_h = sig["binary_h"]                 # (N, K, R) bool
    weighted_h = sig["weighted_topk_h"]        # (N, K, R) float
    binary_v = sig["binary_v"]                 # scalar-ranking
    labels = sig["labels"]
    pseudo = sig["pseudo_labels"]
    conf = sig["confidence"]
    K_list: List[int] = sig["K_list"]
    N, _, R = binary_h.shape

    print(f"[step_2] N={N}  R={R}  K_list={K_list}")
    print(f"[step_2] full-mask top-1 vs ground-truth = "
          f"{(pseudo == labels).float().mean().item():.4f}")

    gen = torch.Generator().manual_seed(args.seed + 7)

    # Pair index sets (computed once; reused across K).
    same_pairs = sample_same_class_pairs(labels, gen, max_pairs=args.max_pairs)
    n_pairs = same_pairs.shape[0]
    diff_pairs = sample_different_class_pairs(labels, n_pairs=n_pairs, generator=gen)
    print(f"[step_2] sampled {n_pairs} same-class and {n_pairs} different-class pairs")

    # Difficulty quantiles
    q33, q66 = torch.quantile(conf, torch.tensor([1.0 / 3, 2.0 / 3]))
    hard_mask = conf <= q33
    easy_mask = conf >= q66
    hard_idx = hard_mask.nonzero(as_tuple=True)[0]
    easy_idx = easy_mask.nonzero(as_tuple=True)[0]
    n_diff_pairs = min(args.max_pairs, 30000)
    eh_gen = torch.Generator().manual_seed(args.seed + 17)
    easy_pairs = _diff_pairs_from_set(easy_idx, n_diff_pairs, eh_gen)
    hard_pairs = _diff_pairs_from_set(hard_idx, n_diff_pairs, eh_gen)

    # Per-K analysis
    per_K: Dict[int, dict] = {}
    for ki, K in enumerate(K_list):
        bin_K = binary_h[:, ki, :]
        w_K = weighted_h[:, ki, :]
        bin_v_K = binary_v[:, ki, :]

        same_j, same_wj = _pair_overlaps(bin_K, w_K, same_pairs)
        diff_j, diff_wj = _pair_overlaps(bin_K, w_K, diff_pairs)

        same_j_v = jaccard(bin_v_K[same_pairs[:, 0]], bin_v_K[same_pairs[:, 1]])
        diff_j_v = jaccard(bin_v_K[diff_pairs[:, 0]], bin_v_K[diff_pairs[:, 1]])

        # Random-set baseline (Monte Carlo)
        rand_j = _random_set_baseline(R, K, n_pairs=min(2000, n_pairs), gen=gen)

        # Label shuffle
        sh_same, sh_diff = _label_shuffle_pairs(labels, n_pairs, gen)
        sh_same_j, _ = _pair_overlaps(bin_K, w_K, sh_same)
        sh_diff_j, _ = _pair_overlaps(bin_K, w_K, sh_diff)

        # Global expert overlap per image
        G_K = global_top_k(sig["weights_full_h"], K)
        per_image_global = jaccard_vs_global(bin_K, G_K)

        # Easy / hard pair overlaps
        easy_j, _ = _pair_overlaps(bin_K, w_K, easy_pairs)
        hard_j, _ = _pair_overlaps(bin_K, w_K, hard_pairs)

        per_K[K] = {
            "same_jaccard": _summary(same_j, args.n_boot, gen),
            "diff_jaccard": _summary(diff_j, args.n_boot, gen),
            "same_wjaccard": _summary(same_wj, args.n_boot, gen),
            "diff_wjaccard": _summary(diff_wj, args.n_boot, gen),
            "same_jaccard_scalar": _summary(same_j_v, args.n_boot, gen),
            "diff_jaccard_scalar": _summary(diff_j_v, args.n_boot, gen),
            "random_jaccard": _summary(rand_j, args.n_boot, gen),
            "shuffle_same_jaccard": _summary(sh_same_j, args.n_boot, gen),
            "shuffle_diff_jaccard": _summary(sh_diff_j, args.n_boot, gen),
            "global_jaccard_per_image": _summary(per_image_global, args.n_boot, gen),
            "easy_jaccard": _summary(easy_j, args.n_boot, gen),
            "hard_jaccard": _summary(hard_j, args.n_boot, gen),
            "delta_class": same_j.float().mean().item() - diff_j.float().mean().item(),
            "delta_shuffle": sh_same_j.float().mean().item() - sh_diff_j.float().mean().item(),
            "lift_class": (
                same_j.float().mean().item() / (diff_j.float().mean().item() + 1e-12)
            ),
            "delta_class_weighted": (
                same_wj.float().mean().item() - diff_wj.float().mean().item()
            ),
        }

    # Nearest-neighbor classification at args.nn_K
    if args.nn_K not in K_list:
        raise ValueError(f"nn_K={args.nn_K} must be one of {K_list}")
    ki = K_list.index(args.nn_K)
    nn_acc, chance = nn_accuracy_binary(binary_h[:, ki, :], labels)
    nn_acc_v, _ = nn_accuracy_binary(binary_v[:, ki, :], labels)

    summary = {
        "config": {
            "signatures": str(args.signatures),
            "n_images": N,
            "n_pairs": n_pairs,
            "K_list": K_list,
            "nn_K": args.nn_K,
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "full_mask_top1_acc": (pseudo == labels).float().mean().item(),
        "per_K": {str(K): v for K, v in per_K.items()},
        "nn_accuracy_K": {
            "K": args.nn_K,
            "binary_vector": nn_acc,
            "binary_scalar": nn_acc_v,
            "chance": chance,
        },
    }
    (args.output_dir / "overlap.json").write_text(json.dumps(summary, indent=2))
    table = render_table(summary)
    (args.output_dir / "table_4.md").write_text(table)

    print("\n=== Table 4 (preview) ===\n" + table)
    print(f"Saved overlap.json + table_4.md to {args.output_dir}/")


def _diff_pairs_from_set(idx: torch.Tensor, n_pairs: int, gen: torch.Generator) -> torch.Tensor:
    """Sample n_pairs random pairs (i, j) from a fixed index set (i != j)."""
    M = idx.numel()
    out = torch.empty(n_pairs, 2, dtype=torch.long)
    filled = 0
    while filled < n_pairs:
        block = max(n_pairs - filled, 1024)
        a = torch.randint(0, M, (block,), generator=gen)
        b = torch.randint(0, M, (block,), generator=gen)
        keep = (a != b)
        sel = keep.nonzero(as_tuple=True)[0]
        take = min(sel.numel(), n_pairs - filled)
        out[filled : filled + take, 0] = idx[a[sel[:take]]]
        out[filled : filled + take, 1] = idx[b[sel[:take]]]
        filled += take
    return out


def render_table(summary: Dict) -> str:
    rows = [
        f"# Experiment 4 -- Input-dependent residual expert sets "
        f"(N = {summary['config']['n_images']})",
        "",
        f"Full-mask top-1 accuracy on val subset = {summary['full_mask_top1_acc']:.4f}",
        "",
        "## Vector ranking: same-class vs different-class Jaccard",
        "",
        "| K | same J | diff J | $\\Delta_{class}$ | lift | random J | global J (mean/img) |",
        "| -: | :-- | :-- | :-- | :-- | :-- | :-- |",
    ]
    for K_str, m in summary["per_K"].items():
        rows.append(
            f"| {K_str} | {m['same_jaccard']['mean']:.4f} ± {m['same_jaccard']['stderr']:.4f} | "
            f"{m['diff_jaccard']['mean']:.4f} ± {m['diff_jaccard']['stderr']:.4f} | "
            f"{m['delta_class']:.4f} | {m['lift_class']:.3f}× | "
            f"{m['random_jaccard']['mean']:.4f} | "
            f"{m['global_jaccard_per_image']['mean']:.4f} ± {m['global_jaccard_per_image']['stderr']:.4f} |"
        )

    rows += [
        "",
        "## Weighted Jaccard (top-$K$ mass)",
        "",
        "| K | same wJ | diff wJ | $\\Delta_{class}^{w}$ |",
        "| -: | :-- | :-- | :-- |",
    ]
    for K_str, m in summary["per_K"].items():
        rows.append(
            f"| {K_str} | {m['same_wjaccard']['mean']:.4f} ± {m['same_wjaccard']['stderr']:.4f} | "
            f"{m['diff_wjaccard']['mean']:.4f} ± {m['diff_wjaccard']['stderr']:.4f} | "
            f"{m['delta_class_weighted']:.4f} |"
        )

    rows += [
        "",
        "## Label-shuffle control",
        "",
        "| K | shuffle $\\Delta$ | true $\\Delta$ | gap (true / shuffle) |",
        "| -: | :-- | :-- | :-- |",
    ]
    for K_str, m in summary["per_K"].items():
        gap = (m['delta_class'] + 1e-12) / (abs(m['delta_shuffle']) + 1e-9)
        rows.append(
            f"| {K_str} | {m['delta_shuffle']:.4f} | {m['delta_class']:.4f} | {gap:.1f}× |"
        )

    rows += [
        "",
        "## Easy vs hard subsets (full-mask confidence quantiles)",
        "",
        "| K | easy J (top-third conf) | hard J (bottom-third conf) | hard - easy |",
        "| -: | :-- | :-- | :-- |",
    ]
    for K_str, m in summary["per_K"].items():
        rows.append(
            f"| {K_str} | {m['easy_jaccard']['mean']:.4f} | {m['hard_jaccard']['mean']:.4f} | "
            f"{m['hard_jaccard']['mean'] - m['easy_jaccard']['mean']:+.4f} |"
        )

    nn = summary["nn_accuracy_K"]
    rows += [
        "",
        "## Scalar ranking sanity check (predicted-class scalar)",
        "",
        "| K | same J (scalar) | diff J (scalar) | $\\Delta_{class}$ |",
        "| -: | :-- | :-- | :-- |",
    ]
    for K_str, m in summary["per_K"].items():
        delta_v = m['same_jaccard_scalar']['mean'] - m['diff_jaccard_scalar']['mean']
        rows.append(
            f"| {K_str} | {m['same_jaccard_scalar']['mean']:.4f} | "
            f"{m['diff_jaccard_scalar']['mean']:.4f} | {delta_v:.4f} |"
        )

    rows += [
        "",
        f"## Nearest-neighbor classification (binary Jaccard, K = {nn['K']})",
        "",
        f"- vector-ranking signatures: **{nn['binary_vector']:.4f}**",
        f"- scalar-ranking signatures: {nn['binary_scalar']:.4f}",
        f"- chance level (sum_c p_c^2): {nn['chance']:.4f}",
    ]
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
