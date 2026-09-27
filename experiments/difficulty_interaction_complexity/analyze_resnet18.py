"""Step 2 -- correlation, easy/hard gap, correctness gap, class-controlled regression.

Loads ``per_image.pt`` (output of step_1) and computes for every
(difficulty, complexity) pair:

* Spearman rank correlation (with sign convention "harder = larger");
* easy / hard quartile gap (top vs bottom 25%);
* correct / wrong gap (label-aware difficulty axis only);
* class-controlled OLS coefficient ``beta_1`` after subtracting per-class
  means from both the difficulty score and the complexity measure.

Writes ``analysis.json`` and ``table_5.md``.

Usage:
    python -m experiment_5.step_2_analyze \\
        --per-image experiment_5/results/per_image.pt \\
        --output-dir experiment_5/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from resnet_routes.resnet18.difficulty import HARDER_IS_LARGER


COMPLEXITY_KEYS_VEC = (
    "h/N_eff", "h/N_ent", "h/kbar", "h/kappa",
    "h/C_le_3", "h/T_gt_3",
    "K_err_010", "K_err_005",
    "K_mass_090", "K_mass_095", "K_mass_099",
)
COMPLEXITY_KEYS_SCALAR = (
    "v/N_eff", "v/N_ent", "v/kbar", "v/kappa",
    "v/C_le_3", "v/T_gt_3",
)
DIFFICULTY_KEYS = ("loss", "wrong", "true_margin", "confidence", "pred_margin")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-image", type=Path,
                   default=Path(__file__).resolve().parent / "generated/resnet18/per_image.pt")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "generated/resnet18")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Tie-aware Spearman rank correlation."""
    value = float(spearmanr(x.detach().cpu().numpy(), y.detach().cpu().numpy()).statistic)
    return 0.0 if np.isnan(value) else value


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.to(torch.float64); y = y.to(torch.float64)
    x = x - x.mean(); y = y - y.mean()
    den = (x.std(unbiased=False) * y.std(unbiased=False)).item()
    if den < 1e-30:
        return 0.0
    return ((x * y).mean() / (x.std(unbiased=False) * y.std(unbiased=False))).item()


def quartile_gap(
    diff: torch.Tensor, comp: torch.Tensor, *, low_q: float = 0.25, high_q: float = 0.75,
) -> Tuple[float, float, float]:
    """Hard - easy gap and stderr.

    ``diff`` is interpreted as "larger = harder" (caller flips sign if not).
    Returns (gap, easy_mean, hard_mean).
    """
    qs = torch.quantile(diff, torch.tensor([low_q, high_q], dtype=diff.dtype))
    q_lo, q_hi = qs[0].item(), qs[1].item()
    easy = comp[diff <= q_lo]
    hard = comp[diff >= q_hi]
    return hard.mean().item() - easy.mean().item(), easy.mean().item(), hard.mean().item()


def correctness_gap(
    wrong: torch.Tensor, comp: torch.Tensor,
) -> Tuple[float, float, float]:
    """``wrong`` is 0/1.  Returns (gap, mean_correct, mean_wrong)."""
    correct_mask = wrong == 0
    wrong_mask = wrong == 1
    return (comp[wrong_mask].mean() - comp[correct_mask].mean()).item(), \
           comp[correct_mask].mean().item(), \
           comp[wrong_mask].mean().item()


def class_controlled_beta(
    diff: torch.Tensor, comp: torch.Tensor, labels: torch.Tensor,
) -> float:
    """OLS slope of ``comp`` on ``diff`` after removing per-class means.

    Algebraically equivalent to the class fixed-effect regression
    ``c = beta0 + beta1 d + gamma_y + eps``.
    """
    d = diff.to(torch.float64); c = comp.to(torch.float64)
    # subtract per-class means
    classes = torch.unique(labels)
    d_demeaned = d.clone()
    c_demeaned = c.clone()
    for cls in classes:
        m = labels == cls
        if m.sum() <= 1:
            continue
        d_demeaned[m] -= d[m].mean()
        c_demeaned[m] -= c[m].mean()
    var_d = (d_demeaned * d_demeaned).sum().item()
    if var_d < 1e-30:
        return 0.0
    return ((d_demeaned * c_demeaned).sum() / var_d).item()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    p = torch.load(args.per_image, map_location="cpu")

    N = p["labels"].numel()
    full_acc = (p["pseudo_labels"] == p["labels"]).float().mean().item()
    print(f"[step_2] N={N}  full-mask top-1 = {full_acc:.4f}")

    diff_keys = [k for k in DIFFICULTY_KEYS if k in p]
    has_labels = "loss" in p

    # Precompute "hard-direction" copies of every difficulty score (so
    # larger = harder for all of them).
    diff_hard: Dict[str, torch.Tensor] = {}
    for k in diff_keys:
        sign = HARDER_IS_LARGER[k]
        diff_hard[k] = (p[k].to(torch.float32) * sign).contiguous()

    # ---- Spearman + quartile gap for every (difficulty, complexity) pair
    rows: List[Dict] = []
    comp_keys = list(COMPLEXITY_KEYS_VEC) + list(COMPLEXITY_KEYS_SCALAR)
    for dk in diff_keys:
        d = diff_hard[dk]
        for ck in comp_keys:
            c = p[ck].to(torch.float32)
            rho = spearman(d, c)
            r = pearson(d, c)
            gap, easy, hard = quartile_gap(d, c)
            row = {
                "difficulty": dk, "complexity": ck,
                "spearman": rho, "pearson": r,
                "gap_hard_minus_easy": gap, "easy_mean": easy, "hard_mean": hard,
                "easy_q": "<=q25", "hard_q": ">=q75",
            }
            if has_labels:
                row["beta_class_controlled"] = class_controlled_beta(
                    d, c, p["labels"]
                )
            rows.append(row)

    correct_rows: List[Dict] = []
    if has_labels:
        wrong = p["wrong"].to(torch.float32)
        for ck in comp_keys:
            c = p[ck].to(torch.float32)
            gap, m_correct, m_wrong = correctness_gap(wrong, c)
            correct_rows.append({
                "complexity": ck,
                "mean_correct": m_correct, "mean_wrong": m_wrong,
                "gap_wrong_minus_correct": gap,
            })

    # ---- aggregate per-complexity summary table ----
    summary = {
        "config": {
            "per_image": str(args.per_image),
            "n_images": N,
            "full_mask_top1_acc": full_acc,
            "complexity_keys_vector": list(COMPLEXITY_KEYS_VEC),
            "complexity_keys_scalar": list(COMPLEXITY_KEYS_SCALAR),
            "difficulty_keys": diff_keys,
            "has_labels": has_labels,
        },
        "rows": rows,
        "correctness_rows": correct_rows,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(summary, indent=2))
    table = render_table(summary)
    (args.output_dir / "table_5.md").write_text(table)
    print("\n=== Table 5 (preview) ===\n" + table)
    print(f"Saved analysis.json + table_5.md to {args.output_dir}/")


def render_table(summary: Dict) -> str:
    rows = summary["rows"]
    has_labels = summary["config"]["has_labels"]

    def filt(diff_key: str, complexity_keys):
        return [r for r in rows
                if r["difficulty"] == diff_key and r["complexity"] in complexity_keys]

    out = [f"# Experiment 5 -- difficulty vs interaction complexity "
           f"(N = {summary['config']['n_images']})",
           "",
           f"Full-mask top-1 = {summary['config']['full_mask_top1_acc']:.4f}",
           "",
           "## Spearman rank correlation (vector ranking complexity)",
           "",
           "| difficulty (harder=larger) | "
           + " | ".join(COMPLEXITY_KEYS_VEC)
           + " |",
           "| :-- | "
           + " | ".join([":--"] * len(COMPLEXITY_KEYS_VEC))
           + " |"]
    for dk in summary["config"]["difficulty_keys"]:
        cells = []
        for ck in COMPLEXITY_KEYS_VEC:
            for r in rows:
                if r["difficulty"] == dk and r["complexity"] == ck:
                    cells.append(f"{r['spearman']:+.3f}")
                    break
        out.append(f"| {dk} | " + " | ".join(cells) + " |")

    out += ["", "## Spearman rank correlation (scalar ranking complexity)",
            "",
            "| difficulty (harder=larger) | "
            + " | ".join(COMPLEXITY_KEYS_SCALAR)
            + " |",
            "| :-- | "
            + " | ".join([":--"] * len(COMPLEXITY_KEYS_SCALAR))
            + " |"]
    for dk in summary["config"]["difficulty_keys"]:
        cells = []
        for ck in COMPLEXITY_KEYS_SCALAR:
            for r in rows:
                if r["difficulty"] == dk and r["complexity"] == ck:
                    cells.append(f"{r['spearman']:+.3f}")
                    break
        out.append(f"| {dk} | " + " | ".join(cells) + " |")

    out += ["", "## Hard-easy quartile gap (loss-based difficulty)",
            "",
            "| complexity | easy mean | hard mean | hard - easy |",
            "| :-- | :-- | :-- | :-- |"]
    for ck in list(COMPLEXITY_KEYS_VEC) + list(COMPLEXITY_KEYS_SCALAR):
        for r in rows:
            if (r["difficulty"] == ("loss" if has_labels else "pred_margin")
                    and r["complexity"] == ck):
                out.append(f"| {ck} | {r['easy_mean']:.4g} | {r['hard_mean']:.4g} | "
                           f"{r['gap_hard_minus_easy']:+.4g} |")
                break

    if has_labels:
        out += ["", "## Correctness gap (wrong - correct)",
                "",
                "| complexity | mean correct | mean wrong | wrong - correct |",
                "| :-- | :-- | :-- | :-- |"]
        for r in summary["correctness_rows"]:
            out.append(f"| {r['complexity']} | {r['mean_correct']:.4g} | "
                       f"{r['mean_wrong']:.4g} | {r['gap_wrong_minus_correct']:+.4g} |")

        out += ["", "## Class-controlled OLS beta_1 (loss as difficulty)",
                "",
                "| complexity | beta_1 within-class |",
                "| :-- | :-- |"]
        for r in rows:
            if r["difficulty"] == "loss":
                out.append(f"| {r['complexity']} | {r.get('beta_class_controlled', 0):+.4g} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
