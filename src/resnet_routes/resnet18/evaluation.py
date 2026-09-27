"""Shared evaluation utilities for steps 3 and 4."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .model import GatedBasicBlock, set_per_sample_masks


@torch.no_grad()
def evaluate_all_masks(
    model: torch.nn.Module,
    gated_blocks: Sequence[GatedBasicBlock],
    x: torch.Tensor,
    masks: torch.Tensor,
    mask_chunk: int,
) -> torch.Tensor:
    """Return logits ``h_x(1_S)`` for every image in ``x`` and every mask in ``masks``.

    Replicates the input across the mask dimension so a single forward pass
    yields all ``(B, M', 1000)`` outputs together.  Reduce ``mask_chunk`` if
    VRAM is tight; ``mask_chunk * batch_size`` images are forwarded at once.
    """
    B = x.shape[0]
    M_total, L_local = masks.shape
    assert L_local == len(gated_blocks)
    out = torch.empty(B, M_total, 1000, device=x.device, dtype=torch.float32)
    for start in range(0, M_total, mask_chunk):
        end = min(start + mask_chunk, M_total)
        m = masks[start:end]
        M_prime = m.shape[0]
        x_rep = x.unsqueeze(1).expand(B, M_prime, *x.shape[1:]).reshape(
            B * M_prime, *x.shape[1:]
        )
        m_rep = m.unsqueeze(0).expand(B, M_prime, L_local).reshape(B * M_prime, L_local)
        set_per_sample_masks(gated_blocks, m_rep)
        logits = model(x_rep)
        out[:, start:end, :] = logits.view(B, M_prime, 1000).to(out.dtype)
    return out


def aggregate(metric_logs: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Mean + standard error across all logged per-image values."""
    agg: Dict[str, Dict[str, float]] = {}
    for name, values in metric_logs.items():
        t = torch.tensor(values, dtype=torch.float64)
        n = max(t.numel(), 1)
        mean = t.mean().item()
        se = (t.std(unbiased=True).item() / (n ** 0.5)) if n > 1 else 0.0
        agg[name] = {"mean": mean, "stderr": se, "n": n}
    return agg


RECON_KEYS = [
    ("Raw subset sum $A_x^{raw}$", "raw"),
    ("Mean subset output $A_x^{mean}$", "mean"),
    ("Centered subset sum $A_x^{ctr}$", "ctr"),
    ("Möbius reconstruction", "mob"),
]


def render_table(agg: Dict[str, Dict[str, float]]) -> str:
    """Render Table 1 (markdown) from the aggregated per-image metrics."""
    header = ("| Reconstruction | Norm ratio | Rel. error | Scaled error | "
              "Cosine | Top-1 agree |\n"
              "| :--- | :--- | :--- | :--- | :--- | :--- |")
    lines = [header]
    for label, key in RECON_KEYS:
        def cell(metric: str) -> str:
            full = f"{key}/{metric}"
            if full not in agg:
                return "—"
            d = agg[full]
            return f"{d['mean']:.3g} ± {d['stderr']:.2g}"
        lines.append(
            f"| {label} | {cell('norm_ratio')} | {cell('rel_err')} | "
            f"{cell('rel_err_scaled')} | {cell('cosine')} | {cell('top1_agree')} |"
        )
    return "\n".join(lines)
