"""Difficulty scores from full-mask logits.

Five scores, three label-aware and two label-free.  All scores are
formulated so that **larger value = harder sample** (so we negate
confidence and margins).
"""

from __future__ import annotations

from typing import Dict

import torch

EPS = 1e-12


def full_softmax(h_full: torch.Tensor) -> torch.Tensor:
    return h_full.softmax(dim=-1)


def predicted_class(h_full: torch.Tensor) -> torch.Tensor:
    return h_full.argmax(dim=-1)


def cross_entropy_loss(
    h_full: torch.Tensor, labels: torch.Tensor,
) -> torch.Tensor:
    """Per-image NLL ``-log softmax(h_full)_y``.  Shape (B,)."""
    log_p = h_full.log_softmax(dim=-1)
    return -log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def correctness(h_full: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """``Wrong(x) = 1[hat_y(x) != y]`` -- (B,) float in {0, 1}."""
    return (h_full.argmax(dim=-1) != labels).float()


def true_class_margin(
    h_full: torch.Tensor, labels: torch.Tensor,
) -> torch.Tensor:
    """``h_y - max_{c != y} h_c``.  Larger = easier."""
    h_y = h_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    masked = h_full.scatter(-1, labels.unsqueeze(-1), float("-inf"))
    other = masked.max(dim=-1).values
    return h_y - other


def confidence(h_full: torch.Tensor) -> torch.Tensor:
    """``p_max(x) = max_c softmax(h_full)_c``.  Larger = easier."""
    return h_full.softmax(dim=-1).max(dim=-1).values


def predicted_margin(h_full: torch.Tensor) -> torch.Tensor:
    """``h_{hat_y} - max_{c != hat_y} h_c``.  Larger = easier."""
    yhat = h_full.argmax(dim=-1)
    h_yhat = h_full.gather(-1, yhat.unsqueeze(-1)).squeeze(-1)
    masked = h_full.scatter(-1, yhat.unsqueeze(-1), float("-inf"))
    other = masked.max(dim=-1).values
    return h_yhat - other


def all_difficulty_scores(
    h_full: torch.Tensor, labels: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    """Compute every difficulty score; label-aware ones are skipped if
    ``labels`` is None.  Returned tensors are all (B,) on the same device
    as ``h_full``.
    """
    out: Dict[str, torch.Tensor] = {
        "confidence": confidence(h_full),
        "pred_margin": predicted_margin(h_full),
    }
    if labels is not None:
        out["loss"] = cross_entropy_loss(h_full, labels)
        out["wrong"] = correctness(h_full, labels)
        out["true_margin"] = true_class_margin(h_full, labels)
    return out


# Sign convention for "larger value = harder": flip confidence and
# margins by negation when correlating against complexity measures.
HARDER_IS_LARGER = {
    "loss": +1,            # already harder = larger
    "wrong": +1,           # already harder = larger
    "confidence": -1,      # easier = larger -> flip
    "pred_margin": -1,
    "true_margin": -1,
}
