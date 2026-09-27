"""Top-$K$ expert signatures and overlap metrics for Experiment 4.

Inputs use the same magnitude rankings as the Top-K analysis (vector
$a_S^h(x) = \\|\\Delta_S h_x\\|_2$ by default; scalar
$a_S^v(x) = |\\Delta_S v_x|$ optional).  This module provides:

* ``binary_signatures`` / ``weighted_signatures`` for a list of K values;
* ``jaccard``, ``weighted_jaccard`` for pair similarity;
* ``mean_pair_jaccard_by_label`` to compute same-class / different-class
  overlaps with bootstrap confidence intervals;
* a global expert set $\\mathcal G_K$ and per-image overlap with it;
* nearest-neighbor classification accuracy in signature space.

For 10000 images we have 5 × 10⁷ unordered pairs, so all-pairs reductions
are out of reach.  We instead reduce within balanced sample groups
(same-class pairs are tractable: ~45000 per class-balanced 10/class
subset) and sample a matched number of different-class pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #

@dataclass
class Signatures:
    """Container for top-$K$ expert signatures of a dataset.

    Each tensor has shape ``(N, len(K_list), R)`` for ``binary`` and
    ``weighted_topk`` (the rest of the weight vector is zeroed outside
    the top-$K$, matching plan §"Expert signatures"), or ``(N, R)`` for
    the un-truncated weight vector ``weights_full``.
    """
    binary: torch.Tensor             # (N, len(K), R) bool
    weighted_topk: torch.Tensor      # (N, len(K), R) float (mass-fraction)
    weights_full: torch.Tensor       # (N, R) float (mass-fraction over all R)
    labels: torch.Tensor             # (N,) int (ground truth)
    pseudo_labels: torch.Tensor      # (N,) int (full-mask top-1)
    confidence: torch.Tensor         # (N,) float (full-mask softmax max)
    margin: torch.Tensor             # (N,) float (predicted vs runner-up)
    K_list: Tuple[int, ...]


def make_topk_signatures(
    mags: torch.Tensor,             # (B, R) magnitudes a_S^h(x)
    K_list: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute binary, top-$K$ weighted, and full mass-fraction signatures.

    Returns:
        binary:        (B, len(K), R) bool
        weighted_topk: (B, len(K), R) float (zero outside the top-$K$)
        weights_full:  (B, R) float
    """
    B, R = mags.shape
    sq = mags.pow(2)
    total = sq.sum(dim=-1, keepdim=True)
    weights_full = sq / (total + EPS)

    bin_out = torch.zeros(B, len(K_list), R, dtype=torch.bool)
    w_out = torch.zeros(B, len(K_list), R, dtype=weights_full.dtype)

    sort_idx = mags.argsort(dim=-1, descending=True)            # (B, R)
    for ki, K in enumerate(K_list):
        topk = sort_idx[:, :K]
        # binary: scatter ones at the top-K positions
        bin_out[:, ki, :].scatter_(1, topk.cpu(), True)
        # weighted top-K: keep weights at top-K, zero elsewhere
        mask = torch.zeros_like(weights_full, dtype=torch.bool)
        mask.scatter_(1, topk, True)
        w_out[:, ki, :] = (weights_full * mask).cpu()
    return bin_out, w_out, weights_full.cpu()


# --------------------------------------------------------------------------- #
# Pair-level overlaps
# --------------------------------------------------------------------------- #

def jaccard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Boolean Jaccard for matched batches.

    Args:
        a, b: ``(P, R)`` boolean signatures.
    Returns:
        ``(P,)`` Jaccard.
    """
    inter = (a & b).sum(dim=-1).float()
    union = (a | b).sum(dim=-1).float()
    return inter / (union + EPS)


def weighted_jaccard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Weighted Jaccard (Tanimoto / Ruzicka) for matched batches.

    Args:
        a, b: ``(P, R)`` non-negative weights.
    Returns:
        ``(P,)`` weighted Jaccard.
    """
    num = torch.minimum(a, b).sum(dim=-1)
    den = torch.maximum(a, b).sum(dim=-1)
    return num / (den + EPS)


# --------------------------------------------------------------------------- #
# Pair sampling helpers
# --------------------------------------------------------------------------- #

def sample_same_class_pairs(
    labels: torch.Tensor, generator: torch.Generator, max_pairs: int | None = None,
) -> torch.Tensor:
    """All within-class pairs ``(i < j)`` collected per label.

    Returns a ``(P, 2)`` tensor of indices.  If ``max_pairs`` is given,
    randomly subsamples to that many pairs.  For 10 images / class,
    each class contributes 45 pairs -> 45000 total.
    """
    pairs = []
    for c in torch.unique(labels):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        ii, jj = torch.combinations(idx, r=2).unbind(dim=-1)
        pairs.append(torch.stack([ii, jj], dim=-1))
    out = torch.cat(pairs, dim=0)
    if max_pairs is not None and out.shape[0] > max_pairs:
        perm = torch.randperm(out.shape[0], generator=generator)[:max_pairs]
        out = out[perm]
    return out


def sample_different_class_pairs(
    labels: torch.Tensor, n_pairs: int, generator: torch.Generator,
) -> torch.Tensor:
    """Independently sample ``n_pairs`` (i, j) with ``label_i != label_j``."""
    N = labels.shape[0]
    out = torch.empty(n_pairs, 2, dtype=torch.long)
    filled = 0
    while filled < n_pairs:
        block = max(n_pairs - filled, 1024)
        a = torch.randint(0, N, (block,), generator=generator)
        b = torch.randint(0, N, (block,), generator=generator)
        keep = (labels[a] != labels[b]) & (a != b)
        sel = keep.nonzero(as_tuple=True)[0]
        take = min(sel.numel(), n_pairs - filled)
        out[filled : filled + take, 0] = a[sel[:take]]
        out[filled : filled + take, 1] = b[sel[:take]]
        filled += take
    return out


def shuffle_labels(labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(labels.numel(), generator=generator)
    return labels[perm].clone()


# --------------------------------------------------------------------------- #
# Global expert set
# --------------------------------------------------------------------------- #

def global_top_k(weights_full: torch.Tensor, K: int) -> torch.Tensor:
    """``G_K = TopK_S E_x[a_S^h(x)^2]``.  Returns a (R,) bool mask of size K."""
    mean = weights_full.mean(dim=0)
    top = mean.argsort(descending=True)[:K]
    out = torch.zeros(weights_full.shape[1], dtype=torch.bool)
    out[top] = True
    return out


def jaccard_vs_global(binary_K: torch.Tensor, global_K: torch.Tensor) -> torch.Tensor:
    """Per-image Jaccard against the global top-$K$ set.  ``binary_K``: (N, R)."""
    inter = (binary_K & global_K.unsqueeze(0)).sum(dim=-1).float()
    union = (binary_K | global_K.unsqueeze(0)).sum(dim=-1).float()
    return inter / (union + EPS)


# --------------------------------------------------------------------------- #
# Bootstrap helpers
# --------------------------------------------------------------------------- #

def bootstrap_mean_ci(
    values: torch.Tensor, n_boot: int, alpha: float, generator: torch.Generator,
) -> Tuple[float, Tuple[float, float]]:
    """Returns mean and (lo, hi) percentile CI."""
    v = values.detach().to(torch.float64).reshape(-1)
    n = v.numel()
    mean = v.mean().item()
    if n_boot <= 0 or n <= 1:
        return mean, (mean, mean)
    # Sample with replacement
    idx = torch.randint(0, n, (n_boot, n), generator=generator)
    means = v[idx].mean(dim=-1)
    lo = means.kthvalue(max(int(alpha / 2 * n_boot), 1)).values.item()
    hi = means.kthvalue(max(int((1 - alpha / 2) * n_boot), 1)).values.item()
    return mean, (lo, hi)


# --------------------------------------------------------------------------- #
# Nearest-neighbor classification in signature space
# --------------------------------------------------------------------------- #

def nn_accuracy_binary(
    binary: torch.Tensor, labels: torch.Tensor, *,
    chunk: int = 256, exclude_self: bool = True,
) -> Tuple[float, float]:
    """Top-1 NN accuracy using boolean Jaccard as the similarity.

    Args:
        binary: (N, R) bool signatures (one K).
        labels: (N,) ground-truth labels.
    Returns:
        (accuracy, chance) where ``chance`` is computed from the empirical
        label frequency (``sum_c p_c^2``).
    """
    N, R = binary.shape
    b_int = binary.float()
    sums = b_int.sum(dim=-1)                                    # |A|
    correct = 0
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        a = b_int[start:end]                                    # (chunk, R)
        inter = a @ b_int.T                                     # (chunk, N)
        union = sums[start:end].unsqueeze(-1) + sums.unsqueeze(0) - inter
        sim = inter / (union + EPS)
        if exclude_self:
            for i, gi in enumerate(range(start, end)):
                sim[i, gi] = -1.0
        nn_idx = sim.argmax(dim=-1)
        correct += (labels[nn_idx] == labels[start:end]).sum().item()

    p_freq = torch.bincount(labels).float() / N
    chance = (p_freq.pow(2)).sum().item()
    return correct / N, chance
