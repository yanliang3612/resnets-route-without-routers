"""Analyze compact ResNet-34 Experiment-4 Top-K signatures.

The evaluator stores only the ranked top ``max(K)`` indices and their vector
mass weights.  This program builds temporary rank lookup tables, computes the
same metrics as the ResNet-18 implementation, and writes a compatible
``overlap.json`` plus compact numerical diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from resnet_routes.resnet18.analyze_signatures import render_table

from .common import atomic_write_json, mean_se, write_result_checksums


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signature-dir", type=Path,
        default=Path("experiments/input_dependent_expert_sets/generated/resnet34"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=45_000)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nn-k", type=int, default=32,
        help="absolute K for the ResNet-18-compatible nearest-neighbor diagnostic",
    )
    parser.add_argument("--pair-chunk", type=int, default=256)
    parser.add_argument(
        "--scratch-dir", type=Path, default=None,
        help="parent for temporary rank lookup arrays (defaults to signature-dir)",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.signature_dir
    if args.max_pairs < 1 or args.n_boot < 0 or args.pair_chunk < 1:
        parser.error("pair/bootstrap/chunk settings are invalid")
    if args.nn_k < 1:
        parser.error("nn-k must be positive")
    return args


def _load_array(root: Path, metadata: Mapping[str, Any], name: str) -> np.ndarray:
    path = root / str(metadata["arrays"][name])
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def build_rank_lookup(
    indices: np.ndarray,
    residual_count: int,
    output: Path,
    *,
    row_chunk: int = 128,
) -> np.memmap:
    """Build ``rank[i, subset]``; unrecorded subsets receive a sentinel."""
    n, max_k = indices.shape
    dtype = np.uint16 if max_k < np.iinfo(np.uint16).max else np.uint32
    sentinel = np.iinfo(dtype).max
    rank = np.lib.format.open_memmap(
        output, mode="w+", dtype=dtype, shape=(n, residual_count)
    )
    rank.fill(sentinel)
    positions = np.arange(max_k, dtype=dtype)
    for start in range(0, n, row_chunk):
        stop = min(start + row_chunk, n)
        block = np.asarray(indices[start:stop], dtype=np.int64)
        if block.size and (int(block.min()) < 0 or int(block.max()) >= residual_count):
            raise ValueError("signature ranking contains an out-of-range subset index")
        rows = np.arange(start, stop)[:, None]
        rank[rows, block] = positions[None, :]
    rank.flush()
    return rank


def sample_same_class_pairs(
    labels: np.ndarray, max_pairs: int, rng: np.random.Generator
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if idx.size < 2:
            continue
        left, right = np.triu_indices(idx.size, k=1)
        chunks.append(np.column_stack((idx[left], idx[right])))
    if not chunks:
        raise ValueError("selection contains no same-class image pairs")
    pairs = np.concatenate(chunks).astype(np.int64, copy=False)
    if pairs.shape[0] > max_pairs:
        pairs = pairs[rng.choice(pairs.shape[0], size=max_pairs, replace=False)]
    return pairs


def sample_different_class_pairs(
    labels: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    n = labels.size
    if n < 2 or np.unique(labels).size < 2:
        raise ValueError("different-class sampling requires at least two classes")
    output = np.empty((count, 2), dtype=np.int64)
    filled = 0
    while filled < count:
        block = max(1024, count - filled)
        left = rng.integers(0, n, size=block)
        right = rng.integers(0, n, size=block)
        keep = (left != right) & (labels[left] != labels[right])
        take = min(int(keep.sum()), count - filled)
        selected = np.flatnonzero(keep)[:take]
        output[filled:filled + take, 0] = left[selected]
        output[filled:filled + take, 1] = right[selected]
        filled += take
    return output


def sample_pairs_from_set(
    indices: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    if indices.size < 2:
        raise ValueError("a pair subset needs at least two images")
    output = np.empty((count, 2), dtype=np.int64)
    filled = 0
    while filled < count:
        block = max(1024, count - filled)
        left = rng.integers(0, indices.size, size=block)
        right = rng.integers(0, indices.size, size=block)
        keep = left != right
        take = min(int(keep.sum()), count - filled)
        selected = np.flatnonzero(keep)[:take]
        output[filled:filled + take, 0] = indices[left[selected]]
        output[filled:filled + take, 1] = indices[right[selected]]
        filled += take
    return output


def pair_jaccard(
    indices: np.ndarray,
    rank: np.ndarray,
    pairs: np.ndarray,
    k: int,
    *,
    chunk: int,
) -> np.ndarray:
    output = np.empty(pairs.shape[0], dtype=np.float64)
    for start in range(0, pairs.shape[0], chunk):
        stop = min(start + chunk, pairs.shape[0])
        left, right = pairs[start:stop, 0], pairs[start:stop, 1]
        left_indices = np.asarray(indices[left, :k], dtype=np.int64)
        right_ranks = np.asarray(rank[right[:, None], left_indices])
        intersection = (right_ranks < k).sum(axis=1, dtype=np.int64)
        output[start:stop] = intersection / np.maximum(2 * k - intersection, 1)
    return output


def pair_weighted_jaccard(
    indices: np.ndarray,
    rank: np.ndarray,
    weights: np.ndarray,
    pairs: np.ndarray,
    k: int,
    *,
    chunk: int,
) -> np.ndarray:
    output = np.empty(pairs.shape[0], dtype=np.float64)
    max_k = indices.shape[1]
    row_sums = np.asarray(weights[:, :k], dtype=np.float64).sum(axis=1)
    for start in range(0, pairs.shape[0], chunk):
        stop = min(start + chunk, pairs.shape[0])
        left, right = pairs[start:stop, 0], pairs[start:stop, 1]
        left_indices = np.asarray(indices[left, :k], dtype=np.int64)
        left_weights = np.asarray(weights[left, :k], dtype=np.float64)
        right_ranks = np.asarray(rank[right[:, None], left_indices], dtype=np.int64)
        common = right_ranks < k
        safe_ranks = np.minimum(right_ranks, max_k - 1)
        right_weights = np.asarray(weights[right[:, None], safe_ranks], dtype=np.float64)
        numerator = np.minimum(left_weights, right_weights) * common
        numerator = numerator.sum(axis=1)
        denominator = row_sums[left] + row_sums[right] - numerator
        output[start:stop] = numerator / np.maximum(denominator, 1e-30)
    return output


def global_jaccard(
    rank: np.ndarray,
    global_indices: np.ndarray,
    k: int,
    *,
    chunk: int,
) -> np.ndarray:
    output = np.empty(rank.shape[0], dtype=np.float64)
    selected = np.asarray(global_indices[:k], dtype=np.int64)
    for start in range(0, rank.shape[0], chunk):
        stop = min(start + chunk, rank.shape[0])
        intersection = (np.asarray(rank[start:stop, selected]) < k).sum(axis=1)
        output[start:stop] = intersection / np.maximum(2 * k - intersection, 1)
    return output


def _bootstrap_summary(
    values: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict[str, float | int]:
    base = mean_se(np.asarray(values, dtype=np.float64).tolist())
    if n_boot <= 0 or values.size <= 1:
        base["ci_lo"] = base["mean"]
        base["ci_hi"] = base["mean"]
        return base
    means = np.empty(n_boot, dtype=np.float64)
    # Chunk bootstrap draws to bound temporary memory for large pair sets.
    for start in range(0, n_boot, 16):
        stop = min(start + 16, n_boot)
        draw = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[draw].mean(axis=1)
    base["ci_lo"] = float(np.quantile(means, 0.025))
    base["ci_hi"] = float(np.quantile(means, 0.975))
    return base


def random_set_jaccard(
    residual_count: int, k: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    # Intersection of two independent uniform K-subsets is hypergeometric.
    intersection = rng.hypergeometric(k, residual_count - k, k, size=count)
    return intersection / np.maximum(2 * k - intersection, 1)


def nn_accuracy_sparse(
    indices: np.ndarray,
    labels: np.ndarray,
    k: int,
    residual_count: int,
) -> tuple[float, float]:
    """Exact binary-Jaccard nearest neighbor via sparse intersections."""
    from scipy.sparse import csr_matrix

    n = labels.size
    if n < 2:
        raise ValueError("nearest-neighbor accuracy requires at least two images")
    if k < 1 or k > indices.shape[1]:
        raise ValueError(f"K={k} is outside the stored ranking width")
    columns = np.asarray(indices[:, :k], dtype=np.int32).reshape(-1)
    indptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    data = np.ones(columns.size, dtype=np.int32)
    signatures = csr_matrix(
        (data, columns, indptr), shape=(n, residual_count), dtype=np.int32
    )
    intersections = (signatures @ signatures.T).toarray()
    np.fill_diagonal(intersections, -1)
    # All rows have cardinality K, so maximizing Jaccard equals maximizing
    # intersection.  This avoids materializing a second N x N float matrix.
    neighbor = intersections.argmax(axis=1)
    accuracy = float(np.mean(labels[neighbor] == labels))
    counts = np.bincount(labels)
    frequencies = counts / counts.sum()
    chance = float(np.square(frequencies).sum())
    return accuracy, chance


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.signature_dir / "signatures.json").read_text())
    indices_h = _load_array(args.signature_dir, metadata, "vector_indices")
    indices_v = _load_array(args.signature_dir, metadata, "scalar_indices")
    weights_h = _load_array(args.signature_dir, metadata, "vector_weights")
    labels = np.asarray(_load_array(args.signature_dir, metadata, "labels"), dtype=np.int64)
    pseudo = np.asarray(
        _load_array(args.signature_dir, metadata, "pseudo_labels"), dtype=np.int64
    )
    confidence = np.asarray(
        _load_array(args.signature_dir, metadata, "confidence"), dtype=np.float64
    )
    global_mass = np.asarray(
        _load_array(args.signature_dir, metadata, "global_mass_mean"), dtype=np.float64
    )
    k_list = [int(value) for value in metadata["K_list"]]
    n, max_k = indices_h.shape
    residual_count = int(metadata["num_residual_subsets"])
    if indices_v.shape != indices_h.shape or weights_h.shape != indices_h.shape:
        raise ValueError("compact signature arrays have inconsistent shapes")
    if n != int(metadata["num_images"]) or max_k != int(metadata["max_K"]):
        raise ValueError("compact signature shapes disagree with signatures.json")
    if labels.shape != (n,) or pseudo.shape != (n,) or confidence.shape != (n,):
        raise ValueError("compact signature metadata arrays have inconsistent lengths")
    if global_mass.shape != (residual_count,):
        raise ValueError("global mass vector has the wrong residual dimension")
    if not k_list or k_list != sorted(set(k_list)):
        raise ValueError("K_list must be non-empty, sorted, and unique")
    if k_list[0] < 1 or k_list[-1] > max_k:
        raise ValueError("K_list lies outside the stored ranking width")
    if args.nn_k > max_k:
        raise ValueError(
            f"nn-k={args.nn_k} exceeds the stored ranking width max_K={max_k}"
        )
    rng = np.random.default_rng(args.seed + 7)
    same_pairs = sample_same_class_pairs(labels, args.max_pairs, rng)
    diff_pairs = sample_different_class_pairs(labels, same_pairs.shape[0], rng)
    shuffled = labels[rng.permutation(n)]
    shuffle_same = sample_same_class_pairs(shuffled, same_pairs.shape[0], rng)
    shuffle_diff = sample_different_class_pairs(shuffled, shuffle_same.shape[0], rng)
    q33, q66 = np.quantile(confidence, [1.0 / 3.0, 2.0 / 3.0])
    easy = np.flatnonzero(confidence >= q66)
    hard = np.flatnonzero(confidence <= q33)
    condition_pairs = min(args.max_pairs, 30_000)
    easy_pairs = sample_pairs_from_set(easy, condition_pairs, rng)
    hard_pairs = sample_pairs_from_set(hard, condition_pairs, rng)
    global_order = np.argsort(-global_mass, kind="stable")

    scratch_parent = args.scratch_dir or args.signature_dir
    scratch_parent.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix=".exp4-ranks-", dir=scratch_parent) as temporary:
        temporary_path = Path(temporary)
        rank_h = build_rank_lookup(
            indices_h, residual_count, temporary_path / "rank_h.npy"
        )
        rank_v = build_rank_lookup(
            indices_v, residual_count, temporary_path / "rank_v.npy"
        )
        per_k: dict[str, Any] = {}
        for k in k_list:
            same_j = pair_jaccard(
                indices_h, rank_h, same_pairs, k, chunk=args.pair_chunk
            )
            diff_j = pair_jaccard(
                indices_h, rank_h, diff_pairs, k, chunk=args.pair_chunk
            )
            same_wj = pair_weighted_jaccard(
                indices_h, rank_h, weights_h, same_pairs, k, chunk=args.pair_chunk
            )
            diff_wj = pair_weighted_jaccard(
                indices_h, rank_h, weights_h, diff_pairs, k, chunk=args.pair_chunk
            )
            same_j_v = pair_jaccard(
                indices_v, rank_v, same_pairs, k, chunk=args.pair_chunk
            )
            diff_j_v = pair_jaccard(
                indices_v, rank_v, diff_pairs, k, chunk=args.pair_chunk
            )
            shuffled_same_j = pair_jaccard(
                indices_h, rank_h, shuffle_same, k, chunk=args.pair_chunk
            )
            shuffled_diff_j = pair_jaccard(
                indices_h, rank_h, shuffle_diff, k, chunk=args.pair_chunk
            )
            easy_j = pair_jaccard(
                indices_h, rank_h, easy_pairs, k, chunk=args.pair_chunk
            )
            hard_j = pair_jaccard(
                indices_h, rank_h, hard_pairs, k, chunk=args.pair_chunk
            )
            global_j = global_jaccard(
                rank_h, global_order, k, chunk=args.pair_chunk
            )
            random_j = random_set_jaccard(
                residual_count, k, min(2000, same_pairs.shape[0]), rng
            )
            per_k[str(k)] = {
                "same_jaccard": _bootstrap_summary(same_j, args.n_boot, rng),
                "diff_jaccard": _bootstrap_summary(diff_j, args.n_boot, rng),
                "same_wjaccard": _bootstrap_summary(same_wj, args.n_boot, rng),
                "diff_wjaccard": _bootstrap_summary(diff_wj, args.n_boot, rng),
                "same_jaccard_scalar": _bootstrap_summary(same_j_v, args.n_boot, rng),
                "diff_jaccard_scalar": _bootstrap_summary(diff_j_v, args.n_boot, rng),
                "random_jaccard": _bootstrap_summary(random_j, args.n_boot, rng),
                "shuffle_same_jaccard": _bootstrap_summary(
                    shuffled_same_j, args.n_boot, rng
                ),
                "shuffle_diff_jaccard": _bootstrap_summary(
                    shuffled_diff_j, args.n_boot, rng
                ),
                "global_jaccard_per_image": _bootstrap_summary(
                    global_j, args.n_boot, rng
                ),
                "easy_jaccard": _bootstrap_summary(easy_j, args.n_boot, rng),
                "hard_jaccard": _bootstrap_summary(hard_j, args.n_boot, rng),
                "delta_class": float(same_j.mean() - diff_j.mean()),
                "delta_shuffle": float(
                    shuffled_same_j.mean() - shuffled_diff_j.mean()
                ),
                "lift_class": float(same_j.mean() / max(diff_j.mean(), 1e-30)),
                "delta_class_weighted": float(same_wj.mean() - diff_wj.mean()),
            }
            diagnostics[f"same_{k}"] = same_j.astype(np.float32)
            diagnostics[f"diff_{k}"] = diff_j.astype(np.float32)
            diagnostics[f"global_{k}"] = global_j.astype(np.float32)
            print(f"[exp4 analyze] K={k} delta={per_k[str(k)]['delta_class']:.5f}")

        nn_k = args.nn_k
        nn_h, chance = nn_accuracy_sparse(indices_h, labels, nn_k, residual_count)
        nn_v, _ = nn_accuracy_sparse(indices_v, labels, nn_k, residual_count)

    summary = {
        "config": {
            "signature_dir": str(args.signature_dir),
            "n_images": n,
            "n_pairs": int(same_pairs.shape[0]),
            "K_list": k_list,
            "num_residual_subsets": residual_count,
            "nn_K": nn_k,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "compact_signature_format": metadata["format"],
        },
        "full_mask_top1_acc": float(np.mean(pseudo == labels)),
        "per_K": per_k,
        "nn_accuracy_K": {
            "K": nn_k,
            "binary_vector": nn_h,
            "binary_scalar": nn_v,
            "chance": chance,
        },
    }
    atomic_write_json(summary, args.output_dir / "overlap.json")
    (args.output_dir / "table_4.md").write_text(render_table(summary))
    np.savez_compressed(args.output_dir / "overlap_diagnostics.npz", **diagnostics)
    write_result_checksums(
        args.output_dir,
        ("overlap.json", "table_4.md", "overlap_diagnostics.npz"),
        checksum_name="ANALYSIS_SHA256SUMS",
    )
    print(
        f"saved Experiment-4 analysis; vector NN={nn_h:.4f}, "
        f"scalar NN={nn_v:.4f}, chance={chance:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_rank_lookup", "global_jaccard", "nn_accuracy_sparse",
    "pair_jaccard", "pair_weighted_jaccard", "sample_different_class_pairs",
    "sample_same_class_pairs",
]
