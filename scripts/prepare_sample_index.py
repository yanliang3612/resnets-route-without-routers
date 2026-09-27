"""Step 2 -- sample images from the parquet shards downloaded in step_1.

Two modes:

* ``--mode uniform`` (default, works on any split including the unlabeled
  test split): sample ``--num-images`` rows uniformly without replacement
  from the whole union of shards.  ``label`` carries the parquet's value
  unchanged (which is ``-1`` for every row in the test split).
* ``--mode per-class`` (requires labels in [0, num_classes), like the
  validation split): bucket rows by class and sample ``--per-class`` per
  class.  Refuses to run if no rows have valid labels (e.g. test split,
  where every label is ``-1``).

The result is written to ``sample_index.json``:

    {
      "config": {... seed, mode, parquet_dir, pattern, ...},
      "entries": [
        {"shard": "/abs/path.parquet", "row": 17, "label": 0},
        ...
      ]
    }

step_3 reads this file verbatim, so the experiment is fully reproducible from
``(parquet shards, seed, mode, per-class / num-images)``.

Usage:
    # uniform on test (test has no real labels) -- default
    python -m experiment_1.step_2_sample \\
        --parquet-dir /root/userdata/liangyan/MOE-ResNet/imagenet-1k/data \\
        --pattern 'test-*.parquet' \\
        --mode uniform --num-images 10000 --seed 0

    # per-class on validation (requires labeled shards)
    python -m experiment_1.step_2_sample \\
        --pattern 'val_images-*.parquet' \\
        --mode per-class --per-class 10 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", type=Path,
                   default=Path(os.environ.get("IMAGENET_PARQUET_DIR", "imagenet-1k/data")))
    p.add_argument("--pattern", default="test-*.parquet",
                   help="filename glob inside --parquet-dir (e.g. 'test-*.parquet' "
                        "for the unlabeled test split, 'val_images-*.parquet' "
                        "or 'validation-*.parquet' for the labeled val split)")
    p.add_argument("--mode", choices=["per-class", "uniform"], default="uniform",
                   help="per-class needs a labeled split (val); uniform works "
                        "on any split, including unlabeled test")
    p.add_argument("--per-class", type=int, default=10,
                   help="(per-class mode) images per class")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--num-images", type=int, default=10000,
                   help="(uniform mode) total images to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path,
                   default=Path("data/indices/sample_index.json"))
    return p.parse_args()


def discover_shards(parquet_dir: Path, pattern: str) -> List[str]:
    paths = sorted(glob.glob(str(parquet_dir / pattern)))
    if not paths:
        raise FileNotFoundError(
            f"no parquet shards found under {parquet_dir} (pattern={pattern!r}); "
            "check --parquet-dir and --pattern"
        )
    return paths


def _shard_columns(shard: str) -> List[str]:
    return [f.name for f in pq.ParquetFile(shard).schema_arrow]


def index_rows(shards: List[str], require_label: bool) -> Tuple[
    Dict[int, List[Tuple[str, int]]], List[Tuple[str, int]]
]:
    """Single pass over every shard.

    Returns ``(buckets_by_label, all_rows)``.  ``buckets_by_label`` only
    includes rows whose label is in ``[0, inf)`` (so ``-1`` placeholders from
    the unlabeled test split are excluded).  ``all_rows`` always contains every
    ``(shard, row)`` pair, used by uniform sampling.
    """
    buckets: Dict[int, List[Tuple[str, int]]] = {}
    all_rows: List[Tuple[str, int]] = []
    for shard in shards:
        pf = pq.ParquetFile(shard)
        cols = [f.name for f in pf.schema_arrow]
        has_label = "label" in cols
        offset = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            if has_label:
                tbl = pf.read_row_group(rg, columns=["label"])
                labels = tbl["label"].to_pylist()
                for i, lab in enumerate(labels):
                    lab_i = int(lab) if lab is not None else -1
                    if lab_i >= 0:
                        buckets.setdefault(lab_i, []).append((shard, offset + i))
                    all_rows.append((shard, offset + i))
            else:
                for i in range(n):
                    all_rows.append((shard, offset + i))
            offset += n
        print(f"  scanned {os.path.basename(shard)} -> {offset} rows "
              f"(label_col={has_label})")

    if require_label and not buckets:
        raise RuntimeError(
            "mode=per-class requires labels in [0, num_classes), but no rows "
            "have a valid label.  This typically means you pointed at the "
            "test split (every test row has label = -1).  Re-run with "
            "--mode uniform, or pass --pattern 'val_images-*.parquet' to "
            "use the labeled validation split."
        )
    return buckets, all_rows


def sample_per_class(
    buckets: Dict[int, List[Tuple[str, int]]],
    per_class: int,
    num_classes: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    entries: List[dict] = []
    missing: List[int] = []
    for cls in range(num_classes):
        rows = buckets.get(cls, [])
        if len(rows) < per_class:
            missing.append(cls)
            picked = rows
        else:
            picked = rng.sample(rows, per_class)
        for shard, row in picked:
            entries.append({"shard": shard, "row": row, "label": cls})
    if missing:
        print(f"WARNING: {len(missing)} classes had < {per_class} rows; "
              f"first 5 = {missing[:5]}")
    entries.sort(key=lambda e: (e["label"], e["shard"], e["row"]))
    return entries


def sample_uniform(
    rows: List[Tuple[str, int]], n: int, seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    if n > len(rows):
        print(f"WARNING: requested {n} > available {len(rows)}; taking all rows")
        picked = list(rows)
    else:
        picked = rng.sample(rows, n)
    entries = [{"shard": s, "row": r, "label": -1} for s, r in picked]
    entries.sort(key=lambda e: (e["shard"], e["row"]))
    return entries


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    shards = discover_shards(args.parquet_dir, args.pattern)
    print(f"[step_2] mode={args.mode}  found {len(shards)} parquet shard(s) "
          f"under {args.parquet_dir}")
    buckets, all_rows = index_rows(shards, require_label=(args.mode == "per-class"))

    if args.mode == "per-class":
        print(f"[step_2] indexed {sum(len(v) for v in buckets.values())} labeled "
              f"rows across {len(buckets)} classes")
        entries = sample_per_class(buckets, args.per_class, args.num_classes, args.seed)
        expected = args.num_classes * args.per_class
    else:
        print(f"[step_2] indexed {len(all_rows)} rows total")
        entries = sample_uniform(all_rows, args.num_images, args.seed)
        expected = args.num_images
    print(f"[step_2] sampled {len(entries)} rows (expected {expected})")

    payload = {
        "config": {
            "parquet_dir": str(args.parquet_dir),
            "pattern": args.pattern,
            "shards": shards,
            "mode": args.mode,
            "per_class": args.per_class,
            "num_classes": args.num_classes,
            "num_images": args.num_images,
            "seed": args.seed,
        },
        "entries": entries,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[step_2] wrote {args.output}")


if __name__ == "__main__":
    main()
