"""Pinned ImageNet samples for the exact ResNet-34 Experiment 3 run.

The frozen ImageNet test index is sorted for efficient I/O.
To keep pilot selection independent from that storage order, this module pins a
full permutation of its 10,000 entries with seed 3403.  Pilot is permutation
positions ``[0, 100)`` and main is ``[100, 600)``.  Selected records may then be
sorted by ``(shard, row)`` for loading; ``sample_id``, ``master_index`` and
``selection_position`` continue to identify the originally selected samples.
"""

from __future__ import annotations

import argparse
import bisect
import io
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .io_utils import atomic_write_json, canonical_json_sha256, load_json, sha256_file


MANIFEST_SCHEMA_VERSION = 1
PERMUTATION_SEED = 3403
PERMUTATION_ALGORITHM = "python_random_mt19937_shuffle_v1"
EXPECTED_MASTER_COUNT = 10_000
DEFAULT_SPLITS: dict[str, dict[str, int]] = {
    "smoke": {"start": 0, "count": 2},
    "benchmark": {"start": 0, "count": 10},
    "pilot": {"start": 0, "count": 100},
    "main": {"start": 100, "count": 500},
    "extension": {"start": 100, "count": 1_000},
    "extension_append": {"start": 600, "count": 500},
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_v1_transform() -> transforms.Compose:
    """Preprocessing associated with torchvision IMAGENET1K_V1 weights."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One selected sample, independent of its eventual processing order."""

    shard: str
    row: int
    label: int
    master_index: int
    sample_id: str
    selection_position: int


@dataclass(frozen=True)
class LoadedManifest:
    """Validated manifest together with its immutable master sample index."""

    path: Path
    source_path: Path
    source_sha256: str
    seed: int
    permutation: tuple[int, ...]
    splits: Mapping[str, Mapping[str, int]]
    source_entries: tuple[Mapping[str, Any], ...]

    @property
    def count(self) -> int:
        return len(self.permutation)


def _validate_source_blob(blob: Any, expected_count: int | None) -> list[Mapping[str, Any]]:
    if not isinstance(blob, dict) or not isinstance(blob.get("entries"), list):
        raise ValueError("sample index must be an object containing entries[]")
    entries = blob["entries"]
    if expected_count is not None and len(entries) != expected_count:
        raise ValueError(
            f"sample index has {len(entries)} entries; expected {expected_count}"
        )
    required = {"shard", "row", "label"}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError(f"sample index entry {idx} lacks {sorted(required)}")
        if int(entry["row"]) < 0:
            raise ValueError(f"sample index entry {idx} has a negative row")
    return entries


def _stable_sample_id(
    source_sha256: str,
    master_index: int,
    entry: Mapping[str, Any],
) -> str:
    identity = {
        "source_sha256": source_sha256,
        "master_index": int(master_index),
        "shard": str(entry["shard"]),
        "row": int(entry["row"]),
    }
    suffix = canonical_json_sha256(identity)[:16]
    return f"r34-{master_index:05d}-{suffix}"


def build_permutation_manifest(
    sample_index_path: str | os.PathLike[str],
    *,
    seed: int = PERMUTATION_SEED,
    expected_count: int | None = EXPECTED_MASTER_COUNT,
    source_reference: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic full permutation manifest in memory."""
    source = Path(sample_index_path)
    blob = load_json(source)
    entries = _validate_source_blob(blob, expected_count)
    permutation = list(range(len(entries)))
    random.Random(int(seed)).shuffle(permutation)
    splits = {name: dict(bounds) for name, bounds in DEFAULT_SPLITS.items()}
    # Small fixtures can still test the machinery but cannot claim official
    # ranges that extend beyond them.  Keep every fully contained split.
    splits = {
        name: bounds
        for name, bounds in splits.items()
        if bounds["start"] + bounds["count"] <= len(entries)
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_sample_index": source_reference or str(source.resolve()),
        "source_sha256": sha256_file(source),
        "seed": int(seed),
        "permutation_algorithm": PERMUTATION_ALGORITHM,
        "num_entries": len(entries),
        "splits": splits,
        "permutation": permutation,
    }


def write_permutation_manifest(
    sample_index_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    seed: int = PERMUTATION_SEED,
    expected_count: int | None = EXPECTED_MASTER_COUNT,
) -> Path:
    """Build and atomically save the deterministic permutation manifest."""
    source = Path(sample_index_path).resolve()
    output = Path(output_path)
    relative_source = os.path.relpath(source, start=output.parent.resolve())
    manifest = build_permutation_manifest(
        source,
        seed=seed,
        expected_count=expected_count,
        source_reference=Path(relative_source).as_posix(),
    )
    return atomic_write_json(manifest, output)


def _resolve_source_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    override: str | os.PathLike[str] | None,
) -> Path:
    if override is not None:
        return Path(override).resolve()
    raw = Path(str(manifest["source_sample_index"]))
    if raw.is_absolute():
        return raw
    return (manifest_path.parent / raw).resolve()


def load_permutation_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    sample_index_path: str | os.PathLike[str] | None = None,
    verify_source: bool = True,
) -> LoadedManifest:
    """Load and fully validate a permutation and its source sample index."""
    path = Path(manifest_path).resolve()
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version={manifest.get('schema_version')!r}"
        )
    source_path = _resolve_source_path(path, manifest, sample_index_path)
    expected_sha = str(manifest.get("source_sha256", "")).lower()
    actual_sha = sha256_file(source_path)
    if verify_source and actual_sha != expected_sha:
        raise ValueError(
            "sample-index SHA256 mismatch: "
            f"manifest={expected_sha}, actual={actual_sha}"
        )
    source_blob = load_json(source_path)
    recorded_count = int(manifest.get("num_entries", -1))
    entries = _validate_source_blob(source_blob, recorded_count)

    raw_permutation = manifest.get("permutation")
    if not isinstance(raw_permutation, list):
        raise ValueError("manifest permutation must be a list")
    permutation = tuple(int(index) for index in raw_permutation)
    if len(permutation) != recorded_count or set(permutation) != set(range(recorded_count)):
        raise ValueError("manifest permutation is not a complete 0..N-1 permutation")
    if manifest.get("permutation_algorithm") != PERMUTATION_ALGORITHM:
        raise ValueError(
            "unsupported permutation_algorithm="
            f"{manifest.get('permutation_algorithm')!r}"
        )
    expected_permutation = list(range(recorded_count))
    random.Random(int(manifest["seed"])).shuffle(expected_permutation)
    if permutation != tuple(expected_permutation):
        raise ValueError("manifest permutation does not match its recorded seed/algorithm")

    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("manifest splits must be an object")
    for name, bounds in splits.items():
        if not isinstance(bounds, dict) or not {"start", "count"}.issubset(bounds):
            raise ValueError(f"split {name!r} must contain start and count")
        start, count = int(bounds["start"]), int(bounds["count"])
        if start < 0 or count < 0 or start + count > recorded_count:
            raise ValueError(f"split {name!r} is outside the permutation")

    return LoadedManifest(
        path=path,
        source_path=source_path,
        source_sha256=actual_sha,
        seed=int(manifest["seed"]),
        permutation=permutation,
        splits=splits,
        source_entries=tuple(entries),
    )


def select_records(
    loaded: LoadedManifest,
    *,
    split: str | None = None,
    start: int | None = None,
    max_images: int | None = None,
    sort_for_io: bool = False,
) -> list[SampleRecord]:
    """Select by permutation position, optionally sorting only for physical I/O.

    ``split`` is mutually exclusive with ``start``/``max_images``.  With no
    selector, the complete 10,000-entry permutation is returned.
    """
    if split is not None and (start is not None or max_images is not None):
        raise ValueError("split is mutually exclusive with start/max_images")
    if split is not None:
        if split not in loaded.splits:
            raise ValueError(
                f"unknown split {split!r}; available: {sorted(loaded.splits)}"
            )
        bounds = loaded.splits[split]
        first, count = int(bounds["start"]), int(bounds["count"])
    else:
        first = 0 if start is None else int(start)
        count = loaded.count - first if max_images is None else int(max_images)
    if first < 0 or first > loaded.count:
        raise ValueError(f"start={first} is outside [0, {loaded.count}]")
    if count < 0:
        raise ValueError("max_images/count must be non-negative")
    stop = min(first + count, loaded.count)

    records: list[SampleRecord] = []
    for position in range(first, stop):
        master_index = loaded.permutation[position]
        entry = loaded.source_entries[master_index]
        records.append(
            SampleRecord(
                shard=str(entry["shard"]),
                row=int(entry["row"]),
                label=int(entry["label"]),
                master_index=master_index,
                sample_id=_stable_sample_id(
                    loaded.source_sha256, master_index, entry
                ),
                selection_position=position,
            )
        )
    if sort_for_io:
        records.sort(key=lambda record: (record.shard, record.row))
    return records


def load_selected_records(
    manifest_path: str | os.PathLike[str],
    *,
    sample_index_path: str | os.PathLike[str] | None = None,
    split: str | None = None,
    start: int | None = None,
    max_images: int | None = None,
    sort_for_io: bool = False,
    verify_source: bool = True,
) -> list[SampleRecord]:
    """Convenience composition of manifest validation and selection."""
    loaded = load_permutation_manifest(
        manifest_path,
        sample_index_path=sample_index_path,
        verify_source=verify_source,
    )
    return select_records(
        loaded,
        split=split,
        start=start,
        max_images=max_images,
        sort_for_io=sort_for_io,
    )


class ManifestParquetDataset(Dataset):
    """Parquet-backed dataset with a one-row-group read-through cache.

    The official manifest selection should normally be passed with
    ``sort_for_io=True``.  Consecutive records from the same row group then share
    one decoded Arrow table.  Only the current row group is retained, keeping
    host memory bounded.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        transform: Callable[[Any], torch.Tensor] | None = None,
        *,
        verify_label: bool = True,
    ) -> None:
        self.records = list(records)
        self.transform = transform if transform is not None else imagenet_v1_transform()
        self.verify_label = bool(verify_label)
        self._current_shard: str | None = None
        self._current_parquet: Any = None
        self._row_group_stops: tuple[int, ...] = ()
        self._current_row_group: int | None = None
        self._current_table: Any = None
        self._row_group_read_count = 0

    def __len__(self) -> int:
        return len(self.records)

    @property
    def row_group_read_count(self) -> int:
        """Number of physical row-group reads (primarily a test/profile aid)."""
        return self._row_group_read_count

    def _open_shard(self, shard: str) -> None:
        if shard == self._current_shard:
            return
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(shard)
        cumulative = 0
        stops: list[int] = []
        for row_group in range(parquet.num_row_groups):
            cumulative += parquet.metadata.row_group(row_group).num_rows
            stops.append(cumulative)
        self._current_shard = shard
        self._current_parquet = parquet
        self._row_group_stops = tuple(stops)
        self._current_row_group = None
        self._current_table = None

    def _locate_row(self, row: int) -> tuple[int, int]:
        if row < 0:
            raise IndexError(f"negative parquet row {row}")
        row_group = bisect.bisect_right(self._row_group_stops, row)
        if row_group >= len(self._row_group_stops):
            total = self._row_group_stops[-1] if self._row_group_stops else 0
            raise IndexError(f"row {row} out of range; shard has {total} rows")
        group_start = 0 if row_group == 0 else self._row_group_stops[row_group - 1]
        return row_group, row - group_start

    def _read_row(self, record: SampleRecord) -> tuple[Any, int]:
        self._open_shard(record.shard)
        row_group, local_row = self._locate_row(record.row)
        if row_group != self._current_row_group:
            names = set(self._current_parquet.schema_arrow.names)
            columns = ["image"] + (["label"] if "label" in names else [])
            self._current_table = self._current_parquet.read_row_group(
                row_group, columns=columns
            )
            self._current_row_group = row_group
            self._row_group_read_count += 1
        image_value = self._current_table["image"][local_row].as_py()
        if "label" in self._current_table.column_names:
            label = int(self._current_table["label"][local_row].as_py())
        else:
            label = record.label
        if self.verify_label and label != record.label:
            raise ValueError(
                f"label mismatch for {record.sample_id}: "
                f"manifest={record.label}, parquet={label}"
            )
        return image_value, label

    @staticmethod
    def _decode_image(image_value: Any):
        from PIL import Image

        value = image_value
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                value = value["bytes"]
            elif value.get("path") is not None:
                value = value["path"]
        if isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(bytes(value))) as image:
                return image.convert("RGB")
        if isinstance(value, (str, os.PathLike)):
            with Image.open(value) as image:
                return image.convert("RGB")
        raise TypeError(f"unsupported parquet image value type: {type(value).__name__}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str]:
        record = self.records[index]
        image_value, label = self._read_row(record)
        image = self._decode_image(image_value)
        x = self.transform(image)
        return x, label, record.master_index, record.sample_id

    def __getstate__(self) -> dict[str, Any]:
        """Do not pickle open Arrow handles when DataLoader uses spawn workers."""
        state = self.__dict__.copy()
        state.update(
            {
                "_current_shard": None,
                "_current_parquet": None,
                "_row_group_stops": (),
                "_current_row_group": None,
                "_current_table": None,
                "_row_group_read_count": 0,
            }
        )
        return state


def build_manifest_loader(
    manifest_path: str | os.PathLike[str],
    *,
    sample_index_path: str | os.PathLike[str] | None = None,
    split: str | None = None,
    start: int | None = None,
    max_images: int | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    sort_for_io: bool = True,
    verify_source: bool = True,
    verify_label: bool = True,
    transform: Callable[[Any], torch.Tensor] | None = None,
) -> tuple[DataLoader, list[SampleRecord]]:
    """Build a non-shuffling loader and return its exact records as provenance."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    records = load_selected_records(
        manifest_path,
        sample_index_path=sample_index_path,
        split=split,
        start=start,
        max_images=max_images,
        sort_for_io=sort_for_io,
        verify_source=verify_source,
    )
    dataset = ManifestParquetDataset(
        records, transform=transform, verify_label=verify_label
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return loader, records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-index",
        type=Path,
        default=Path("data/indices/imagenet_test_10000.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/indices/resnet34_test_splits_seed3403.json"),
    )
    parser.add_argument("--seed", type=int, default=PERMUTATION_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = write_permutation_manifest(
        args.sample_index,
        args.output,
        seed=args.seed,
        expected_count=EXPECTED_MASTER_COUNT,
    )
    manifest = load_permutation_manifest(output)
    print(
        f"wrote {output} ({manifest.count} entries, seed={manifest.seed}, "
        f"source_sha256={manifest.source_sha256})"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_SPLITS",
    "EXPECTED_MASTER_COUNT",
    "LoadedManifest",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestParquetDataset",
    "PERMUTATION_SEED",
    "PERMUTATION_ALGORITHM",
    "SampleRecord",
    "build_manifest_loader",
    "build_permutation_manifest",
    "imagenet_v1_transform",
    "load_permutation_manifest",
    "load_selected_records",
    "select_records",
    "write_permutation_manifest",
]
