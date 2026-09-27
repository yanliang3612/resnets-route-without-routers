"""Common data, provenance, checkpoint, and summary helpers.

The committed ImageNet sample indices contain absolute paths from the machine
on which they were created.  ``--parquet-dir`` intentionally remaps every
entry by basename so the experiments remain runnable after the repository is
moved without changing the frozen sample identities.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import (
    ManifestParquetDataset,
    SampleRecord,
    load_permutation_manifest,
    select_records,
)
from .model import GATE_NAMES
from .io_utils import (
    atomic_save_checkpoint,
    atomic_write_json,
    canonical_json_sha256,
    load_json,
    load_torch_checkpoint,
    sha256_file,
    write_sha256sums,
)


L = len(GATE_NAMES)
NUM_MASKS = 1 << L
NUM_RESIDUAL = NUM_MASKS - 1
EPS = 1e-12
CI95 = 1.959963984540054


@dataclass(frozen=True)
class Selection:
    records: tuple[SampleRecord, ...]
    source_sha256: str
    selector: Mapping[str, Any]
    manifest_sha256: str | None = None


def configure_determinism(seed: int, *, cudnn_benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def software_hardware_metadata(device: torch.device | str) -> dict[str, Any]:
    import torchvision

    resolved = torch.device(device)
    metadata: dict[str, Any] = {
        "device": str(resolved),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda": torch.version.cuda,
    }
    if resolved.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved)
        metadata.update(
            {
                "gpu": properties.name,
                "gpu_total_bytes": int(properties.total_memory),
                "compute_capability": list(torch.cuda.get_device_capability(resolved)),
            }
        )
    return metadata


def _stable_sample_id(source_sha256: str, master_index: int, entry: Mapping[str, Any]) -> str:
    payload = {
        "source_sha256": source_sha256,
        "master_index": int(master_index),
        "shard": Path(str(entry["shard"])).name,
        "row": int(entry["row"]),
    }
    return f"r34-suite-{master_index:05d}-{canonical_json_sha256(payload)[:16]}"


def _remap_record(record: SampleRecord, parquet_dir: Path | None) -> SampleRecord:
    if parquet_dir is None:
        return record
    shard = (parquet_dir / Path(record.shard).name).resolve()
    if not shard.is_file():
        raise FileNotFoundError(
            f"missing parquet shard {shard}; check --parquet-dir"
        )
    return replace(record, shard=str(shard))


def load_manifest_selection(
    sample_index: Path,
    selection_manifest: Path,
    *,
    split: str,
    parquet_dir: Path | None,
) -> Selection:
    manifest = load_permutation_manifest(
        selection_manifest,
        sample_index_path=sample_index,
        verify_source=True,
    )
    records = select_records(manifest, split=split, sort_for_io=False)
    records = [_remap_record(record, parquet_dir) for record in records]
    # I/O order may differ from the frozen logical order.  selection_position
    # remains the stable output slot and identity anchor.
    records.sort(key=lambda record: (record.shard, record.row))
    return Selection(
        records=tuple(records),
        source_sha256=manifest.source_sha256,
        manifest_sha256=sha256_file(selection_manifest),
        selector={"kind": "manifest_split", "split": split},
    )


def load_index_selection(
    sample_index: Path,
    *,
    parquet_dir: Path | None,
    per_class: int | None,
    max_images: int | None,
    seed: int,
) -> Selection:
    """Load a direct sample index, optionally taking a balanced class subset.

    ``per_class`` is the recommended selector for label-aware Experiments 4/5.
    It preserves class balance and makes same-class pairs available.  Selection
    is deterministic and independent of physical Parquet I/O order.
    """
    blob = load_json(sample_index)
    entries = blob.get("entries") if isinstance(blob, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("sample index must be an object with non-empty entries[]")
    source_sha = sha256_file(sample_index)
    indexed = list(enumerate(entries))
    if per_class is not None:
        if per_class < 1:
            raise ValueError("per_class must be positive or omitted")
        buckets: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
        for master_index, entry in indexed:
            label = int(entry["label"])
            if label < 0:
                raise ValueError("per-class selection requires ground-truth labels")
            buckets.setdefault(label, []).append((master_index, entry))
        rng = random.Random(int(seed))
        selected: list[tuple[int, Mapping[str, Any]]] = []
        for label in sorted(buckets):
            bucket = list(buckets[label])
            rng.shuffle(bucket)
            if len(bucket) < per_class:
                raise ValueError(
                    f"class {label} has {len(bucket)} entries, needs {per_class}"
                )
            selected.extend(bucket[:per_class])
        indexed = selected
    elif max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be positive")
        indexed = indexed[: int(max_images)]

    records: list[SampleRecord] = []
    for position, (master_index, entry) in enumerate(indexed):
        record = SampleRecord(
            shard=str(entry["shard"]),
            row=int(entry["row"]),
            label=int(entry["label"]),
            master_index=int(master_index),
            sample_id=_stable_sample_id(source_sha, master_index, entry),
            selection_position=position,
        )
        records.append(_remap_record(record, parquet_dir))
    records.sort(key=lambda record: (record.shard, record.row))
    return Selection(
        records=tuple(records),
        source_sha256=source_sha,
        selector={
            "kind": "direct_index",
            "per_class": per_class,
            "max_images": max_images,
            "seed": int(seed),
        },
    )


class SyntheticDataset(Dataset):
    def __init__(self, count: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(count, 3, 224, 224, generator=generator)
        # Adjacent pairs share a label, making a small synthetic run useful for
        # Experiment 4's same-class diagnostics as well as for evaluation.
        self.labels = (
            torch.arange(count, dtype=torch.long).floor_divide(2).remainder(1000)
        )

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index]), index, f"synthetic-{index:05d}"


def build_loader(
    selection: Selection | None,
    *,
    synthetic_images: int | None,
    seed: int,
    num_workers: int,
    pin_memory: bool = True,
) -> tuple[DataLoader, dict[str, SampleRecord], int]:
    if synthetic_images is not None:
        if synthetic_images < 1:
            raise ValueError("synthetic_images must be positive")
        dataset: Dataset = SyntheticDataset(synthetic_images, seed)
        lookup = {
            f"synthetic-{index:05d}": SampleRecord(
                shard="<synthetic>", row=index, label=(index // 2) % 1000,
                master_index=index, sample_id=f"synthetic-{index:05d}",
                selection_position=index,
            )
            for index in range(synthetic_images)
        }
        total = synthetic_images
    else:
        if selection is None:
            raise ValueError("selection is required for a non-synthetic run")
        dataset = ManifestParquetDataset(selection.records, verify_label=True)
        lookup = {record.sample_id: record for record in selection.records}
        total = len(selection.records)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return loader, lookup, total


def sample_id_from_batch(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0])
    return str(value)


def mean_se(values: Sequence[float]) -> dict[str, float | int]:
    tensor = torch.as_tensor(list(values), dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError("cannot summarize an empty sequence")
    mean = tensor.mean()
    if tensor.numel() > 1:
        std = tensor.std(unbiased=True)
        se = std / math.sqrt(tensor.numel())
    else:
        std = torch.zeros((), dtype=torch.float64)
        se = torch.zeros((), dtype=torch.float64)
    return {
        "mean": float(mean),
        "std": float(std),
        "stderr": float(se),
        "se": float(se),
        "ci_low": float(mean - CI95 * se),
        "ci_high": float(mean + CI95 * se),
        "n": int(tensor.numel()),
        "count": int(tensor.numel()),
    }


def coordinate_summary(values: Sequence[Sequence[float]]) -> dict[str, Any]:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim != 2 or tensor.shape[0] == 0:
        raise ValueError("coordinate_summary expects a non-empty N x D array")
    mean = tensor.mean(dim=0)
    if tensor.shape[0] > 1:
        std = tensor.std(dim=0, unbiased=True)
        se = std / math.sqrt(tensor.shape[0])
    else:
        std = torch.zeros_like(mean)
        se = torch.zeros_like(mean)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "stderr": se.tolist(),
        "se": se.tolist(),
        "ci_low": (mean - CI95 * se).tolist(),
        "ci_high": (mean + CI95 * se).tolist(),
        "n": int(tensor.shape[0]),
        "count": int(tensor.shape[0]),
    }


def atomic_save_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def implementation_hashes(paths: Iterable[Path], repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        result[resolved.relative_to(repo_root.resolve()).as_posix()] = sha256_file(resolved)
    return result


def config_fingerprint(config: Mapping[str, Any]) -> str:
    return canonical_json_sha256(config)


def load_resume_checkpoint(path: Path, fingerprint: str) -> Mapping[str, Any]:
    state = load_torch_checkpoint(path, map_location="cpu")
    if state.get("config_fingerprint") != fingerprint:
        raise ValueError(
            "checkpoint configuration/code fingerprint differs from this run"
        )
    return state


def write_result_checksums(
    output_dir: Path,
    names: Sequence[str],
    *,
    checksum_name: str = "SHA256SUMS",
) -> Path:
    if Path(checksum_name).name != checksum_name:
        raise ValueError("checksum_name must be a plain filename")
    paths = [output_dir / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot checksum missing artifacts: {missing}")
    return write_sha256sums(paths, output_dir / checksum_name)


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


__all__ = [
    "CI95", "EPS", "L", "NUM_MASKS", "NUM_RESIDUAL", "Selection",
    "atomic_save_checkpoint", "atomic_save_parquet", "atomic_write_json",
    "build_loader", "config_fingerprint", "configure_determinism",
    "coordinate_summary", "implementation_hashes", "json_ready",
    "load_index_selection", "load_manifest_selection", "load_resume_checkpoint",
    "mean_se", "sample_id_from_batch", "sha256_file", "write_result_checksums",
    "software_hardware_metadata",
]
