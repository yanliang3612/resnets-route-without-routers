"""Dataset loaders for Experiment 1.

Two paths:

1. ``--parquet-glob '...val_images-*.parquet'`` reads the HF
   ``ILSVRC/imagenet-1k`` validation parquet shards (each row has a JPEG-encoded
   ``image`` column and an integer ``label`` column).  step_2 writes a
   ``sample_index.json`` that pins the exact (shard_path, row_index, label)
   tuples step_3 will use.
2. ``--synthetic`` falls back to deterministic random tensors so the pipeline
   can be smoke-tested without the dataset.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_val_transforms() -> transforms.Compose:
    """The standard transform that matches torchvision IMAGENET1K_V1 weights."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


@dataclass
class SampleEntry:
    shard: str            # absolute path to the parquet shard
    row: int              # row index within the shard
    label: int            # ImageNet class id (0..999)


# -------- parquet-backed dataset (pinned by step_2) --------

class ParquetSubset(Dataset):
    """Reads a fixed list of ``SampleEntry``s out of HF imagenet-1k parquet shards."""

    def __init__(self, entries: Sequence[SampleEntry]) -> None:
        from PIL import Image  # local import keeps top-level deps minimal
        import pyarrow.parquet as pq
        self._Image = Image
        self._pq = pq
        self.entries = list(entries)
        self.transform = imagenet_val_transforms()
        self._shard_cache: dict[str, "pq.ParquetFile"] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _open(self, shard: str):
        f = self._shard_cache.get(shard)
        if f is None:
            f = self._pq.ParquetFile(shard)
            self._shard_cache[shard] = f
        return f

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        e = self.entries[idx]
        pf = self._open(e.shard)
        # Read the single needed row.  pyarrow's read API is row-group aware;
        # the per-shard cost is dominated by image decoding, not the seek.
        rg, local = _row_to_group(pf, e.row)
        tbl = pf.read_row_group(rg, columns=["image", "label"])
        img_struct = tbl["image"][local].as_py()
        # HF imagenet-1k parquet format: image is {"bytes": <jpeg>, "path": ...}
        if isinstance(img_struct, dict) and "bytes" in img_struct:
            raw = img_struct["bytes"]
        else:
            raw = img_struct
        img = self._Image.open(io.BytesIO(raw)).convert("RGB")
        return self.transform(img), int(e.label)


def _row_to_group(pf, row_idx: int) -> Tuple[int, int]:
    """Translate a global row index into (row_group, row_in_group)."""
    cum = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if row_idx < cum + n:
            return rg, row_idx - cum
        cum += n
    raise IndexError(f"row {row_idx} out of range; shard has {cum} rows")


def load_sample_index(
    path: str | Path,
    parquet_dir: str | Path | None = None,
) -> List[SampleEntry]:
    """Load pinned rows and optionally relocate the parquet shard directory.

    The published JSON records retain the original absolute paths for exact
    provenance.  ``parquet_dir`` (or ``IMAGENET_PARQUET_DIR``) replaces only
    each shard's parent directory, keeping its filename and pinned row intact.
    """
    blob = json.loads(Path(path).read_text())
    root = parquet_dir or os.environ.get("IMAGENET_PARQUET_DIR")
    entries = [SampleEntry(**e) for e in blob["entries"]]
    if root is None:
        return entries
    root_path = Path(root).expanduser().resolve()
    return [
        SampleEntry(
            shard=str(root_path / Path(entry.shard).name),
            row=entry.row,
            label=entry.label,
        )
        for entry in entries
    ]


# -------- synthetic fallback (smoke test) --------

class _SyntheticImageNet(Dataset):
    def __init__(self, length: int, seed: int = 0) -> None:
        self.length = int(length)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed + idx)
        x = torch.randn(3, 224, 224, generator=g)
        y = int(torch.randint(0, 1000, (1,), generator=g).item())
        return x, y


# -------- public builder used by step_3 --------

def build_loader(
    sample_index: str | None,
    synthetic_n: int | None,
    batch_size: int,
    seed: int,
    num_workers: int = 4,
    parquet_dir: str | Path | None = None,
) -> Tuple[DataLoader, int]:
    if sample_index is not None:
        entries = load_sample_index(sample_index, parquet_dir=parquet_dir)
        ds: Dataset = ParquetSubset(entries)
    else:
        assert synthetic_n is not None
        ds = _SyntheticImageNet(synthetic_n, seed=seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )
    return loader, len(ds)
