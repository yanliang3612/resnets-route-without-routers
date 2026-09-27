"""Small, crash-safe I/O helpers for the ResNet-34 experiment.

The evaluator runs for hours, so a checkpoint must never replace the last good
checkpoint until the new file has been completely written and flushed.  The
helpers in this module write a sibling temporary file and commit it with
``os.replace`` (an atomic operation on a single filesystem).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


PathLike = str | os.PathLike[str]


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA256 hex digest of *path* without loading it all."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using a stable, whitespace-free representation."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_target(path: PathLike) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _sync_directory(directory: Path) -> None:
    """Best-effort directory fsync so the rename survives a machine crash."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems do not support fsync on directories.
        pass
    finally:
        os.close(fd)


def atomic_write_json(
    value: Any,
    path: PathLike,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> Path:
    """Serialize JSON and atomically replace *path*.

    If serialization or writing fails, the old target remains untouched and the
    temporary file is removed.
    """
    target = _prepare_target(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=indent,
                sort_keys=sort_keys,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        _sync_directory(target.parent)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_json(path: PathLike) -> Any:
    """Load a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_torch_save(value: Any, path: PathLike) -> Path:
    """Write a PyTorch object to *path* using an atomic sibling rename."""
    target = _prepare_target(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        _sync_directory(target.parent)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_torch_checkpoint(path: PathLike, *, map_location: Any = "cpu") -> Any:
    """Load a trusted experiment checkpoint.

    Experiment checkpoints contain optimizer-free Python dictionaries and are
    produced locally by :func:`atomic_torch_save`.  ``weights_only=False`` is
    explicit because these are not model weight files.
    """
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def atomic_save_checkpoint(value: Any, path: PathLike) -> Path:
    """Semantic alias used by the evaluator's checkpoint/resume path."""
    return atomic_torch_save(value, path)


def write_sha256sums(paths: Iterable[PathLike], output: PathLike) -> Path:
    """Atomically write a conventional ``SHA256SUMS`` file.

    Paths are sorted by their string representation.  The recorded filename is
    relative to the checksum file whenever possible, which keeps result folders
    portable.
    """
    output_path = _prepare_target(output)
    rows: list[str] = []
    for item in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        resolved = item.resolve()
        try:
            display = resolved.relative_to(output_path.parent.resolve())
        except ValueError:
            display = resolved
        rows.append(f"{sha256_file(resolved)}  {display.as_posix()}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write("\n".join(rows))
            handle.write("\n" if rows else "")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        temporary = None
        _sync_directory(output_path.parent)
        return output_path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_sha256sums(
    checksums: Mapping[PathLike, str],
) -> dict[str, bool]:
    """Return a per-path result for a mapping of path to expected digest."""
    return {
        str(path): sha256_file(path) == expected.lower()
        for path, expected in checksums.items()
    }


__all__ = [
    "atomic_save_checkpoint",
    "atomic_torch_save",
    "atomic_write_json",
    "canonical_json_sha256",
    "load_json",
    "load_torch_checkpoint",
    "sha256_file",
    "verify_sha256sums",
    "write_sha256sums",
]
