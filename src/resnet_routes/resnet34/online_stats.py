"""Small, checkpointable online statistics for the ResNet-34 experiment.

The evaluator sees one image at a time.  Keeping every per-image K curve would
make the CPU-side result grow unnecessarily, so this module implements a
tensor-valued Welford accumulator.  All internal arithmetic is float64.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import torch


def _shape_tuple(shape: Sequence[int] | torch.Size | int | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(x) for x in shape)


@dataclass(frozen=True)
class FinalizedMoments:
    """Final estimates from an :class:`OnlineMoments` accumulator."""

    count: int
    mean: torch.Tensor
    std: torch.Tensor
    se: torch.Tensor
    ci_low: torch.Tensor
    ci_high: torch.Tensor

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


class OnlineMoments:
    """Welford mean/variance accumulator for equally shaped observations.

    ``update`` accepts one observation.  ``update_batch`` treats dimension 0
    as the sample dimension.  The observed shape may be supplied at
    construction or inferred from the first update.
    """

    def __init__(
        self,
        shape: Sequence[int] | torch.Size | int | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self._shape = _shape_tuple(shape)
        self.device = torch.device(device)
        self.count = 0
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None
        if self._shape is not None:
            self._initialize(self._shape)

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self._shape

    def _initialize(self, shape: tuple[int, ...]) -> None:
        self._shape = shape
        self.mean = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.m2 = torch.zeros_like(self.mean)

    def _coerce(self, value: torch.Tensor | float | int) -> torch.Tensor:
        x = torch.as_tensor(value, dtype=torch.float64, device=self.device)
        if self._shape is None:
            self._initialize(tuple(x.shape))
        if tuple(x.shape) != self._shape:
            raise ValueError(f"expected observation shape {self._shape}, got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("online statistics require finite observations")
        return x

    def update(self, value: torch.Tensor | float | int) -> None:
        """Add exactly one tensor-valued observation."""
        x = self._coerce(value)
        assert self.mean is not None and self.m2 is not None
        self.count += 1
        delta = x - self.mean
        self.mean.add_(delta / self.count)
        self.m2.add_(delta * (x - self.mean))

    def update_batch(self, values: torch.Tensor) -> None:
        """Add a batch whose leading dimension indexes observations."""
        x = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        if x.ndim == 0:
            raise ValueError("update_batch expects a leading sample dimension")
        expected = tuple(x.shape[1:])
        if self._shape is None:
            self._initialize(expected)
        if expected != self._shape:
            raise ValueError(f"expected batched shape (N, {self._shape}), got {tuple(x.shape)}")
        if x.shape[0] == 0:
            return
        if not torch.isfinite(x).all():
            raise ValueError("online statistics require finite observations")

        batch_count = int(x.shape[0])
        batch_mean = x.mean(dim=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)
        self._merge_values(batch_count, batch_mean, batch_m2)

    def _merge_values(
        self, other_count: int, other_mean: torch.Tensor, other_m2: torch.Tensor
    ) -> None:
        if other_count == 0:
            return
        assert self.mean is not None and self.m2 is not None
        if self.count == 0:
            self.count = other_count
            self.mean.copy_(other_mean)
            self.m2.copy_(other_m2)
            return
        total = self.count + other_count
        delta = other_mean - self.mean
        self.mean.add_(delta * (other_count / total))
        self.m2.add_(other_m2 + delta.square() * (self.count * other_count / total))
        self.count = total

    def merge(self, other: "OnlineMoments") -> None:
        """Merge an independent accumulator without replaying observations."""
        if other.shape is None or other.count == 0:
            return
        if self._shape is None:
            self._initialize(other.shape)
        if self._shape != other.shape:
            raise ValueError(f"cannot merge shapes {self._shape} and {other.shape}")
        assert other.mean is not None and other.m2 is not None
        self._merge_values(
            other.count,
            other.mean.to(device=self.device, dtype=torch.float64),
            other.m2.to(device=self.device, dtype=torch.float64),
        )

    def finalize(self, confidence: float = 0.95) -> FinalizedMoments:
        """Return mean, sample std, standard error and normal-approximation CI."""
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one")
        if self._shape is None:
            raise RuntimeError("cannot finalize an accumulator with unknown shape")
        assert self.mean is not None and self.m2 is not None
        if self.count == 0:
            nan = torch.full_like(self.mean, float("nan"))
            return FinalizedMoments(0, nan, nan.clone(), nan.clone(), nan.clone(), nan.clone())
        if self.count == 1:
            std = torch.full_like(self.mean, float("nan"))
            se = std.clone()
        else:
            std = torch.sqrt(torch.clamp(self.m2 / (self.count - 1), min=0.0))
            se = std / math.sqrt(self.count)
        z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        return FinalizedMoments(
            self.count,
            self.mean.clone(),
            std,
            se,
            self.mean - z * se,
            self.mean + z * se,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a ``torch.save``-compatible, CPU-resident checkpoint state."""
        return {
            "version": 1,
            "shape": self._shape,
            "count": self.count,
            "mean": None if self.mean is None else self.mean.detach().cpu().clone(),
            "m2": None if self.m2 is None else self.m2.detach().cpu().clone(),
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> "OnlineMoments":
        if int(state.get("version", 1)) != 1:
            raise ValueError(f"unsupported OnlineMoments state version {state.get('version')}")
        obj = cls(state.get("shape"), device=device)
        obj.count = int(state["count"])
        if obj._shape is None:
            if obj.count != 0:
                raise ValueError("non-empty state is missing its observation shape")
            return obj
        if state.get("mean") is None or state.get("m2") is None:
            raise ValueError("initialized state must contain mean and m2")
        assert obj.mean is not None and obj.m2 is not None
        mean = torch.as_tensor(state["mean"], dtype=torch.float64, device=obj.device)
        m2 = torch.as_tensor(state["m2"], dtype=torch.float64, device=obj.device)
        if tuple(mean.shape) != obj._shape or tuple(m2.shape) != obj._shape:
            raise ValueError("checkpoint tensor shape does not match declared shape")
        obj.mean.copy_(mean)
        obj.m2.copy_(m2)
        return obj


class OnlineStats:
    """Named online accumulators with optional sample-ID deduplication.

    The ID set is intentionally checkpointed.  On resume, replaying an already
    completed image raises instead of silently double-counting it.
    """

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.metrics: dict[str, OnlineMoments] = {}
        self.seen_sample_ids: set[str] = set()

    def update(
        self,
        values: Mapping[str, torch.Tensor | float | int],
        *,
        sample_id: str | None = None,
    ) -> None:
        if sample_id is not None and sample_id in self.seen_sample_ids:
            raise ValueError(f"sample {sample_id!r} has already been accumulated")
        if not values:
            raise ValueError("at least one metric is required")
        if self.metrics and set(values) != set(self.metrics):
            raise ValueError(
                f"metric names changed: expected {sorted(self.metrics)}, got {sorted(values)}"
            )
        # Validate the entire observation before changing any accumulator.  A
        # bad second metric must not leave the first metric one sample ahead.
        prepared: dict[str, torch.Tensor] = {}
        for name, value in values.items():
            tensor = torch.as_tensor(value, dtype=torch.float64, device=self.device)
            expected = self.metrics[name].shape if name in self.metrics else None
            if expected is not None and tuple(tensor.shape) != expected:
                raise ValueError(
                    f"metric {name!r} expected shape {expected}, got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"metric {name!r} contains a non-finite observation")
            prepared[name] = tensor
        for name, value in prepared.items():
            accumulator = self.metrics.setdefault(name, OnlineMoments(device=self.device))
            accumulator.update(value)
        if sample_id is not None:
            self.seen_sample_ids.add(sample_id)

    @property
    def count(self) -> int:
        if not self.metrics:
            return 0
        counts = {metric.count for metric in self.metrics.values()}
        if len(counts) != 1:
            raise RuntimeError("named metric accumulators have inconsistent counts")
        return next(iter(counts))

    def has_sample(self, sample_id: str) -> bool:
        return sample_id in self.seen_sample_ids

    def merge(self, other: "OnlineStats") -> None:
        overlap = self.seen_sample_ids.intersection(other.seen_sample_ids)
        if overlap:
            preview = sorted(overlap)[:3]
            raise ValueError(f"cannot merge duplicate sample IDs: {preview}")
        if self.metrics and other.metrics and set(self.metrics) != set(other.metrics):
            raise ValueError("cannot merge stores with different metric names")
        for name, accumulator in other.metrics.items():
            self.metrics.setdefault(name, OnlineMoments(device=self.device)).merge(accumulator)
        self.seen_sample_ids.update(other.seen_sample_ids)

    def finalize(self, confidence: float = 0.95) -> dict[str, FinalizedMoments]:
        return {name: value.finalize(confidence) for name, value in self.metrics.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "seen_sample_ids": sorted(self.seen_sample_ids),
            "metrics": {name: value.state_dict() for name, value in self.metrics.items()},
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> "OnlineStats":
        if int(state.get("version", 1)) != 1:
            raise ValueError(f"unsupported OnlineStats state version {state.get('version')}")
        obj = cls(device=device)
        obj.seen_sample_ids = {str(x) for x in state.get("seen_sample_ids", [])}
        obj.metrics = {
            str(name): OnlineMoments.from_state_dict(metric_state, device=device)
            for name, metric_state in state.get("metrics", {}).items()
        }
        if obj.seen_sample_ids and obj.count != len(obj.seen_sample_ids):
            raise ValueError("checkpoint sample-ID count does not match metric count")
        return obj


@dataclass(frozen=True)
class FinalizedScalarDistribution:
    """Exact scalar quantiles used for the K-effective summaries."""

    count: int
    median: float
    q25: float
    q75: float
    iqr: float


class OnlineScalarDistribution:
    """Checkpointable exact distribution for small scalar summaries.

    Full K curves use :class:`OnlineMoments`; only one K-effective scalar per
    image is retained here, so exact median/IQR cost O(N), not O(N*M).
    """

    def __init__(self) -> None:
        self.values: list[float] = []

    @property
    def count(self) -> int:
        return len(self.values)

    def update(self, value: torch.Tensor | float | int) -> None:
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if tensor.numel() != 1:
            raise ValueError("OnlineScalarDistribution accepts scalar observations only")
        scalar = float(tensor.item())
        if not math.isfinite(scalar):
            raise ValueError("scalar distribution requires finite observations")
        self.values.append(scalar)

    def update_batch(self, values: torch.Tensor | Sequence[float]) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        if not torch.isfinite(tensor).all():
            raise ValueError("scalar distribution requires finite observations")
        self.values.extend(float(x) for x in tensor.tolist())

    def merge(self, other: "OnlineScalarDistribution") -> None:
        self.values.extend(other.values)

    def finalize(self) -> FinalizedScalarDistribution:
        if not self.values:
            nan = float("nan")
            return FinalizedScalarDistribution(0, nan, nan, nan, nan)
        tensor = torch.tensor(self.values, dtype=torch.float64)
        q25, median, q75 = torch.quantile(
            tensor, torch.tensor([0.25, 0.50, 0.75], dtype=torch.float64)
        ).tolist()
        return FinalizedScalarDistribution(
            self.count, float(median), float(q25), float(q75), float(q75 - q25)
        )

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "values": list(self.values)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "OnlineScalarDistribution":
        if int(state.get("version", 1)) != 1:
            raise ValueError(
                f"unsupported OnlineScalarDistribution state version {state.get('version')}"
            )
        obj = cls()
        obj.update_batch(state.get("values", []))
        return obj
