"""Official ImageNet ResNet-34 with per-sample residual-branch gates.

The sixteen gates follow the BasicBlock traversal order.  Integer masks use
little-endian bit order: bit ``l`` (``1 << l``) controls ``GATE_NAMES[l]``.
Only the residual branch is gated; identity and projection shortcuts always
remain active.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34 as tv_resnet34
from torchvision.models.resnet import BasicBlock, ResNet


STAGE_COUNTS: tuple[int, ...] = (3, 4, 6, 3)
GATE_NAMES: tuple[str, ...] = tuple(
    f"layer{stage_index}.{block_index}"
    for stage_index, block_count in enumerate(STAGE_COUNTS, start=1)
    for block_index in range(block_count)
)

# An immutable, JSON-friendly sequence of (integer bit value, gate name) pairs.
LSB_GATE_MAPPING: tuple[tuple[int, str], ...] = tuple(
    (1 << gate_index, gate_name)
    for gate_index, gate_name in enumerate(GATE_NAMES)
)


class GatedBasicBlock(nn.Module):
    """A torchvision ``BasicBlock`` with a gate on ``F(x)`` only."""

    expansion: int = BasicBlock.expansion

    def __init__(self, source: BasicBlock) -> None:
        super().__init__()
        # Reuse every trained module verbatim.  Conversion therefore introduces
        # no parameter copies or changes to the official checkpoint.
        self.conv1 = source.conv1
        self.bn1 = source.bn1
        self.relu = source.relu
        self.conv2 = source.conv2
        self.bn2 = source.bn2
        self.downsample = source.downsample
        self.stride = source.stride
        # Keep the established ``mask`` attribute name used by the other
        # experiments, while this module's public helpers own all mutation.
        self.register_buffer("mask", torch.ones(1, dtype=torch.float32))

    def set_mask(self, mask: torch.Tensor) -> None:
        """Set a scalar or one scalar per sample for this block."""
        if mask.ndim > 1:
            raise ValueError(
                f"a block gate must be scalar or 1-D, got shape {tuple(mask.shape)}"
            )
        self.mask = mask.detach().to(device=self.mask.device, dtype=self.mask.dtype)

    def reset_mask(self) -> None:
        """Restore the all-open scalar gate."""
        self.mask = torch.ones(1, device=self.mask.device, dtype=self.mask.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        gate = self.mask
        if gate.numel() == 1:
            out = identity + out * gate.to(dtype=out.dtype)
        else:
            if gate.numel() != x.shape[0]:
                raise ValueError(
                    f"gate batch has {gate.numel()} samples, input has {x.shape[0]}"
                )
            broadcast_shape = (gate.numel(),) + (1,) * (out.ndim - 1)
            out = identity + out * gate.to(dtype=out.dtype).view(broadcast_shape)

        return self.relu(out)


def _convert_basic_blocks(model: ResNet) -> list[GatedBasicBlock]:
    """Convert a torchvision ResNet-34 in place, preserving traversal order."""
    gated_blocks: list[GatedBasicBlock] = []
    discovered_names: list[str] = []
    discovered_counts: list[int] = []

    for stage_index, expected_count in enumerate(STAGE_COUNTS, start=1):
        stage_name = f"layer{stage_index}"
        stage = getattr(model, stage_name)
        discovered_counts.append(len(stage))
        if len(stage) != expected_count:
            raise AssertionError(
                f"{stage_name} has {len(stage)} blocks; expected {expected_count}"
            )
        for block_index, block in enumerate(stage):
            gate_name = f"{stage_name}.{block_index}"
            if not isinstance(block, BasicBlock):
                raise TypeError(f"{gate_name} is {type(block).__name__}, not BasicBlock")
            gated_block = GatedBasicBlock(block)
            stage[block_index] = gated_block
            gated_blocks.append(gated_block)
            discovered_names.append(gate_name)

    if tuple(discovered_counts) != STAGE_COUNTS:
        raise AssertionError(
            f"stage counts {tuple(discovered_counts)} do not match {STAGE_COUNTS}"
        )
    if tuple(discovered_names) != GATE_NAMES:
        raise AssertionError("gate traversal order does not match fixed GATE_NAMES")
    if len(gated_blocks) != 16:
        raise AssertionError(f"expected 16 gated blocks, got {len(gated_blocks)}")
    return gated_blocks


def build_gated_resnet34() -> tuple[ResNet, list[GatedBasicBlock], tuple[str, ...]]:
    """Load the required official V1 checkpoint and install all sixteen gates.

    This public constructor intentionally has no random-initialization or
    alternative-weight option: Experiment 3 is pinned to
    ``ResNet34_Weights.IMAGENET1K_V1``.
    """
    model = tv_resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    gated_blocks = _convert_basic_blocks(model)
    model.eval()
    return model, gated_blocks, GATE_NAMES


def set_per_sample_masks(
    gated_blocks: Sequence[GatedBasicBlock], masks: torch.Tensor
) -> None:
    """Apply a ``(batch, 16)`` matrix in the fixed ``GATE_NAMES`` order."""
    if masks.ndim != 2:
        raise ValueError(f"masks must be 2-D, got shape {tuple(masks.shape)}")
    if len(gated_blocks) != len(GATE_NAMES):
        raise ValueError(
            f"expected {len(GATE_NAMES)} gated blocks, got {len(gated_blocks)}"
        )
    if masks.shape[1] != len(GATE_NAMES):
        raise ValueError(
            f"masks have {masks.shape[1]} gates; expected {len(GATE_NAMES)}"
        )
    for gate_index, block in enumerate(gated_blocks):
        block.set_mask(masks[:, gate_index])


def reset_masks(gated_blocks: Sequence[GatedBasicBlock]) -> None:
    """Restore every supplied block to its all-open state."""
    for block in gated_blocks:
        block.reset_mask()


@contextmanager
def temporary_per_sample_masks(
    gated_blocks: Sequence[GatedBasicBlock], masks: torch.Tensor
) -> Iterator[None]:
    """Set per-sample gates and always reset them, including on exceptions."""
    try:
        set_per_sample_masks(gated_blocks, masks)
        yield
    finally:
        reset_masks(gated_blocks)


def masks_from_integers(
    mask_ids: torch.Tensor, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Decode integer masks using the experiment's fixed LSB-first mapping.

    The result has shape ``mask_ids.shape + (16,)`` and FP32 values.  Thus row
    ``masks_from_integers(tensor([1 << l]))[0]`` has only column ``l`` open.
    """
    if mask_ids.is_floating_point() or mask_ids.is_complex():
        raise TypeError("mask_ids must have an integer dtype")
    target_device = mask_ids.device if device is None else torch.device(device)
    values = mask_ids.to(device=target_device, dtype=torch.int64)
    max_mask = 1 << len(GATE_NAMES)
    if bool(((values < 0) | (values >= max_mask)).any()):
        raise ValueError(f"mask_ids must be in [0, {max_mask - 1}]")
    bit_indices = torch.arange(len(GATE_NAMES), device=target_device)
    return values.unsqueeze(-1).bitwise_right_shift(bit_indices).bitwise_and(1).float()
