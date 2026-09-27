"""Exact mask enumeration for the gated ImageNet ResNet-34.

The evaluator deliberately supports only ``B=1``.  A single input is expanded
to each mask chunk, so row ``m`` in the returned tensor is always the logits for
integer mask ``m``.  Integer masks follow the LSB-first convention implemented
by :func:`resnet_routes.resnet34.model.masks_from_integers`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from .model import (
    GATE_NAMES,
    GatedBasicBlock,
    masks_from_integers,
    reset_masks,
    set_per_sample_masks,
)


ProgressCallback = Callable[[int, int], None]


def evaluate_all_masks(
    model: nn.Module,
    gated_blocks: Sequence[GatedBasicBlock],
    x: torch.Tensor,
    mask_chunk: int = 64,
    progress_callback: ProgressCallback | None = None,
) -> torch.Tensor:
    """Evaluate one image under all 65,536 residual-branch masks.

    Args:
        model: A model returned by ``build_gated_resnet34``.  The caller should
            put it in evaluation mode before calling this function.
        gated_blocks: The sixteen blocks returned alongside ``model``.
        x: One input image, with leading batch dimension exactly one.
        mask_chunk: Maximum number of masks evaluated in one model call.
        progress_callback: Optional ``callback(completed, total)`` invoked
            after every successfully evaluated chunk.

    Returns:
        A contiguous FP32 tensor of shape ``(1, 65536, C)`` on ``x.device``.
        ``C`` is inferred from the first model output rather than hard-coded.

    All gates are reset to the all-open state, even if model evaluation or the
    progress callback raises an exception.
    """
    if x.ndim < 1 or x.shape[0] != 1:
        shape = tuple(x.shape)
        raise ValueError(f"exact ResNet-34 mask evaluation requires B=1, got {shape}")
    if not isinstance(mask_chunk, int) or isinstance(mask_chunk, bool):
        raise TypeError("mask_chunk must be an integer")
    if mask_chunk <= 0:
        raise ValueError(f"mask_chunk must be positive, got {mask_chunk}")
    if len(gated_blocks) != len(GATE_NAMES):
        raise ValueError(
            f"expected {len(GATE_NAMES)} gated blocks, got {len(gated_blocks)}"
        )

    total_masks = 1 << len(GATE_NAMES)
    output_chunks: list[torch.Tensor] = []
    output_width: int | None = None

    try:
        with torch.inference_mode():
            for start in range(0, total_masks, mask_chunk):
                stop = min(start + mask_chunk, total_masks)
                mask_ids = torch.arange(
                    start, stop, dtype=torch.int64, device=x.device
                )
                masks = masks_from_integers(mask_ids, device=x.device)
                set_per_sample_masks(gated_blocks, masks)

                # ``expand`` avoids making a second full image batch.  Standard
                # convolution accepts this stride-zero batch view.
                x_chunk = x.expand(stop - start, *x.shape[1:])
                logits = model(x_chunk)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(
                        "gated ResNet-34 must return a Tensor of class logits"
                    )
                if logits.ndim != 2 or logits.shape[0] != stop - start:
                    raise ValueError(
                        "model output must have shape (mask_chunk, C), got "
                        f"{tuple(logits.shape)}"
                    )
                if output_width is None:
                    output_width = logits.shape[1]
                    if output_width <= 0:
                        raise ValueError("model output must contain at least one class")
                elif logits.shape[1] != output_width:
                    raise ValueError(
                        f"model output width changed from {output_width} "
                        f"to {logits.shape[1]}"
                    )

                output_chunks.append(logits.detach().to(dtype=torch.float32))
                if progress_callback is not None:
                    progress_callback(stop, total_masks)
    finally:
        reset_masks(gated_blocks)

    if output_width is None:  # Defensive: total_masks is nonzero by construction.
        raise RuntimeError("no masks were evaluated")
    return torch.cat(output_chunks, dim=0).unsqueeze(0).contiguous()
