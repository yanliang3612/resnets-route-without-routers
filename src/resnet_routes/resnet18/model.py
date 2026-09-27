"""ResNet-18 with per-sample residual-branch gates.

Loads torchvision's pretrained ResNet-18 and replaces every ``BasicBlock``
with a ``GatedBasicBlock`` that exposes a per-sample binary mask
``m_l in {0, 1}^B`` on the residual branch:

    x_{l+1}(m) = x_l(m) + m_l * F_l(x_l(m)),     ReLU applied after the sum.

Shortcut (including projection / downsample) is unchanged when the gate
is open or closed, matching the experiment plan.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
from torchvision.models import resnet18 as tv_resnet18
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import BasicBlock


class GatedBasicBlock(nn.Module):
    """Drop-in replacement for ``torchvision.models.resnet.BasicBlock``.

    The forward path is identical to the upstream block except that the
    residual branch ``F_l(x)`` is multiplied by a per-sample scalar gate
    before being added to the shortcut.
    """

    def __init__(self, src: BasicBlock) -> None:
        super().__init__()
        self.conv1 = src.conv1
        self.bn1 = src.bn1
        self.relu = src.relu
        self.conv2 = src.conv2
        self.bn2 = src.bn2
        self.downsample = src.downsample
        self.stride = src.stride
        # mask broadcasts over (B,) -> (B,1,1,1).  Default: gate fully open.
        self.register_buffer("mask", torch.ones(1))

    def set_mask(self, mask: torch.Tensor) -> None:
        self.mask = mask.to(device=self.mask.device, dtype=torch.float32)

    def reset_mask(self) -> None:
        self.mask = torch.ones(1, device=self.mask.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        m = self.mask
        if m.numel() == 1:
            out = identity + m * out
        else:
            view = [-1] + [1] * (out.dim() - 1)
            out = identity + m.view(*view).to(out.dtype) * out
        return self.relu(out)


def _swap_basic_blocks(model: nn.Module) -> List[GatedBasicBlock]:
    gated: List[GatedBasicBlock] = []
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(model, stage_name)
        for i, blk in enumerate(stage):
            assert isinstance(blk, BasicBlock), f"{stage_name}.{i} is not BasicBlock"
            new_blk = GatedBasicBlock(blk)
            stage[i] = new_blk
            gated.append(new_blk)
    return gated


def build_gated_resnet18(pretrained: bool = True) -> "tuple[nn.Module, List[GatedBasicBlock]]":
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_resnet18(weights=weights)
    gated_blocks = _swap_basic_blocks(model)
    assert len(gated_blocks) == 8, f"expected 8 BasicBlocks, got {len(gated_blocks)}"
    model.eval()
    return model, gated_blocks


def set_per_sample_masks(
    gated_blocks: Sequence[GatedBasicBlock], masks: torch.Tensor
) -> None:
    """Apply a ``(B, L)`` mask matrix where row ``b`` is the mask for sample ``b``."""
    assert masks.dim() == 2 and masks.shape[1] == len(gated_blocks)
    for l, blk in enumerate(gated_blocks):
        blk.set_mask(masks[:, l])


def reset_masks(gated_blocks: Sequence[GatedBasicBlock]) -> None:
    for blk in gated_blocks:
        blk.reset_mask()
