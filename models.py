"""CIFAR ResNet models with a configurable number of blocks per stage.

Provides ``resnet20()`` and ``resnet56()`` factory functions.  The forward pass
can optionally return intermediate features after the third residual stage for
fingerprint extraction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard BasicBlock for CIFAR-scale ResNets.

    Architecture (He et al., 2016):
        Conv2d(in_planes, planes, 3, stride) -> BN -> ReLU
        Conv2d(planes, planes, 3, 1)       -> BN
        out = ReLU(out + shortcut(x))
    """

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    """ResNet variant tuned for 32x32 CIFAR images.

    - conv1: 3x3, stride 1, 16 filters, no max-pool.
    - Three residual stages with channel counts [16, 32, 64].
    - Adaptive average pooling + linear classifier.
    """

    def __init__(self, num_blocks: list[int], num_classes: int = 100) -> None:
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Build a sequential container of ``num_blocks`` BasicBlock instances."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[BasicBlock] = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 32, 32).
            return_features: If True, also return the feature maps after
                ``layer3`` (before pooling), shape (B, 64, 8, 8).

        Returns:
            Logits of shape (B, num_classes), or a tuple ``(logits, features)``
            when *return_features* is True.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        features = out  # (B, 64, 8, 8)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)

        if return_features:
            return logits, features
        return logits


def resnet20(num_classes: int = 100) -> CIFARResNet:
    """ResNet-20 (3 blocks per stage)."""
    return CIFARResNet([3, 3, 3], num_classes=num_classes)


def resnet56(num_classes: int = 100) -> CIFARResNet:
    """ResNet-56 (9 blocks per stage)."""
    return CIFARResNet([9, 9, 9], num_classes=num_classes)
