"""CIFAR-scale neural architectures for homogeneous and cross-architecture distillation.

Provides:
- Homogeneous ResNet family: ``resnet20()``, ``resnet56()``, ``resnet110()``
- Wide ResNet-style CIFAR family: ``resnet8x4()``, ``resnet32x4()``
- Heterogeneous lightweight CNNs: ``mobilenetv2_cifar()``, ``shufflenetv2_cifar()``

Every model exposes:
- ``feature_dims = (l2_dim, l3_dim)``
- ``classifier_weight`` property returning the final linear classifier weight ``(num_classes, l3_dim)``
- ``forward(x, return_features=False | True | "multi")``
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard BasicBlock for CIFAR-scale ResNets (He et al., 2016)."""

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
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    """ResNet variant tuned for 32x32 CIFAR images.

    - conv1: 3x3, stride 1, base_width filters, no max-pool.
    - Three residual stages with channel counts [base, 2*base, 4*base].
    - Adaptive average pooling + linear classifier.
    """

    def __init__(
        self,
        num_blocks: list[int],
        num_classes: int = 100,
        widths: tuple[int, int, int] = (16, 32, 64),
    ) -> None:
        super().__init__()
        w1, w2, w3 = widths
        self.in_planes = w1
        self.feature_dims = (w2, w3)

        self.conv1 = nn.Conv2d(3, w1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w1)

        self.layer1 = self._make_layer(w1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(w2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(w3, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w3, num_classes)

    @property
    def classifier_weight(self) -> torch.Tensor:
        return self.fc.weight

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[BasicBlock] = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_features: bool | str = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        l2_out = self.layer2(out)    # (B, w2, 16, 16)
        l3_out = self.layer3(l2_out)  # (B, w3, 8, 8)
        out = self.avgpool(l3_out)
        out = out.flatten(1)
        logits = self.fc(out)

        if return_features:
            if isinstance(return_features, str) and return_features == "multi":
                return logits, l2_out, l3_out
            return logits, l3_out
        return logits


class _InvertedResidual(nn.Module):
    """Inverted residual block for CIFAR MobileNetV2."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        hidden = int(round(inp * expand_ratio))
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2CIFAR(nn.Module):
    """Lightweight MobileNetV2 adapted for 32x32 CIFAR-100 (heterogeneous architecture).

    Uses depthwise-separable inverted residuals and completely different feature
    channel dimensions (64 at stage-2, 320 at stage-3 vs. ResNet's 32 and 64).
    """

    def __init__(self, num_classes: int = 100, width_mult: float = 0.5) -> None:
        super().__init__()
        def _c(ch: int) -> int:
            return max(8, int(round(ch * width_mult / 8) * 8))

        c0 = _c(32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU6(inplace=True),
        )
        # Stage 1 (32x32)
        c1 = _c(24)
        self.stage1 = nn.Sequential(
            _InvertedResidual(c0, _c(16), stride=1, expand_ratio=1),
            _InvertedResidual(_c(16), c1, stride=1, expand_ratio=6),
            _InvertedResidual(c1, c1, stride=1, expand_ratio=6),
        )
        # Stage 2 (16x16)
        c2 = _c(64)
        self.stage2 = nn.Sequential(
            _InvertedResidual(c1, _c(32), stride=2, expand_ratio=6),
            _InvertedResidual(_c(32), _c(32), stride=1, expand_ratio=6),
            _InvertedResidual(_c(32), c2, stride=1, expand_ratio=6),
            _InvertedResidual(c2, c2, stride=1, expand_ratio=6),
        )
        # Stage 3 (8x8)
        c3 = _c(160)
        last_ch = _c(320)
        self.stage3 = nn.Sequential(
            _InvertedResidual(c2, _c(96), stride=2, expand_ratio=6),
            _InvertedResidual(_c(96), _c(96), stride=1, expand_ratio=6),
            _InvertedResidual(_c(96), c3, stride=1, expand_ratio=6),
            _InvertedResidual(c3, c3, stride=1, expand_ratio=6),
            nn.Conv2d(c3, last_ch, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU6(inplace=True),
        )
        self.feature_dims = (c2, last_ch)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_ch, num_classes)

    @property
    def classifier_weight(self) -> torch.Tensor:
        return self.fc.weight

    def forward(
        self, x: torch.Tensor, return_features: bool | str = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.stem(x)
        out = self.stage1(out)
        l2_out = self.stage2(out)
        l3_out = self.stage3(l2_out)
        pooled = self.avgpool(l3_out).flatten(1)
        logits = self.fc(pooled)
        if return_features:
            if isinstance(return_features, str) and return_features == "multi":
                return logits, l2_out, l3_out
            return logits, l3_out
        return logits


def resnet20(num_classes: int = 100) -> CIFARResNet:
    """ResNet-20 (3 blocks per stage, widths [16, 32, 64], ~278k params)."""
    return CIFARResNet([3, 3, 3], num_classes=num_classes)


def resnet56(num_classes: int = 100) -> CIFARResNet:
    """ResNet-56 (9 blocks per stage, widths [16, 32, 64], ~862k params)."""
    return CIFARResNet([9, 9, 9], num_classes=num_classes)


def resnet110(num_classes: int = 100) -> CIFARResNet:
    """ResNet-110 (18 blocks per stage, widths [16, 32, 64], ~1.73M params)."""
    return CIFARResNet([18, 18, 18], num_classes=num_classes)


def resnet8x4(num_classes: int = 100) -> CIFARResNet:
    """ResNet-8x4 (1 block per stage, widths [64, 128, 256], ~1.23M params)."""
    return CIFARResNet([1, 1, 1], num_classes=num_classes, widths=(64, 128, 256))


def resnet32x4(num_classes: int = 100) -> CIFARResNet:
    """ResNet-32x4 (5 blocks per stage, widths [64, 128, 256], ~7.43M params)."""
    return CIFARResNet([5, 5, 5], num_classes=num_classes, widths=(64, 128, 256))


class VGG8CIFAR(nn.Module):
    """Standard CIFAR VGG-8 with BatchNorm (classic CRD/DKD cross-architecture student).

    Completely heterogeneous from ResNet (no residual skip connections, MaxPool
    downsampling, and feature_dims=(256, 512) vs. ResNet-56's (32, 64)).
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()

        def _conv_bn_relu(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.block1 = nn.Sequential(_conv_bn_relu(3, 64), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(_conv_bn_relu(64, 128), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(_conv_bn_relu(128, 256), _conv_bn_relu(256, 256), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(_conv_bn_relu(256, 512), _conv_bn_relu(512, 512), nn.MaxPool2d(2, 2))
        self.block5 = nn.Sequential(_conv_bn_relu(512, 512))
        self.feature_dims = (256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @property
    def classifier_weight(self) -> torch.Tensor:
        return self.fc.weight

    def forward(
        self, x: torch.Tensor, return_features: bool | str = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.block1(x)
        out = self.block2(out)
        l2_out = self.block3(out)   # (B, 256, 4, 4)
        out = self.block4(l2_out)
        l3_out = self.block5(out)   # (B, 512, 2, 2)
        pooled = self.avgpool(l3_out).flatten(1)
        logits = self.fc(pooled)
        if return_features:
            if isinstance(return_features, str) and return_features == "multi":
                return logits, l2_out, l3_out
            return logits, l3_out
        return logits


def vgg8_cifar(num_classes: int = 100) -> VGG8CIFAR:
    """CIFAR VGG-8-BN (~3.96M params, feature_dims=(256, 512))."""
    return VGG8CIFAR(num_classes=num_classes)


def mobilenetv2_cifar(num_classes: int = 100, width_mult: float = 0.5) -> MobileNetV2CIFAR:
    """CIFAR MobileNetV2 (width_mult=0.5, ~239k params, feature_dims=(32, 160))."""
    return MobileNetV2CIFAR(num_classes=num_classes, width_mult=width_mult)


MODEL_REGISTRY = {
    "resnet20": resnet20,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet8x4": resnet8x4,
    "resnet32x4": resnet32x4,
    "mobilenetv2": mobilenetv2_cifar,
    "vgg8": vgg8_cifar,
}


def get_model(name: str, num_classes: int = 100) -> nn.Module:
    """Instantiate a CIFAR model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes)
