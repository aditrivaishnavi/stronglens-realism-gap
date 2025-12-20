from __future__ import annotations
import torch.nn as nn
import torchvision.models as models
import torch


# =============================================================================
# ResNet-18 (standard, ~11.2M params)
# =============================================================================

def build_resnet18(in_ch: int = 3) -> nn.Module:
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(512, 1)
    return m


# =============================================================================
# Bottlenecked ResNet (Lanusse-style, ~0.2-1M params)
# Paper IV parity: reduces capacity to match Paper IV's 194K-param custom ResNet.
# Uses 1x1 channel reduction after each residual block.
# =============================================================================

class _BottleneckBlock(nn.Module):
    """Residual block with 1x1 bottleneck reduction (Lanusse et al. 2018 style)."""
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


class BottleneckedResNet(nn.Module):
    """
    Compact ResNet with 1x1 reductions, targeting ~200K-500K params.
    Inspired by Lanusse et al. 2018 shielded-ResNet architecture.
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 16):
        super().__init__()
        c = base_ch  # 16 by default -> ~250K params
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        # 4 stages with increasing channels, 1x1 bottlenecks
        self.layer1 = _BottleneckBlock(c, c, c * 2, stride=1)       # 16 -> 32
        self.layer2 = _BottleneckBlock(c * 2, c, c * 4, stride=2)   # 32 -> 64
        self.layer3 = _BottleneckBlock(c * 4, c * 2, c * 8, stride=2)  # 64 -> 128
        self.layer4 = _BottleneckBlock(c * 8, c * 2, c * 8, stride=2)  # 128 -> 128
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 8, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_bottlenecked_resnet(in_ch: int = 3, base_ch: int = 16) -> nn.Module:
    return BottleneckedResNet(in_ch=in_ch, base_ch=base_ch)


# =============================================================================
# EfficientNetV2-S (ImageNet pretrained, ~21.5M params)
# Paper IV parity: matches Paper IV's EfficientNetV2 architecture.
# =============================================================================

def build_efficientnet_v2_s(in_ch: int = 3, pretrained: bool = True) -> nn.Module:
    """Build EfficientNetV2-S with single-logit output for binary classification."""
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_v2_s(weights=weights)

    # Adapt first conv for non-3-channel inputs if needed.
    # NOTE (Q4.2): When in_ch != 3, pretrained weights for the first conv layer
    # are discarded and replaced with a freshly initialized Conv2d. This means
    # the model loses the benefit of pretrained low-level features for non-RGB inputs.
    if in_ch != 3:
        old_conv = m.features[0][0]
        m.features[0][0] = nn.Conv2d(
            in_ch, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

    # Replace classifier head: 1280 -> 1 logit
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(m.classifier[1].in_features, 1),
    )
    return m


# =============================================================================
# Model factory
# =============================================================================

def build_model(arch: str, in_ch: int = 3, **kwargs) -> nn.Module:
    """Build model by architecture name.

    Supported:
        resnet18           - Standard ResNet-18 (~11.2M params)
        bottlenecked_resnet - Lanusse-style compact ResNet (~250K params)
        efficientnet_v2_s  - EfficientNetV2-S pretrained (~21.5M params)
    """
    if arch == "resnet18":
        return build_resnet18(in_ch)
    elif arch == "bottlenecked_resnet":
        base_ch = kwargs.get("base_ch", 16)
        return build_bottlenecked_resnet(in_ch, base_ch=base_ch)
    elif arch == "efficientnet_v2_s":
        pretrained = kwargs.get("pretrained", True)
        return build_efficientnet_v2_s(in_ch, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {arch}. "
                         f"Supported: resnet18, bottlenecked_resnet, efficientnet_v2_s")
