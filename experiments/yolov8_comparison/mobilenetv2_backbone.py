"""Standard MobileNetV2 feature stages, using Ultralytics Conv for BN fusion.

No ImageNet weights, final 1280-channel classification expansion or classifier.
The width multiplier belongs to this backbone alone, not the YOLO neck/head.
"""

from torch import nn
from ultralytics.nn import tasks
from ultralytics.nn.modules import Conv


def make_divisible(value, divisor=8):
    """MobileNet channel rounding, including the 10-percent safeguard."""
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class InvertedResidual(nn.Module):
    """Expansion, depthwise convolution and linear projection with optional skip."""

    def __init__(self, c1, c2, stride, expansion):
        super().__init__()
        hidden = round(c1 * expansion)
        layers = []
        if expansion != 1:
            layers.append(Conv(c1, hidden, 1, act=nn.ReLU6(inplace=True)))
        layers.extend([
            Conv(hidden, hidden, 3, stride, g=hidden, act=nn.ReLU6(inplace=True)),
            Conv(hidden, c2, 1, act=False),
        ])
        self.conv = nn.Sequential(*layers)
        self.use_res_connect = stride == 1 and c1 == c2

    def forward(self, x):
        output = self.conv(x)
        return x + output if self.use_res_connect else output


class MobileNetV2Backbone(nn.Module):
    """Return stride 8/16/32 features from standard MobileNetV2 stages."""

    def __init__(self, width_mult=0.625):
        super().__init__()
        if width_mult <= 0:
            raise ValueError('width_mult must be positive')
        self.width_mult = width_mult
        channels = make_divisible(32 * width_mult)
        self.stem = Conv(3, channels, 3, 2, act=nn.ReLU6(inplace=True))
        # expansion, unscaled output channels, repeats, first-block stride
        settings = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2),
                    (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
        stages = []
        for expansion, output, repeats, stride in settings:
            output = make_divisible(output * width_mult)
            blocks = []
            for i in range(repeats):
                blocks.append(InvertedResidual(channels, output, stride if i == 0 else 1, expansion))
                channels = output
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.out_channels = tuple(make_divisible(c * width_mult) for c in (32, 96, 320))

    def forward(self, x):
        x = self.stem(x)
        features = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in (2, 4, 6):
                features.append(x)
        return features


def register_backbone():
    """Expose the local layer to the YAML parser without editing site-packages."""
    tasks.MobileNetV2Backbone = MobileNetV2Backbone
