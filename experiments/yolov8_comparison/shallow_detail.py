"""Extra stride-4 output, preserving the existing backbone weights and P3/P4/P5 outputs."""
from ultralytics.nn import tasks
from mobilenetv2_backbone import MobileNetV2Backbone, make_divisible, register_backbone


class MobileNetV2ShallowBackbone(MobileNetV2Backbone):
    def __init__(self, width_mult=0.625):
        super().__init__(width_mult)
        self.out_channels = (*self.out_channels, make_divisible(24 * width_mult))

    def forward(self, x):
        x = self.stem(x)
        features = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index == 1:
                shallow = x
            if index in (2, 4, 6):
                features.append(x)
        return [*features, shallow]


def register_shallow():
    register_backbone()
    tasks.MobileNetV2ShallowBackbone = MobileNetV2ShallowBackbone


register_shallow()
