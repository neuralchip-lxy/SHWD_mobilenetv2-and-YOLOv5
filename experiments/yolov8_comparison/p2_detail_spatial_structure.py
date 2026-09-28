"""Partial detail-preserving downsample and spatial task fusion for P2/P3.

This research variant keeps the original four scales, head widths, native
YOLOv8 assignment/DFL/CIoU loss, and training policy.  Each component can be
enabled independently for a controlled ablation.
"""

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn import tasks
from ultralytics.nn.modules import Conv
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import initialize_weights

from mobilenetv2_backbone import make_divisible
from p2_task_structure import P2TaskModel, TaskScaleFusion
from shallow_detail import MobileNetV2ShallowBackbone


DESIGN = 'partial_spd_spatial_task_fusion_v1'
VARIANTS = ('detail', 'spatial', 'combo')


class PartialSPDDownsample(nn.Module):
    """Keep one quarter of expanded channels as 2x2 phase samples.

    The other channels retain the MobileNetV2 stride-2 depthwise operation.
    PixelUnshuffle itself rearranges values exactly; the subsequent projection
    is learned compression and is not claimed to be information-lossless.
    """

    def __init__(self, channels):
        super().__init__()
        if channels < 32 or channels % 8:
            raise ValueError(f'Expected expanded channels divisible by 8, got {channels}.')
        self.detail_channels = make_divisible(channels / 4)
        self.strided_channels = channels - self.detail_channels
        self.strided = Conv(self.strided_channels, self.strided_channels, 3, 2,
                            g=self.strided_channels, act=nn.ReLU6(inplace=True))
        self.detail = nn.Sequential(
            nn.PixelUnshuffle(2),
            Conv(4 * self.detail_channels, self.detail_channels, 1,
                 act=nn.ReLU6(inplace=True)),
        )

    def forward(self, x):
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError('PartialSPDDownsample requires even feature-map height and width.')
        strided, detail = torch.split(
            x, (self.strided_channels, self.detail_channels), dim=1
        )
        return torch.cat((self.strided(strided), self.detail(detail)), dim=1)


class MobileNetV2PartialDetailBackbone(MobileNetV2ShallowBackbone):
    """Change only the first stride-4-producing depthwise downsample."""

    def __init__(self, width_mult=0.625):
        super().__init__(width_mult)
        block = self.stages[1][0]
        old = block.conv[1]
        if (not isinstance(old, Conv) or old.conv.stride != (2, 2)
                or old.conv.groups != old.conv.in_channels
                or old.conv.in_channels != old.conv.out_channels):
            raise TypeError('Unexpected MobileNetV2 stride-4 block structure.')
        block.conv[1] = PartialSPDDownsample(old.conv.in_channels)


def register_detail_backbone():
    tasks.MobileNetV2PartialDetailBackbone = MobileNetV2PartialDetailBackbone


class SpatialTaskScaleFusion(TaskScaleFusion):
    """Learn separate per-location gains for local detail and adjacent context."""

    def __init__(self, local_channels, semantic_channels, local_width,
                 semantic_width, output_width):
        super().__init__(local_channels, semantic_channels, local_width,
                         semantic_width, output_width)
        channels = local_width + semantic_width
        self.spatial = nn.Sequential(
            Conv(channels, channels, 3, g=channels),
            nn.Conv2d(channels, 2, 1),
        )
        # 2 * hardsigmoid(0) = 1.  The initial forward exactly matches the
        # ungated fusion after loading identical local/semantic/mix weights.
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)

    @classmethod
    def from_existing(cls, original):
        return cls(original.local.conv.in_channels,
                   original.semantic.conv.in_channels,
                   original.local.conv.out_channels,
                   original.semantic.conv.out_channels,
                   original.mix.conv.out_channels)

    def forward(self, pair):
        local, semantic = pair
        detail = self.local(local)
        context = F.interpolate(self.semantic(semantic), size=local.shape[-2:],
                                mode='nearest')
        joined = torch.cat((detail, context), dim=1)
        gains = 2.0 * F.hardsigmoid(self.spatial(joined))
        weighted = torch.cat((detail * gains[:, 0:1], context * gains[:, 1:2]), dim=1)
        return self.mix(weighted)


class P2DetailSpatialModel(P2TaskModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        register_detail_backbone()
        super().__init__(cfg, ch=ch, nc=nc, verbose=False)
        if self.yaml.get('detail_spatial_design') != DESIGN:
            raise ValueError('Missing or mismatched detail_spatial_design metadata.')
        variant = self.yaml.get('detail_spatial_variant')
        if variant not in VARIANTS:
            raise ValueError(f'Unknown detail_spatial_variant: {variant!r}')
        has_detail = isinstance(self.model[0], MobileNetV2PartialDetailBackbone)
        if has_detail != (variant in ('detail', 'combo')):
            raise ValueError('Backbone and variant metadata disagree.')
        head = self.model[-1]
        if variant in ('spatial', 'combo'):
            for tower in (head.cv2, head.cv3):
                for level in (0, 1):
                    original = tower[level][0]
                    if not isinstance(original, TaskScaleFusion):
                        raise TypeError('Expected task fusion at P2/P3 tower input.')
                    tower[level][0] = SpatialTaskScaleFusion.from_existing(original)
            initialize_weights(head)
        head.type = f'P2TaskDetect+{DESIGN}+{variant}'
        head.np = sum(parameter.numel() for parameter in head.parameters())
        if verbose:
            self.info()


class P2DetailSpatialTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P2DetailSpatialModel(
            cfg, ch=self.data['channels'], nc=self.data['nc'],
            verbose=verbose and RANK == -1,
        )
        if weights is not None:
            model.load(weights)
        return model


class P2DetailSpatialYOLO(YOLO):
    @property
    def task_map(self):
        mapping = super().task_map
        mapping['detect']['model'] = P2DetailSpatialModel
        mapping['detect']['trainer'] = P2DetailSpatialTrainer
        return mapping
