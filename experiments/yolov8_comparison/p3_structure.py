"""P3 gating and reparameterized feature extraction for the fixed W0.625 ablation.

Design inspired by RepVGG/RepViT and weighted cross-scale fusion; not an
implementation of those papers. Use P3YOLO for YAML construction and training.
"""

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2f, Concat, Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import fuse_conv_and_bn, initialize_weights

from mobilenetv2_backbone import register_backbone


class GatedP3Concat(nn.Module):
    """Modulate the semantic input, retaining the original [semantic, lateral] order."""

    def __init__(self, semantic_channels=96, lateral_channels=64, hidden=16):
        super().__init__()
        self.gate = nn.Sequential(
            Conv(semantic_channels + lateral_channels, hidden, 1),
            Conv(hidden, hidden, 3, g=hidden),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def gate_weight(logits):
        return 2 * logits.sigmoid()

    def forward(self, features):
        semantic, lateral = features
        joined = torch.cat((semantic, lateral), dim=1)
        weight = self.gate_weight(self.gate(joined))
        return torch.cat((semantic * weight, lateral), dim=1)


class BoundedGatedP3Concat(GatedP3Concat):
    """Same generator and gated branch; change only the transfer function to [0.5, 1.5]."""

    @staticmethod
    def gate_weight(logits):
        # At zero, both this function and 2*sigmoid have value 1 and derivative 0.5.
        return 1 + 0.5 * logits.tanh()


class RepDepthwise5(nn.Module):
    """Linear DW5/DW3/DW1 branches, exactly convertible to a single DW5 at eval."""

    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            ) for k in (5, 3, 1)
        )

    def forward(self, x):
        if hasattr(self, 'reparam'):
            return self.reparam(x)
        return self.branches[0](x) + self.branches[1](x) + self.branches[2](x)

    @torch.no_grad()
    def switch_to_deploy(self):
        if hasattr(self, 'reparam'):
            return
        if self.training:
            raise RuntimeError('Call eval() before reparameterizing Conv-BN branches.')
        convolutions = [fuse_conv_and_bn(branch[0], branch[1]) for branch in self.branches]
        fused = convolutions[0]
        for convolution in convolutions[1:]:
            pad = (5 - convolution.kernel_size[0]) // 2
            fused.weight.add_(F.pad(convolution.weight, (pad, pad, pad, pad)))
            fused.bias.add_(convolution.bias)
        self.reparam = fused
        del self.branches


class RepP3Bottleneck(nn.Module):
    """PW expansion -> linear parallel depthwise branches -> activation -> PW projection + skip."""

    def __init__(self, channels):
        super().__init__()
        self.expand = Conv(channels, channels * 2, 1)
        self.spatial = RepDepthwise5(channels * 2)
        self.act = nn.SiLU()
        self.project = Conv(channels * 2, channels, 1, act=False)

    def forward(self, x):
        return x + self.project(self.act(self.spatial(self.expand(x))))


class P3DetectionModel(DetectionModel):
    """Build the normal parser graph, then apply the two serialized P3 design switches.

No installed parser changes. YAML flags survive checkpointing, trainer rebuilds
and resume. Strides stay 8/16/32: both modifications preserve their interfaces.
"""

    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        register_backbone()
        super().__init__(cfg, ch=ch, nc=nc, verbose=False)
        design = self.yaml.get('p3_design')
        if not isinstance(design, dict) or set(design) not in ({'gate', 'rep'}, {'gate', 'rep', 'gate_mode'}):
            raise ValueError('p3_design requires gate/rep booleans and an optional gate_mode.')
        if any(type(design[key]) is not bool for key in ('gate', 'rep')):
            raise ValueError('p3_design switches must be YAML booleans.')
        gate_mode = design.get('gate_mode', 'standard')
        if gate_mode not in ('standard', 'bounded') or (gate_mode == 'bounded' and not design['gate']):
            raise ValueError('gate_mode must be standard or bounded; bounded requires gate=true.')
        if len(self.model) != 21 or not isinstance(self.model[12], Concat):
            raise ValueError('This experiment requires the fixed 21-layer realloc graph.')
        if self.model[12].f != [-1, 2]:
            raise ValueError('Expected P3 inputs in [upsampled semantic, lateral] order.')
        c2f = self.model[13]
        if not isinstance(c2f, C2f) or c2f.c != 32 or len(c2f.m) != 1 or c2f.cv1.conv.in_channels != 160:
            raise ValueError('Expected P3 C2f(160, 64) with one internal 32-channel block.')
        if design['gate']:
            original = self.model[12]
            replacement = BoundedGatedP3Concat() if gate_mode == 'bounded' else GatedP3Concat()
            initialize_weights(replacement)  # match native YOLO BN eps/momentum
            replacement.i, replacement.f = original.i, original.f
            replacement.type = f'p3_structure.{type(replacement).__name__}'
            replacement.np = sum(p.numel() for p in replacement.parameters())
            self.model[12] = replacement
        if design['rep']:
            replacement = RepP3Bottleneck(c2f.c)
            initialize_weights(replacement)
            c2f.m[0] = replacement
            c2f.np = sum(p.numel() for p in c2f.parameters())
        if verbose:
            LOGGER.info(f'P3 design: gate={design["gate"]}, rep={design["rep"]}, '
                        f'gate_mode={gate_mode}; channels=64/96/192')
            self.info()

    def fuse(self, verbose=True):
        # Validation/export call model.fuse(); convert custom branches before native Conv-BN fusion.
        self.eval()
        for module in list(self.modules()):
            if isinstance(module, RepDepthwise5):
                module.switch_to_deploy()
        return super().fuse(verbose=verbose)


class P3DetectionTrainer(DetectionTrainer):
    """Use the same custom constructor on trainer rebuild and interrupted-run resume."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P3DetectionModel(cfg, nc=self.data['nc'], ch=self.data['channels'],
                                 verbose=verbose and RANK == -1)
        if weights is not None:
            model.load(weights)
        return model


class P3YOLO(YOLO):
    @property
    def task_map(self):
        mapping = super().task_map
        mapping['detect']['model'] = P3DetectionModel
        mapping['detect']['trainer'] = P3DetectionTrainer
        return mapping
