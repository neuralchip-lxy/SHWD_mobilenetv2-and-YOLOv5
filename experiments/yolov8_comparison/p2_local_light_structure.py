"""Local-only lightweight P2/P3 control for the adjacent-scale task head.

Use P2LocalLightYOLO for YAML construction/training. No installed package modifications.
This is an experimental design, not an implementation of TOOD or DyHead.
"""
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK, LOGGER
from ultralytics.utils.torch_utils import initialize_weights
from shallow_detail import register_shallow


class LocalTaskFusion(nn.Module):
    """Two independent projections of the same local feature, then mixing.

    Keep both branch output widths and the mix/DW/PW tower of the full model.
    The second projection uses local input channels and the local spatial grid.
    This is an approximate-budget control, not exact FLOPs/parameter matching.
    """
    def __init__(self, local_channels, local_width, secondary_width, output_width):
        super().__init__()
        self.local = Conv(local_channels, local_width, 1)
        self.secondary = Conv(local_channels, secondary_width, 1)
        self.mix = Conv(local_width + secondary_width, output_width, 1)

    def forward(self, local):
        return self.mix(torch.cat((self.local(local), self.secondary(local)), dim=1))


def task_tower(local, local_width, secondary_width, hidden, outputs):
    return nn.Sequential(
        LocalTaskFusion(local, local_width, secondary_width, hidden),
        Conv(hidden, hidden, 3, g=hidden),
        Conv(hidden, hidden, 1),
        nn.Conv2d(hidden, outputs, 1),
    )


class P2LocalLightDetect(Detect):
    """Native DFL/decode/loss contract; custom feature towers only at P2/P3."""
    legacy = True

    def __init__(self, nc=2, ch=(32,64,96,192)):
        if tuple(ch) != (32,64,96,192):
            raise ValueError('This experiment requires P2/P3/P4/P5=32/64/96/192.')
        super().__init__(nc=nc, reg_max=16, end2end=False, ch=ch)
        self.cv2[0] = task_tower(32,24,8,32,4*self.reg_max)
        self.cv3[0] = task_tower(32,16,32,32,nc)
        self.cv2[1] = task_tower(64,48,16,64,4*self.reg_max)
        self.cv3[1] = task_tower(64,24,48,64,nc)
        # Preserve C's P4/P5 tower shapes. Native four-level construction would
        # shrink every classification tower because ch[0] changes from 64 to 32.
        self.cv3[2] = nn.Sequential(Conv(96,64,3),Conv(64,64,3),nn.Conv2d(64,nc,1))
        self.cv3[3] = nn.Sequential(Conv(192,64,3),Conv(64,64,3),nn.Conv2d(64,nc,1))

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        boxes, scores = [], []
        for index in range(self.nl):
            source = x[index]
            boxes.append(box_head[index](source).reshape(bs,4*self.reg_max,-1))
            scores.append(cls_head[index](source).reshape(bs,self.nc,-1))
        # Original feature grids determine anchor locations and native assignment.
        return dict(boxes=torch.cat(boxes,dim=-1),scores=torch.cat(scores,dim=-1),feats=x)


class P2LocalLightModel(DetectionModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        register_shallow()
        super().__init__(cfg,ch=ch,nc=nc,verbose=False)
        if self.yaml.get('p2_local_light_design') != 'local_only_task_towers_v1':
            raise ValueError('Missing/mismatched p2_local_light_design metadata.')
        original = self.model[-1]
        if len(self.model)!=30 or not isinstance(original,Detect) or original.f != [18,22,25,28]:
            raise ValueError('Expected the fixed four-level pyramid graph.')
        head = P2LocalLightDetect(nc=original.nc)
        head.stride = original.stride.clone()
        head.i,head.f = original.i,original.f
        head.type = 'p2_local_light_structure.P2LocalLightDetect'
        head.np = sum(p.numel() for p in head.parameters())
        initialize_weights(head)
        head.bias_init()
        self.model[-1] = head
        self.stride = head.stride
        if verbose:
            LOGGER.info('P2 task model: strides=4/8/16/32, channels=32/64/96/192; local-only lightweight task towers at P2/P3.')
            self.info()


class P2LocalLightTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P2LocalLightModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose and RANK==-1)
        if weights is not None:
            model.load(weights)
        return model


class P2LocalLightYOLO(YOLO):
    @property
    def task_map(self):
        mapping = super().task_map
        mapping['detect']['model'] = P2LocalLightModel
        mapping['detect']['trainer'] = P2LocalLightTrainer
        return mapping
