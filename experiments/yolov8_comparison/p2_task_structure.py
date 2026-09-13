"""High-resolution pyramid with task-specific adjacent-scale fusion at P2/P3.

Use P2TaskYOLO for YAML construction/training. No installed package modifications.
This is an experimental design, not an implementation of TOOD or DyHead.
"""
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK, LOGGER
from ultralytics.utils.torch_utils import initialize_weights
from shallow_detail import register_shallow


class TaskScaleFusion(nn.Module):
    """Separate projections before resizing; concatenate on the local grid.

Local and semantic describe feature sources, not guaranteed learned information.
Wider branches allocate capacity; they do not impose learned attention weights.
"""
    def __init__(self, local_channels, semantic_channels, local_width, semantic_width, output_width):
        super().__init__()
        self.local = Conv(local_channels, local_width, 1)
        self.semantic = Conv(semantic_channels, semantic_width, 1)
        self.mix = Conv(local_width + semantic_width, output_width, 1)

    def forward(self, pair):
        local, semantic = pair
        detail = self.local(local)
        context = F.interpolate(self.semantic(semantic), size=local.shape[-2:], mode='nearest')
        return self.mix(torch.cat((detail, context), dim=1))


def task_tower(local, semantic, local_width, semantic_width, hidden, outputs):
    return nn.Sequential(
        TaskScaleFusion(local, semantic, local_width, semantic_width, hidden),
        Conv(hidden, hidden, 3, g=hidden),
        Conv(hidden, hidden, 1),
        nn.Conv2d(hidden, outputs, 1),
    )


class P2TaskDetect(Detect):
    """Native DFL/decode/loss contract; custom feature towers only at P2/P3."""
    legacy = True

    def __init__(self, nc=2, ch=(32,64,96,192)):
        if tuple(ch) != (32,64,96,192):
            raise ValueError('This experiment requires P2/P3/P4/P5=32/64/96/192.')
        super().__init__(nc=nc, reg_max=16, end2end=False, ch=ch)
        self.cv2[0] = task_tower(32,64,24,8,32,4*self.reg_max)
        self.cv3[0] = task_tower(32,64,16,32,32,nc)
        self.cv2[1] = task_tower(64,96,48,16,64,4*self.reg_max)
        self.cv3[1] = task_tower(64,96,24,48,64,nc)
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
            source = (x[index],x[index+1]) if index < 2 else x[index]
            boxes.append(box_head[index](source).reshape(bs,4*self.reg_max,-1))
            scores.append(cls_head[index](source).reshape(bs,self.nc,-1))
        # Original feature grids determine anchor locations and native assignment.
        return dict(boxes=torch.cat(boxes,dim=-1),scores=torch.cat(scores,dim=-1),feats=x)


class P2TaskModel(DetectionModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        register_shallow()
        super().__init__(cfg,ch=ch,nc=nc,verbose=False)
        if self.yaml.get('p2_task_design') != 'adjacent_task_fusion_v1':
            raise ValueError('Missing/mismatched p2_task_design metadata.')
        original = self.model[-1]
        if len(self.model)!=30 or not isinstance(original,Detect) or original.f != [18,22,25,28]:
            raise ValueError('Expected the fixed four-level pyramid graph.')
        head = P2TaskDetect(nc=original.nc)
        head.stride = original.stride.clone()
        head.i,head.f = original.i,original.f
        head.type = 'p2_task_structure.P2TaskDetect'
        head.np = sum(p.numel() for p in head.parameters())
        initialize_weights(head)
        head.bias_init()
        self.model[-1] = head
        self.stride = head.stride
        if verbose:
            LOGGER.info('P2 task model: strides=4/8/16/32, channels=32/64/96/192; task-specific fusion at P2/P3.')
            self.info()


class P2TaskTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = P2TaskModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose and RANK==-1)
        if weights is not None:
            model.load(weights)
        return model


class P2TaskYOLO(YOLO):
    @property
    def task_map(self):
        mapping = super().task_map
        mapping['detect']['model'] = P2TaskModel
        mapping['detect']['trainer'] = P2TaskTrainer
        return mapping
