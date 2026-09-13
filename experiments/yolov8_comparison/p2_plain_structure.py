"""Full-combo pyramid with conventional C-style, single-scale detection towers."""
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK, LOGGER
from ultralytics.utils.torch_utils import initialize_weights
from shallow_detail import register_shallow


class P2PlainDetect(Detect):
    """C's 64-channel box/class towers, extended to P2. Native forward and loss."""
    legacy = True

    def __init__(self,nc=2,ch=(32,64,96,192)):
        if tuple(ch)!=(32,64,96,192):
            raise ValueError('Expected P2/P3/P4/P5 channels 32/64/96/192.')
        super().__init__(nc=nc,reg_max=16,end2end=False,ch=ch)
        # Native Detect derives the class width from ch[0], which would shrink
        # all towers to 32 just because P2 is added. Preserve C's width=64.
        self.cv3=nn.ModuleList(nn.Sequential(Conv(c,64,3),Conv(64,64,3),nn.Conv2d(64,nc,1)) for c in ch)


class P2PlainModel(DetectionModel):
    def __init__(self,cfg,ch=3,nc=None,verbose=True):
        register_shallow()
        super().__init__(cfg,ch=ch,nc=nc,verbose=False)
        if self.yaml.get('p2_plain_design')!='c_style_head_v1' or 'p2_task_design' in self.yaml:
            raise ValueError('Expected the plain P2 experiment metadata.')
        original=self.model[-1]
        if len(self.model)!=30 or not isinstance(original,Detect) or original.f!=[18,22,25,28]:
            raise ValueError('Expected the same four-level neck as the full combo.')
        head=P2PlainDetect(nc=original.nc)
        head.stride=original.stride.clone()
        head.i,head.f=original.i,original.f
        head.type='p2_plain_structure.P2PlainDetect'
        head.np=sum(p.numel() for p in head.parameters())
        initialize_weights(head)
        head.bias_init()
        self.model[-1]=head
        self.stride=head.stride
        if verbose:
            LOGGER.info('P2 plain: full-combo neck, conventional 64-channel towers, no cross-scale task fusion in head.')
            self.info()


class P2PlainTrainer(DetectionTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=P2PlainModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose and RANK==-1)
        if weights is not None:
            model.load(weights)
        return model


class P2PlainYOLO(YOLO):
    @property
    def task_map(self):
        mapping=super().task_map
        mapping['detect']['model']=P2PlainModel
        mapping['detect']['trainer']=P2PlainTrainer
        return mapping
