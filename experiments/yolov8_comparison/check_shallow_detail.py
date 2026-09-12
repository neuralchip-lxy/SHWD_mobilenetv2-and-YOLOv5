"""Synthetic contract checks; no training or optimizer updates."""
import copy
import tempfile
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import get_flops
from shallow_detail import MobileNetV2ShallowBackbone, register_shallow
from mobilenetv2_backbone import MobileNetV2Backbone
from train_yolov8n_s0 import TRAIN_ARGS


def check(path, train_args):
    register_shallow()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    get_cfg(overrides=train_args)
    assert {k for k in train_args if train_args[k] != TRAIN_ARGS[k]} == {'name'}
    a = MobileNetV2Backbone().eval()
    b = MobileNetV2ShallowBackbone().eval()
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        old, new = a(x), b(x)
        for left, right in zip(old, new[:3]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert new[3].shape == (1,16,160,160)
    yolo = YOLO(str(path), task='detect')
    model = yolo.model.eval()
    assert model.model[9].conv.groups == 16
    assert model.model[16].cv1.conv.in_channels == 176
    assert model.model[-1].f == [16,19,22]
    assert model.stride.tolist() == [8,16,32]
    with torch.no_grad():
        output = model(x)[0]
        assert output.shape == (1,6,8400) and torch.isfinite(output).all()
    # Trainer reconstruction is the path used by scratch training and resume.
    trainer = object.__new__(DetectionTrainer)
    trainer.data = {'nc':2, 'channels':3}
    rebuilt = trainer.get_model(cfg=copy.deepcopy(model.yaml), weights=model, verbose=False)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training = copy.deepcopy(model).to(device).train()
    training.args = get_cfg(overrides=train_args)
    batch = dict(img=torch.rand(2,3,128,128,device=device),
                 batch_idx=torch.tensor([0,1],device=device), cls=torch.tensor([[0.],[1.]],device=device),
                 bboxes=torch.tensor([[.5,.5,.3,.3],[.4,.6,.2,.2]],device=device))
    with torch.autocast(device_type=device, enabled=device=='cuda'):
        loss,_ = training(batch)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    for name,p in training.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert training.model[9].conv.weight.grad.abs().sum()>0
    assert training.model[10].conv.weight.grad.abs().sum()>0
    # Verify the exact extra channels reach P3; untrained final predictions can
    # be numerically indistinguishable because the initial head is near constant.
    captured=[]
    capture=model.model[16].register_forward_pre_hook(lambda m,i: captured.append(i[0].detach().clone()))
    with torch.no_grad():
        model(x)
    hook = model.model[10].register_forward_hook(lambda m,i,o: torch.zeros_like(o))
    with torch.no_grad():
        suppressed=model(x)[0]
    hook.remove()
    capture.remove()
    assert torch.equal(captured[0][:,:160],captured[1][:,:160])
    assert captured[0][:,160:].abs().sum()>0
    assert captured[1][:,160:].abs().sum()==0
    with tempfile.TemporaryDirectory() as directory:
        checkpoint=Path(directory)/'check.pt'
        yolo.save(str(checkpoint))
        restored=YOLO(str(checkpoint),task='detect').model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(x)[0],output,rtol=.01,atol=.03)
        trainer.get_model(cfg=copy.deepcopy(restored.yaml),weights=restored,verbose=False)
    before=sum(p.numel() for p in model.parameters())
    # Non-default BN statistics exercise the fusion rather than only identity BN.
    for m in model.modules():
        if isinstance(m,torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-.1,.1); m.running_var.uniform_(.8,1.2)
    with torch.no_grad():
        reference=model(x)[0]
        model.fuse(verbose=False)
        torch.testing.assert_close(model(x)[0],reference,rtol=1e-3,atol=1e-3)
    print(f'Unfused parameters: {before:,}; fused: {sum(p.numel() for p in model.parameters()):,}; GFLOPs: {get_flops(model,imgsz=640):.4f}')
    print(f'CHECK PASSED: backbone equivalence, shapes, bypass effect, {device} AMP/loss/gradients, trainer rebuild, checkpoint and fusion. No training started.')
