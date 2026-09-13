"""Architecture, training contract and batch-8 memory checks; no optimizer steps."""
import copy
import tempfile
from pathlib import Path
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Detect
from ultralytics.utils.torch_utils import get_flops
from p2_plain_structure import P2PlainYOLO,P2PlainTrainer,P2PlainDetect
from p2_task_structure import P2TaskYOLO
from train_yolov8n_s0 import TRAIN_ARGS


def check(path,train_args):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    get_cfg(overrides=train_args)
    assert {k for k in train_args if train_args[k]!=TRAIN_ARGS[k]}=={'name'}
    full_path=Path(path).with_name('yolov8n_mobilenetv2_w0625_p2_task_combo.yaml')
    a=yaml.safe_load(Path(path).read_text()); b=yaml.safe_load(full_path.read_text())
    assert a['backbone']==b['backbone'] and a['head']==b['head']
    yolo=P2PlainYOLO(str(path),task='detect')
    model=yolo.model.eval()
    head=model.model[-1]
    assert isinstance(head,P2PlainDetect) and not head.end2end
    assert head.stride.tolist()==[4,8,16,32] and head.reg_max==16
    full=P2TaskYOLO(str(full_path),task='detect').model.eval()
    for plain_layer,full_layer in zip(list(model.model)[:-1],list(full.model)[:-1],strict=True):
        assert type(plain_layer)==type(full_layer)
        plain_layer.load_state_dict(full_layer.state_dict(),strict=True)
    del full
    class LegacyDetect(Detect):
        legacy=True
    baseline_head=LegacyDetect(nc=2,ch=(64,96,192))
    for i in range(1,4):
        for attr in ('cv2','cv3'):
            left=getattr(head,attr)[i].state_dict(); right=getattr(baseline_head,attr)[i-1].state_dict()
            assert {k:v.shape for k,v in left.items()}=={k:v.shape for k,v in right.items()}
    del baseline_head
    features=[torch.rand(1,c,s,s) for c,s in zip((32,64,96,192),(16,8,4,2))]
    altered=[f.clone() for f in features]; altered[1].add_(1)
    with torch.no_grad():
        before=head.forward_head(features,**head.one2many)
        after=head.forward_head(altered,**head.one2many)
        for key in ('boxes','scores'):
            torch.testing.assert_close(before[key][:,:,:256],after[key][:,:,:256],rtol=0,atol=0)
            assert not torch.allclose(before[key][:,:,256:320],after[key][:,:,256:320])
    sample=torch.rand(1,3,640,640)
    with torch.no_grad():
        output,preds=model(sample)
        assert output.shape==(1,6,34000)
        assert [f.shape[1] for f in preds['feats']]==[32,64,96,192]
        assert model(torch.rand(1,3,320,512))[0].shape==(1,6,13600)
    trainer=object.__new__(P2PlainTrainer); trainer.data={'nc':2,'channels':3}
    rebuilt=trainer.get_model(cfg=copy.deepcopy(model.yaml),weights=model,verbose=False)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,rebuilt.state_dict()[key],rtol=0,atol=0)
    del rebuilt
    with tempfile.TemporaryDirectory() as directory:
        checkpoint=Path(directory)/'check.pt'; yolo.save(str(checkpoint))
        restored=P2PlainYOLO(str(checkpoint),task='detect').model.eval()
        assert isinstance(restored.model[-1],P2PlainDetect)
        with torch.no_grad():
            torch.testing.assert_close(restored(sample)[0],output,rtol=.01,atol=.03)
        trainer.get_model(cfg=copy.deepcopy(restored.yaml),weights=restored,verbose=False)
        del restored
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    training=copy.deepcopy(model).to(device).train(); training.args=get_cfg(overrides=train_args)
    bs,side,targets=(8,640,80) if device=='cuda' else (2,128,4)
    widths=torch.rand(bs*targets,2,device=device)*.08+.01
    widths[::4]=torch.rand_like(widths[::4])*.2+.15
    widths[1::8]=torch.rand_like(widths[1::8])*.2+.4
    centers=torch.rand_like(widths)*(1-widths)+widths/2
    batch=dict(img=torch.rand(bs,3,side,side,device=device),batch_idx=torch.arange(bs,device=device).repeat_interleave(targets),
               cls=torch.randint(0,2,(bs*targets,1),device=device).float(),bboxes=torch.cat((centers,widths),1))
    with torch.autocast(device_type=device,enabled=device=='cuda'):
        loss,_=training(batch)
    assert torch.isfinite(loss).all(); loss.sum().backward()
    for name,p in training.named_parameters():
        if p.requires_grad: assert p.grad is not None and torch.isfinite(p.grad).all(),name
    for tower in (training.model[-1].cv2,training.model[-1].cv3):
        assert tower[0][0].conv.weight.grad.abs().sum()>0
    if device=='cuda':
        print(f'CUDA AMP batch8/640,80 labels/image: allocated={torch.cuda.max_memory_allocated()/2**30:.2f}GiB reserved={torch.cuda.max_memory_reserved()/2**30:.2f}GiB')
    del training,batch,loss
    if device=='cuda': torch.cuda.empty_cache()
    unfused=sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            module.running_mean.uniform_(-.1,.1); module.running_var.uniform_(.8,1.2)
    with torch.no_grad():
        reference=model(sample)[0]; model.fuse(verbose=False)
        torch.testing.assert_close(model(sample)[0],reference,rtol=1e-3,atol=1e-3)
    print(f'Parameters unfused={unfused:,}, fused={sum(p.numel() for p in model.parameters()):,}; GFLOPs={get_flops(model,imgsz=640):.4f}')
    print('CHECK PASSED: identical combo neck, C-style tower shapes, single-scale head routing, four-grid inference, native loss/backward, trainer/checkpoint rebuild and fusion. No optimizer steps or training run.')
