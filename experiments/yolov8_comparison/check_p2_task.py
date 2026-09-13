"""Synthetic checks for the full design; no optimizer steps or dataset training."""
import copy
import tempfile
from pathlib import Path
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Detect
from ultralytics.utils.torch_utils import get_flops
from p2_task_structure import P2TaskYOLO, P2TaskTrainer, P2TaskDetect
from train_yolov8n_s0 import TRAIN_ARGS


def check(path, train_args):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    get_cfg(overrides=train_args)
    assert {k for k in train_args if train_args[k]!=TRAIN_ARGS[k]}=={'name'}
    yolo=P2TaskYOLO(str(path),task='detect')
    model=yolo.model.eval()
    head=model.model[-1]
    assert isinstance(head,P2TaskDetect) and not head.end2end
    assert head.stride.tolist()==[4,8,16,32]
    assert head.reg_max==16 and head.nc==2
    assert model.model[19].conv.groups==32
    # P4/P5 retain C's native head tower dimensions.
    class LegacyDetect(Detect):
        legacy=True
    native=LegacyDetect(nc=2,ch=(64,96,192))
    for target,source in [(2,1),(3,2)]:
        for attr in ('cv2','cv3'):
            a=getattr(head,attr)[target].state_dict()
            b=getattr(native,attr)[source].state_dict()
            assert {k:v.shape for k,v in a.items()}=={k:v.shape for k,v in b.items()}
    del native
    sample=torch.rand(1,3,640,640)
    with torch.no_grad():
        output,preds=model(sample)
        assert output.shape==(1,6,34000)
        assert [tuple(f.shape) for f in preds['feats']]==[(1,32,160,160),(1,64,80,80),(1,96,40,40),(1,192,20,20)]
        assert preds['boxes'].shape==(1,64,34000)
        assert model(torch.rand(1,3,320,512))[0].shape==(1,6,13600)
    # Head routing: P2 outputs depend on P2/P3, but not directly on P4/P5.
    features=[torch.rand(1,c,s,s) for c,s in zip((32,64,96,192),(16,8,4,2))]
    originals=[x.clone() for x in features]
    with torch.no_grad():
        first=head.forward_head(features,**head.one2many)
        changed=[x.clone() for x in features]; changed[1].add_(1)
        second=head.forward_head(changed,**head.one2many)
        far=[x.clone() for x in features]; far[2].add_(1)
        third=head.forward_head(far,**head.one2many)
    for key in ('boxes','scores'):
        assert not torch.allclose(first[key][:,:,:256],second[key][:,:,:256])
        torch.testing.assert_close(first[key][:,:,:256],third[key][:,:,:256],rtol=0,atol=0)
    for a,b in zip(features,originals):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    trainer=object.__new__(P2TaskTrainer)
    trainer.data={'nc':2,'channels':3}
    rebuilt=trainer.get_model(cfg=copy.deepcopy(model.yaml),weights=model,verbose=False)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,rebuilt.state_dict()[key],rtol=0,atol=0)
    del rebuilt
    with tempfile.TemporaryDirectory() as directory:
        ckpt=Path(directory)/'check.pt'
        yolo.save(str(ckpt))
        restored=P2TaskYOLO(str(ckpt),task='detect').model.eval()
        assert isinstance(restored.model[-1],P2TaskDetect)
        with torch.no_grad():
            torch.testing.assert_close(restored(sample)[0],output,rtol=.01,atol=.03)
        trainer.get_model(cfg=copy.deepcopy(restored.yaml),weights=restored,verbose=False)
        del restored
    # Actual proposed batch/image size, with a dense synthetic batch. No optimizer.
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    training=copy.deepcopy(model).to(device).train()
    training.args=get_cfg(overrides=train_args)
    batch_size,side,targets=(8,640,80) if device=='cuda' else (2,128,4)
    widths=torch.rand(batch_size*targets,2,device=device)*.08+.01
    widths[::4]=torch.rand_like(widths[::4])*.2+.15
    widths[1::8]=torch.rand_like(widths[1::8])*.2+.4
    centers=torch.rand_like(widths)*(1-widths)+widths/2
    batch=dict(img=torch.rand(batch_size,3,side,side,device=device),
               batch_idx=torch.arange(batch_size,device=device).repeat_interleave(targets),
               cls=torch.randint(0,2,(batch_size*targets,1),device=device).float(),
               bboxes=torch.cat((centers,widths),dim=1))
    with torch.autocast(device_type=device,enabled=device=='cuda'):
        loss,_=training(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for name,p in training.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
    for tower in (training.model[-1].cv2,training.model[-1].cv3):
        for i in (0,1):
            for branch in ('local','semantic'):
                assert getattr(tower[i][0],branch).conv.weight.grad.abs().sum()>0, (i,branch)
    if device=='cuda':
        print(f'CUDA AMP batch=8 imgsz=640, 80 labels/image: peak allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB')
    del training,batch,loss
    if device=='cuda':
        torch.cuda.empty_cache()
    unfused=sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            module.running_mean.uniform_(-.1,.1)
            module.running_var.uniform_(.8,1.2)
    with torch.no_grad():
        reference=model(sample)[0]
        model.fuse(verbose=False)
        torch.testing.assert_close(model(sample)[0],reference,rtol=1e-3,atol=1e-3)
        # Native tensor-only export path, not an ONNX or board-deployment claim.
        model.model[-1].export=True
        exported=model(sample)
        assert isinstance(exported,torch.Tensor) and exported.shape==(1,6,34000)
        torch.testing.assert_close(exported,reference,rtol=1e-3,atol=1e-3)
        model.model[-1].export=False
    print(f'Parameters: unfused={unfused:,}, fused={sum(p.numel() for p in model.parameters()):,}; GFLOPs@640={get_flops(model,imgsz=640):.4f}')
    print('CHECK PASSED: four scales, task routing, rectangular input, trainer rebuild, checkpoint, native loss/gradients, fusion and tensor export. No optimizer updates or training run started.')
