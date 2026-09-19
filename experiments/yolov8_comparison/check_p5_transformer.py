"""Synthetic architecture checks only; no optimizer updates or real dataset evaluation."""
import copy
import tempfile
from pathlib import Path
import torch
from thop import profile
from ultralytics.cfg import get_cfg
from p2_task_structure import P2TaskYOLO, P2TaskDetect
from p5_transformer_structure import P5TransformerYOLO, P5TransformerTrainer, AttentionProducts, count_attention_products
from train_yolov8n_s0 import TRAIN_ARGS


def check(path, settings):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    assert {k for k in settings if settings[k] != TRAIN_ARGS[k]} <= {'name', 'seed'}
    yolo = P5TransformerYOLO(str(path), task='detect')
    model = yolo.model.eval()
    baseline = P2TaskYOLO(str(Path(path).with_name('yolov8n_mobilenetv2_w0625_p2_task_combo.yaml')), task='detect').model.eval()
    assert len(model.model) == 30 and isinstance(model.model[-1], P2TaskDetect)
    assert model.stride.tolist() == [4, 8, 16, 32]
    for i in range(30):
        target = model.model[7][0] if i == 7 else model.model[i]
        assert {k: v.shape for k, v in target.state_dict().items()} == {k: v.shape for k, v in baseline.model[i].state_dict().items()}
    sample = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        reference = model(sample)[0]
        assert reference.shape == (1, 6, 34000)
        assert model(torch.rand(1, 3, 320, 512))[0].shape == (1, 6, 13600)
        assert torch.isfinite(reference).all()
    trainer = object.__new__(P5TransformerTrainer)
    trainer.data = {'nc': 2, 'channels': 3}
    rebuilt = trainer.get_model(cfg=copy.deepcopy(model.yaml), weights=model, verbose=False)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)
    del rebuilt
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'check.pt'
        yolo.save(str(checkpoint))
        restored = P5TransformerYOLO(str(checkpoint), task='detect').model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(sample)[0], reference, rtol=.01, atol=.03)
        del restored
    training = copy.deepcopy(model).train()
    training.args = get_cfg(overrides=settings)
    batch = dict(img=torch.rand(2,3,128,128), batch_idx=torch.tensor([0,1]),
                 cls=torch.tensor([[0.],[1.]]), bboxes=torch.tensor([[.5,.5,.2,.2],[.4,.4,.15,.15]]))
    loss, _ = training(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for name, parameter in training.named_parameters():
        assert not parameter.requires_grad or (parameter.grad is not None and torch.isfinite(parameter.grad).all()), name
    assert training.model[7][1].qkv.weight.grad.abs().sum() > 0
    del training, batch, loss
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        training = copy.deepcopy(model).cuda().train()
        training.args = get_cfg(overrides=settings)
        batch = dict(img=torch.rand(8,3,640,640,device='cuda'),
                     batch_idx=torch.arange(8,device='cuda').repeat_interleave(80),
                     cls=torch.randint(0,2,(640,1),device='cuda').float(),
                     bboxes=torch.cat((torch.rand(640,2,device='cuda')*.8+.1, torch.rand(640,2,device='cuda')*.08+.01),1))
        with torch.autocast('cuda'):
            loss, _ = training(batch)
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        for name, parameter in training.named_parameters():
            assert not parameter.requires_grad or (parameter.grad is not None and torch.isfinite(parameter.grad).all()), name
        print(f'AMP synthetic batch8/640: peak allocated={torch.cuda.max_memory_allocated()/2**30:.2f} GiB; reserved={torch.cuda.max_memory_reserved()/2**30:.2f} GiB')
        del training, batch, loss
        torch.cuda.empty_cache()
    with torch.no_grad():
        model.fuse(verbose=False)
        torch.testing.assert_close(model(sample)[0], reference, rtol=1e-3, atol=1e-3)
    baseline.fuse(verbose=False)
    for name, net in [('baseline_combo', baseline), ('p5_transformer', model)]:
        macs, _ = profile(copy.deepcopy(net), inputs=(sample,), custom_ops={AttentionProducts: count_attention_products}, verbose=False)
        print(f'{name}: fused parameters={sum(p.numel() for p in net.parameters())}; profiled GFLOPs@640={2*macs/1e9:.4f}')
    print('FLOPs include QK/AV matrix products; softmax/position/elementwise overhead not fully counted. Not a latency benchmark.')
    print('CHECK PASSED: unchanged original layer shapes, square/rectangular forward, trainer rebuild, checkpoint reload, CPU/available CUDA loss gradients and fusion. No training started.')
