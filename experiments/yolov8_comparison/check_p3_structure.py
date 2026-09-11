"""Synthetic checks for the nonstandard graph, training reconstruction and deploy conversion."""

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch import nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

from p3_structure import (BoundedGatedP3Concat, GatedP3Concat, P3DetectionModel, P3DetectionTrainer,
                          P3YOLO, RepDepthwise5, RepP3Bottleneck)
from train_yolov8n_s0 import TRAIN_ARGS as BASELINE_TRAIN_ARGS
from train_yolov8n_mobilenetv2_w0625_s0 import ROOT as BASE_ROOT


def unit_checks():
    gate = GatedP3Concat().eval()
    semantic, lateral = torch.randn(2, 96, 9, 11), torch.randn(2, 64, 9, 11)
    assert torch.equal(gate([semantic, lateral]), torch.cat((semantic, lateral), 1))
    with torch.no_grad():
        gate.gate[-1].weight.normal_(0, 0.1)
    gated = gate([semantic, lateral])
    assert torch.equal(gated[:, 96:], lateral), 'The lateral path must be unchanged.'
    assert not torch.equal(gated[:, :96], semantic), 'Gate must actually affect the semantic input.'

    bounded = BoundedGatedP3Concat().eval()
    assert torch.equal(bounded([semantic, lateral]), torch.cat((semantic, lateral), 1))
    assert {k: v.shape for k, v in bounded.state_dict().items()} == {
        k: v.shape for k, v in gate.state_dict().items()}
    logits = torch.linspace(-100, 100, 1001)
    weights = bounded.gate_weight(logits)
    assert weights.min() >= 0.5 and weights.max() <= 1.5
    for module in (gate, bounded):
        zero = torch.tensor(0., requires_grad=True)
        value = module.gate_weight(zero)
        value.backward()
        assert value.item() == 1 and zero.grad.item() == 0.5
    bounded.load_state_dict(gate.state_dict())
    bounded_out = bounded([semantic, lateral])
    assert torch.equal(bounded_out[:, 96:], lateral)
    assert not torch.equal(bounded_out[:, :96], semantic)

    block = RepDepthwise5(8).eval()
    with torch.no_grad():
        for branch in block.branches:
            branch[1].running_mean.normal_()
            branch[1].running_var.uniform_(0.5, 2)
            branch[1].weight.uniform_(0.5, 1.5)
            branch[1].bias.normal_()
        sample = torch.randn(2, 8, 11, 13)
        before = block(sample)
        block.switch_to_deploy()
        block.switch_to_deploy()  # repeated deployment conversion must be harmless
        torch.testing.assert_close(block(sample), before, rtol=1e-4, atol=1e-5)


def loss_backward(model, train_args, device='cpu', amp=False):
    model = copy.deepcopy(model).to(device).train()
    model.args = get_cfg(overrides=train_args)
    # Make the gate nonconstant so every part of its generator is exercised.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GatedP3Concat):
                module.gate[-1].weight.normal_(0, 0.02)
    batch = dict(img=torch.rand(2, 3, 128, 128, device=device),
                 batch_idx=torch.tensor([0, 1], device=device), cls=torch.tensor([[0.], [1.]], device=device),
                 bboxes=torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.6, 0.2, 0.2]], device=device))
    with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=amp):
        loss, _ = model(batch)
    assert torch.isfinite(loss).all(), loss
    loss.sum().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if name.startswith(('model.12.', 'model.13.m.')):
                assert parameter.grad.abs().sum() > 0, f'No effective gradient: {name}'
    return model.eval().cpu()


def check(model_path, train_args, val_args):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    unit_checks()
    get_cfg(overrides=train_args)
    get_cfg(overrides=val_args)
    assert {k for k in train_args if train_args[k] != BASELINE_TRAIN_ARGS[k]} == {'name'}
    data = yaml.safe_load(Path(train_args['data']).read_text(encoding='utf-8'))
    assert data['names'] == ['hat', 'person']
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    for split, expected in [('train', 5457), ('val', 607), ('test', 1517)]:
        count = sum(p.is_file() and p.suffix.lower() in extensions
                    for p in (Path(data['path']) / data[split]).rglob('*'))
        assert count == expected, (split, count)
        print(f'{split}: {count}', flush=True)

    yolo = P3YOLO(str(model_path), task='detect')
    model = yolo.model.eval()
    design = model.yaml['p3_design']
    original = yaml.safe_load((BASE_ROOT / 'yolov8n_mobilenetv2_w0625_realloc.yaml').read_text(encoding='utf-8'))
    selected = yaml.safe_load(Path(model_path).read_text(encoding='utf-8'))
    assert {k: v for k, v in selected.items() if k != 'p3_design'} == original
    reference = DetectionModel(copy.deepcopy(original), nc=2, verbose=False)
    for index, (new, old) in enumerate(zip(model.model, reference.model, strict=True)):
        if index not in (12, 13):
            assert type(new) is type(old)
            assert {k: v.shape for k, v in new.state_dict().items()} == {
                k: v.shape for k, v in old.state_dict().items()}
    assert isinstance(model.model[12], GatedP3Concat) == design['gate']
    assert isinstance(model.model[12], BoundedGatedP3Concat) == (design.get('gate_mode') == 'bounded')
    assert isinstance(model.model[13].m[0], RepP3Bottleneck) == design['rep']
    assert model.stride.tolist() == [8, 16, 32]
    head_shapes = []
    handle = model.model[-1].register_forward_pre_hook(
        lambda module, inputs: head_shapes.extend(tuple(x.shape) for x in inputs[0]))
    sample = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        predictions = model(sample)[0]
    handle.remove()
    assert predictions.shape == (1, 6, 8400) and torch.isfinite(predictions).all()
    assert head_shapes == [(1, 64, 80, 80), (1, 96, 40, 40), (1, 192, 20, 20)]
    trained = loss_backward(model, train_args)
    if torch.cuda.is_available():
        loss_backward(model, train_args, device='cuda', amp=True)
        print('CUDA AMP synthetic forward/backward passed (no optimizer step).', flush=True)

    # Exercise the exact trainer constructor used by YOLO.train(), without a trainer run.
    assert yolo.task_map['detect']['trainer'] is P3DetectionTrainer
    trainer = SimpleNamespace(data={'nc': 2, 'channels': 3})
    rebuilt = P3DetectionTrainer.get_model(trainer, copy.deepcopy(model.yaml), model, verbose=False).eval()
    assert isinstance(rebuilt, P3DetectionModel)
    assert rebuilt.yaml['p3_design'] == design
    assert set(rebuilt.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(value, rebuilt.state_dict()[key]), key

    with tempfile.TemporaryDirectory(prefix='p3_structure_check_') as directory:
        checkpoint = Path(directory) / 'check.pt'
        yolo.save(str(checkpoint))
        # Plain YOLO can load .pt because the serialized model retains its custom class.
        restored = YOLO(str(checkpoint), task='detect').model.eval()
        assert isinstance(restored, P3DetectionModel) and restored.yaml['p3_design'] == design
        with torch.no_grad():
            torch.testing.assert_close(restored(sample)[0], predictions, rtol=0.01, atol=0.02)
        resumed = P3DetectionTrainer.get_model(trainer, copy.deepcopy(restored.yaml), restored, verbose=False)
        assert resumed.yaml['p3_design'] == design
        assert set(resumed.state_dict()) == set(model.state_dict())

    # Use non-default BN statistics and a nonconstant gate for full-model equivalence.
    with torch.no_grad():
        before = trained(sample)[0]
    unfused_count = sum(p.numel() for p in trained.parameters())
    trained.fuse(verbose=False)
    trained.fuse(verbose=False)
    with torch.no_grad():
        torch.testing.assert_close(trained(sample)[0], before, rtol=1e-4, atol=1e-3)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in trained.modules())
    assert all(hasattr(m, 'reparam') and not hasattr(m, 'branches')
               for m in trained.modules() if isinstance(m, RepDepthwise5))
    print(f'Unfused parameters: {unfused_count:,}', flush=True)
    print(f'Deploy parameters: {sum(p.numel() for p in trained.parameters()):,}; '
          f'GFLOPs@640: {get_flops(trained, imgsz=640):.4f}', flush=True)
    print('CHECK PASSED: gates, branch fusion, graph, gradients, trainer rebuild, checkpoint and deploy. '
          'No training, optimizer updates or test-set evaluation started.', flush=True)
