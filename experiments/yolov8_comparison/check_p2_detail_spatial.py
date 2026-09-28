"""Synthetic architecture, gradient, checkpoint, and fusion checks; no training."""

import copy
import tempfile
from pathlib import Path

import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import get_flops, initialize_weights

from p2_detail_spatial_structure import (
    MobileNetV2PartialDetailBackbone, P2DetailSpatialTrainer,
    P2DetailSpatialYOLO, PartialSPDDownsample, SpatialTaskScaleFusion,
)
from p2_task_structure import TaskScaleFusion


def check(path, settings):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    get_cfg(overrides=settings)
    yolo = P2DetailSpatialYOLO(str(path), task='detect')
    model = yolo.model.eval()
    variant = model.yaml['detail_spatial_variant']
    detail_enabled = variant in ('detail', 'combo')
    spatial_enabled = variant in ('spatial', 'combo')
    assert model.stride.tolist() == [4, 8, 16, 32]
    assert len(model.model) == 30 and model.model[-1].reg_max == 16
    assert isinstance(model.model[0], MobileNetV2PartialDetailBackbone) == detail_enabled
    if detail_enabled:
        module = model.model[0].stages[1][0].conv[1]
        assert isinstance(module, PartialSPDDownsample)
        assert module.strided_channels + module.detail_channels == 96
        assert module(torch.rand(1, 96, 32, 48)).shape == (1, 96, 16, 24)

    head = model.model[-1]
    for tower in (head.cv2, head.cv3):
        for level in (0, 1):
            fusion = tower[level][0]
            assert isinstance(fusion, SpatialTaskScaleFusion) == spatial_enabled
            if spatial_enabled:
                # A zero-initialized gate starts exactly at the old fusion.
                base = TaskScaleFusion(
                    fusion.local.conv.in_channels, fusion.semantic.conv.in_channels,
                    fusion.local.conv.out_channels, fusion.semantic.conv.out_channels,
                    fusion.mix.conv.out_channels,
                ).eval()
                initialize_weights(base)
                for name in ('local', 'semantic', 'mix'):
                    getattr(base, name).load_state_dict(getattr(fusion, name).state_dict())
                pair = (torch.rand(1, fusion.local.conv.in_channels, 16, 16),
                        torch.rand(1, fusion.semantic.conv.in_channels, 8, 8))
                with torch.no_grad():
                    torch.testing.assert_close(fusion(pair), base(pair), rtol=0, atol=0)

    sample = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        reference = model(sample)[0]
        assert reference.shape == (1, 6, 34000) and torch.isfinite(reference).all()
        assert model(torch.rand(1, 3, 320, 512))[0].shape == (1, 6, 13600)

    trainer = object.__new__(P2DetailSpatialTrainer)
    trainer.data = {'nc': 2, 'channels': 3}
    rebuilt = trainer.get_model(cfg=copy.deepcopy(model.yaml), weights=model, verbose=False)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)
    del rebuilt
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'check.pt'
        yolo.save(str(checkpoint))
        restored = P2DetailSpatialYOLO(str(checkpoint), task='detect').model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(sample)[0], reference, rtol=1e-3, atol=1e-3)
        del restored

    # A small synthetic batch verifies that both new paths receive gradients.
    training = copy.deepcopy(model).train()
    training.args = get_cfg(overrides=settings)
    batch = dict(img=torch.rand(2, 3, 128, 128),
                 batch_idx=torch.tensor([0, 0, 1, 1]),
                 cls=torch.tensor([[0.], [1.], [0.], [1.]]),
                 bboxes=torch.tensor([[.30, .35, .13, .18], [.70, .65, .10, .14],
                                      [.40, .60, .12, .15], [.70, .30, .17, .20]]))
    loss, _ = training(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    if detail_enabled:
        module = training.model[0].stages[1][0].conv[1]
        assert module.detail[1].conv.weight.grad.abs().sum() > 0
        assert module.strided.conv.weight.grad.abs().sum() > 0
    if spatial_enabled:
        # The final 1x1 gets gradients immediately; preceding gate layers
        # start learning after the zero-initialized final projection updates.
        gate = training.model[-1].cv3[0][0].spatial[-1]
        assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0
    del training, batch, loss

    with torch.no_grad():
        model.fuse(verbose=False)
        torch.testing.assert_close(model(sample)[0], reference, rtol=1e-3, atol=1e-3)
        model.model[-1].export = True
        exported = model(sample)
        assert isinstance(exported, torch.Tensor) and exported.shape == (1, 6, 34000)
        model.model[-1].export = False
    print(f'{variant}: fused parameters={sum(p.numel() for p in model.parameters())}; '
          f'GFLOPs@640={get_flops(model, imgsz=640):.4f}')
    print('CHECK PASSED: 640 and rectangular inputs, unchanged four-level output, '
          'identity-initialized spatial gate, native loss gradients, checkpoint '
          'reload, fusion, and tensor export. No training started.')
