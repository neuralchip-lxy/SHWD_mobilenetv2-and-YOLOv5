"""YOLOv8n + MobileNetV2 W0.625 backbone ablation; default action is training."""

import argparse
import copy
import tempfile
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

from mobilenetv2_backbone import register_backbone
from train_yolov8n_s0 import DATA, PROJECT, TRAIN_ARGS as BASELINE_TRAIN_ARGS


register_backbone()
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'yolov8n_mobilenetv2_w0625.yaml'
NAME = 'yolov8n_mobilenetv2_w0625_640_s0_scratch'
TRAIN_ARGS = {**BASELINE_TRAIN_ARGS, 'name': NAME}
VAL_ARGS = dict(
    data=TRAIN_ARGS['data'], split='test', imgsz=640, batch=4, device=0,
    workers=4, conf=0.001, iou=0.6, max_det=300,
    quantize=32, rect=True, augment=False, agnostic_nms=False,
    plots=True, save_json=True, project=str(PROJECT), name=NAME + '_test', exist_ok=False,
)


def check(model_path=MODEL, train_args=TRAIN_ARGS, val_args=VAL_ARGS, neck_channels=(64, 128, 256)):
    """Check shapes, native loss gradients, fusion and checkpoint reload; no optimizer steps."""
    torch.set_num_threads(4)
    torch.manual_seed(0)
    get_cfg(overrides=train_args)
    get_cfg(overrides=val_args)
    data = yaml.safe_load(DATA.read_text(encoding='utf-8'))
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    for split, expected in [('train', 5457), ('val', 607), ('test', 1517)]:
        directory = Path(data['path']) / data[split]
        count = sum(p.is_file() and p.suffix.lower() in extensions for p in directory.rglob('*'))
        assert count == expected, (split, count, expected)
        print(f'{split}: {count} images', flush=True)
    assert data['names'] == ['hat', 'person'], data['names']

    yolo = YOLO(str(model_path), task='detect')
    model = yolo.model.eval()
    baseline = DetectionModel(copy.deepcopy(YOLO('yolov8n.yaml').model.yaml), nc=2, verbose=False).eval()
    # Channel reallocation keeps module types and topology; original mode also keeps shapes.
    for new, old in zip(list(model.model)[7:], list(baseline.model)[9:], strict=True):
        assert type(new) is type(old), (type(new), type(old))
        assert new.state_dict().keys() == old.state_dict().keys()
        if neck_channels == (64, 128, 256):
            assert {k: v.shape for k, v in new.state_dict().items()} == {
                k: v.shape for k, v in old.state_dict().items()
            }
    original_cfg = yaml.safe_load(MODEL.read_text(encoding='utf-8'))
    selected_cfg = yaml.safe_load(Path(model_path).read_text(encoding='utf-8'))
    for new, old in zip(selected_cfg['backbone'] + selected_cfg['head'],
                        original_cfg['backbone'] + original_cfg['head'], strict=True):
        assert new[:3] == old[:3], (new, old)  # same connections, repeats and layer types
        if new[2] in ('Conv', 'C2f', 'SPPF'):
            expected_channel = dict(zip((64, 128, 256), neck_channels))[old[3][0]]
            assert new[3] == [expected_channel, *old[3][1:]], (new, old)
        else:
            assert new[3] == old[3], (new, old)  # includes backbone width and Index outputs
    changed = {key for key in train_args if train_args[key] != BASELINE_TRAIN_ARGS[key]}
    assert changed == {'name'}, changed
    sample = torch.rand(1, 3, 640, 640)
    head_shapes = []
    handle = model.model[-1].register_forward_pre_hook(
        lambda module, inputs: head_shapes.extend(tuple(x.shape) for x in inputs[0]))
    with torch.no_grad():
        features = model.model[0](sample)
        assert [tuple(x.shape) for x in features] == [(1, 24, 80, 80), (1, 64, 40, 40), (1, 200, 20, 20)]
        predictions = model(sample)[0]
        assert predictions.shape == (1, 6, 8400), predictions.shape
        assert torch.isfinite(predictions).all()
    handle.remove()
    assert head_shapes == [(1, c, s, s) for c, s in zip(neck_channels, (80, 40, 20))], head_shapes
    assert model.stride.tolist() == [8, 16, 32]

    # A separate copy exercises the real detection loss and every trainable parameter.
    training_model = copy.deepcopy(model).train()
    training_model.args = get_cfg(overrides=train_args)
    batch = dict(img=torch.rand(2, 3, 128, 128), batch_idx=torch.tensor([0, 1]),
                 cls=torch.tensor([[0.], [1.]]),
                 bboxes=torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.6, 0.2, 0.2]]))
    loss, _ = training_model(batch)
    assert torch.isfinite(loss).all(), loss
    loss.sum().backward()
    for name, parameter in training_model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name

    # Local custom classes must also survive the .pt save/load path used for validation.
    with tempfile.TemporaryDirectory(prefix='m2v8_check_') as directory:
        checkpoint = Path(directory) / 'check.pt'
        yolo.save(str(checkpoint))
        restored = YOLO(str(checkpoint), task='detect').model.eval()
        with torch.no_grad():
            restored_output = restored(sample)[0]
        # YOLO.save stores FP16 weights; permit the resulting round-off.
        torch.testing.assert_close(restored_output, predictions, rtol=0.01, atol=0.02)
        rebuilt = DetectionModel(copy.deepcopy(restored.yaml), nc=2, verbose=False)
        rebuilt.load(restored, verbose=False)

    unfused_count = sum(p.numel() for p in model.parameters())
    model.fuse(verbose=False)
    with torch.no_grad():
        torch.testing.assert_close(model(sample)[0], predictions, rtol=1e-4, atol=1e-3)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    baseline.fuse(verbose=False)
    for label, item in [('YOLOv8n baseline', baseline), (Path(model_path).stem, model)]:
        print(f'{label}: fused params={sum(p.numel() for p in item.parameters()):,}; '
              f'GFLOPs@640={get_flops(item, imgsz=640):.4f}', flush=True)
    print(f'Unfused custom params: {unfused_count:,}')
    print(f'Neck channels: {neck_channels}; run: {train_args["name"]}')
    print('CHECK PASSED: data, topology/channel contract, forward, loss/backward, checkpoint and fusion. '
          'No training or test-set evaluation started.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--realloc', action='store_true', help='Use P3/P4/P5 channels 64/96/192 in a separate run')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Run synthetic checks without training')
    mode.add_argument('--resume', action='store_true', help='Resume this run from weights/last.pt')
    mode.add_argument('--test', action='store_true', help='Evaluate this run best.pt on the test split')
    args = parser.parse_args()
    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('Use ultralytics 8.4.89, the same version as the YOLOv8n baseline.')
    model_path = ROOT / 'yolov8n_mobilenetv2_w0625_realloc.yaml' if args.realloc else MODEL
    run_name = 'yolov8n_mobilenetv2_w0625_realloc_640_s0_scratch' if args.realloc else NAME
    train_args = {**TRAIN_ARGS, 'name': run_name}
    val_args = {**VAL_ARGS, 'name': run_name + '_test'}
    run = Path(train_args['project']) / run_name
    print(f'Model: {model_path}\nRun: {run}', flush=True)
    if args.check:
        check(model_path, train_args, val_args, (64, 96, 192) if args.realloc else (64, 128, 256))
    elif args.resume:
        checkpoint = run / 'weights/last.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        YOLO(str(checkpoint), task='detect').train(resume=True)
    elif args.test:
        checkpoint = run / 'weights/best.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metrics = YOLO(str(checkpoint), task='detect').val(**val_args)
        print(f'Test mAP50: {metrics.box.map50:.6f}')
        print(f'Test mAP50-95: {metrics.box.map:.6f}')
        print(f'Results: {metrics.save_dir}')
    else:
        if run.exists():
            raise FileExistsError(f'{run} already exists; use --resume for an interrupted run.')
        if not DATA.is_file():
            raise FileNotFoundError(DATA)
        YOLO(str(model_path), task='detect').train(**train_args)


if __name__ == '__main__':
    main()
