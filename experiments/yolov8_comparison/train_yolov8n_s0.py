"""YOLOv8n SHWD comparison. Run in PyCharm or its PowerShell terminal."""

import argparse
import copy
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg


ROOT = Path(__file__).resolve().parent
DATA = Path('E:/codex_work/experiment of mobilenet2 and YOLO5/yolov5-paper/SHWD_YOLO/shwd.yaml')
PROJECT = Path('E:/experiment_M2Y5/analysis_runs')
NAME = 'yolov8n_640_s0_scratch'
RUN = PROJECT / NAME

# Same available training budget, initialization policy, SGD schedule and nominal
# augmentations as W0.625. Loss, assignment, augmentation implementation and
# checkpoint selection remain framework-specific: this is a model-system comparison.
TRAIN_ARGS = dict(
    data=str(DATA), imgsz=640, epochs=200, patience=50, batch=8,
    device=0, workers=4, seed=0, deterministic=True,
    pretrained=False, optimizer='SGD', lr0=0.01, lrf=0.1,
    momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
    warmup_momentum=0.8, warmup_bias_lr=0.1, nbs=64,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0,
    translate=0.1, scale=0.9, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1,
    copy_paste=0.0, cutmix=0.0, bgr=0.0, close_mosaic=0,
    rect=False, multi_scale=0.0, cos_lr=False, cache=False,
    amp=True, compile=False,
    # Native YOLOv8 detection loss gains, not YOLOv5's gains.
    box=7.5, cls=0.5, dfl=1.5, cls_pw=0.0,
    conf=0.001, iou=0.6, max_det=300, agnostic_nms=False,
    val=True, plots=True, save=True, save_period=-1,
    project=str(PROJECT), name=NAME, exist_ok=False,
)


def check():
    """Read-only dataset/config checks and one CPU inference, without training."""
    from ultralytics.nn.tasks import DetectionModel

    get_cfg(overrides=TRAIN_ARGS)
    data = yaml.safe_load(DATA.read_text(encoding='utf-8'))
    dataset_root = Path(data['path'])
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    for split, expected in [('train', 5457), ('val', 607), ('test', 1517)]:
        directory = dataset_root / data[split]
        count = sum(p.suffix.lower() in extensions for p in directory.rglob('*') if p.is_file())
        if count != expected:
            raise RuntimeError(f'{split}: expected {expected} images, found {count} in {directory}')
        print(f'{split}: {count} images', flush=True)
    assert data['names'] == ['hat', 'person'], data['names']
    torch.set_num_threads(4)
    yolo = YOLO('yolov8n.yaml')
    model = DetectionModel(copy.deepcopy(yolo.model.yaml), nc=2, verbose=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640))
    predictions = output[0] if isinstance(output, tuple) else output
    assert predictions.shape == (1, 6, 8400), predictions.shape
    assert torch.isfinite(predictions).all()
    model.fuse()
    model.info(imgsz=640)
    print('CHECK PASSED: no training started.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Validate only; do not train')
    mode.add_argument('--resume', action='store_true', help='Resume this run from weights/last.pt')
    args = parser.parse_args()
    print(f'Ultralytics {ultralytics.__version__}; PyTorch {torch.__version__}', flush=True)
    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('This experiment was prepared for ultralytics 8.4.89; review version changes before running.')
    if args.check:
        check()
        return
    if args.resume:
        checkpoint = RUN / 'weights/last.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        YOLO(str(checkpoint)).train(resume=True)
        return
    if RUN.exists():
        raise FileExistsError(f'{RUN} already exists. Use --resume for an interrupted run; do not overwrite it.')
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    YOLO('yolov8n.yaml').train(**TRAIN_ARGS)


if __name__ == '__main__':
    main()
