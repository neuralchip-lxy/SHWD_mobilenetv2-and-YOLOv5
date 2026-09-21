"""Evaluate the fixed YOLOv8n checkpoint on the existing SHWD test split."""

import argparse
from pathlib import Path

import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg


from experiment_paths import shwd_data, path_for
WEIGHTS = path_for('runs') / 'yolov8n_640_s0_scratch/weights/best.pt'
DATA = shwd_data()
VAL_ARGS = dict(
    data=str(DATA), split='test', imgsz=640, batch=4, device=0,
    workers=4, conf=0.001, iou=0.6, max_det=300,
    quantize=32, rect=True, augment=False, agnostic_nms=False,
    plots=True, save_json=True,
    project=str(path_for('runs')),
    name='yolov8n_640_s0_scratch_test', exist_ok=False,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check paths and options without inference')
    args = parser.parse_args()
    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('Use the same ultralytics 8.4.89 environment as training.')
    for path in (WEIGHTS, DATA):
        if not path.is_file():
            raise FileNotFoundError(path)
    get_cfg(overrides=VAL_ARGS)
    if args.check:
        print('CHECK PASSED: checkpoint, data YAML and validation options exist; no evaluation started.')
        return
    metrics = YOLO(str(WEIGHTS)).val(**VAL_ARGS)
    print(f'Test mAP50: {metrics.box.map50:.6f}')
    print(f'Test mAP50-95: {metrics.box.map:.6f}')
    print(f'Results: {metrics.save_dir}')


if __name__ == '__main__':
    main()
