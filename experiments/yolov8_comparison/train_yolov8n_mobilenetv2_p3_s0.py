"""Scheme 1: P3 gate + reparameterized block; default runs the combination from scratch."""

import argparse
from pathlib import Path

import ultralytics

from p3_structure import P3YOLO
from train_yolov8n_s0 import DATA, TRAIN_ARGS as BASELINE_TRAIN_ARGS
from train_yolov8n_mobilenetv2_w0625_s0 import VAL_ARGS as BASELINE_VAL_ARGS


ROOT = Path(__file__).resolve().parent


def configuration(variant):
    model = ROOT / f'yolov8n_mobilenetv2_w0625_realloc_p3_{variant}.yaml'
    name = f'yolov8n_mobilenetv2_w0625_realloc_p3_{variant}_640_s0_scratch'
    train_args = {**BASELINE_TRAIN_ARGS, 'name': name}
    val_args = {**BASELINE_VAL_ARGS, 'name': name + '_test'}
    return model, train_args, val_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('combo', 'gate', 'rep'), default='combo',
                        help='combo: both modules; gate/rep: the corresponding single-module ablation')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Synthetic checks only, no training')
    mode.add_argument('--resume', action='store_true', help='Resume this variant from last.pt')
    mode.add_argument('--test', action='store_true', help='Evaluate this variant best.pt on test')
    args = parser.parse_args()
    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('Use ultralytics 8.4.89, matching the existing experiments.')
    model, train_args, val_args = configuration(args.variant)
    run = Path(train_args['project']) / train_args['name']
    print(f'Variant: {args.variant}\nModel: {model}\nRun: {run}', flush=True)
    if args.check:
        from check_p3_structure import check
        check(model, train_args, val_args)
        return
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    if args.resume or args.test:
        checkpoint = run / ('weights/last.pt' if args.resume else 'weights/best.pt')
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        yolo = P3YOLO(str(checkpoint), task='detect')
        expected = {'gate': args.variant != 'rep', 'rep': args.variant != 'gate'}
        if yolo.model.yaml.get('p3_design') != expected:
            raise RuntimeError('Checkpoint P3 design does not match the selected variant.')
        if args.resume:
            yolo.train(resume=True)
        else:
            metrics = yolo.val(**val_args)
            print(f'Test mAP50: {metrics.box.map50:.6f}')
            print(f'Test mAP50-95: {metrics.box.map:.6f}')
            print(f'Results: {metrics.save_dir}')
    else:
        if run.exists():
            raise FileExistsError(f'{run} already exists; use --resume only for an interrupted run.')
        P3YOLO(str(model), task='detect').train(**train_args)


if __name__ == '__main__':
    main()
