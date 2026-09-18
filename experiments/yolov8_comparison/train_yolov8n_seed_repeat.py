"""Original YOLOv8n baseline repeats. Only seed/name differ from seed 0."""
import argparse
from pathlib import Path
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from train_yolov8n_s0 import TRAIN_ARGS, DATA


def settings(seed):
    if seed not in (0, 1, 2):
        raise ValueError('Supported seeds: 0, 1, 2')
    return {**TRAIN_ARGS, 'seed': seed, 'name': f'yolov8n_640_s{seed}_scratch'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(0, 1, 2))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('Use ultralytics 8.4.89, matching seed 0.')
    cfg = settings(args.seed)
    get_cfg(overrides=cfg)
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    run = Path(cfg['project']) / cfg['name']
    print(f'Model: original yolov8n.yaml\nSeed: {args.seed}\nRun: {run}', flush=True)
    print('640, batch8, epochs200, patience50, workers4, SGD, AMP, scratch', flush=True)
    if args.dry_run:
        print(f'DRY RUN: directory exists={run.exists()}; no training started.')
        return
    if args.resume:
        checkpoint = run / 'weights' / 'last.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint), task='detect')
        if model.model.args.get('seed') != args.seed:
            raise RuntimeError('Checkpoint seed mismatch.')
        model.train(resume=True)
    else:
        if run.exists():
            raise FileExistsError(f'{run} exists. Use --resume only for interrupted training.')
        YOLO('yolov8n.yaml', task='detect').train(**cfg)


if __name__ == '__main__':
    main()
