"""Run the partial-SPD and spatial-fusion P2/P3 ablation on SHWD.

Default variant is the joint model.  Use --check before starting a run.
Training is from scratch with the existing 200-epoch YOLOv8 comparison policy.
"""

import argparse
from pathlib import Path

import ultralytics
from ultralytics.cfg import get_cfg

from p2_detail_spatial_structure import DESIGN, P2DetailSpatialYOLO, VARIANTS
from train_yolov8n_mobilenetv2_w0625_s0 import VAL_ARGS
from train_yolov8n_s0 import DATA, TRAIN_ARGS


HERE = Path(__file__).resolve().parent
MODEL_STEM = 'yolov8n_mobilenetv2_w0625_p2_task_detail_spatial'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, default='combo')
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    mode = parser.add_mutually_exclusive_group()
    for flag in ('check', 'dry-run', 'resume', 'val', 'test'):
        mode.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()

    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('Use the comparison environment with ultralytics 8.4.89.')
    model_path = HERE / f'{MODEL_STEM}_{args.variant}.yaml'
    name = f'{MODEL_STEM}_{args.variant}_640_s{args.seed}_scratch'
    settings = {**TRAIN_ARGS, 'name': name, 'seed': args.seed}
    run = Path(settings['project']) / name
    if not DATA.is_file() or not model_path.is_file():
        raise FileNotFoundError('Dataset YAML or model YAML is missing.')
    print(f'Model: {model_path}\nVariant: {args.variant}; seed={args.seed}\nRun: {run}\n'
          '640, batch8, epochs200, patience50, SGD, AMP, scratch', flush=True)

    if args.dry_run:
        get_cfg(overrides=settings)
        print('DRY RUN PASSED: no training started.', flush=True)
    elif args.check:
        from check_p2_detail_spatial import check
        check(model_path, settings)
    elif args.resume or args.val or args.test:
        checkpoint = run / 'weights' / ('last.pt' if args.resume else 'best.pt')
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = P2DetailSpatialYOLO(str(checkpoint), task='detect')
        if model.ckpt['train_args']['seed'] != args.seed:
            raise RuntimeError('Checkpoint seed mismatch.')
        metadata = model.model.yaml
        if (metadata.get('detail_spatial_design') != DESIGN
                or metadata.get('detail_spatial_variant') != args.variant):
            raise RuntimeError('Checkpoint structure does not match --variant.')
        if args.resume:
            model.train(resume=True)
        else:
            split = 'test' if args.test else 'val'
            metrics = model.val(**{**VAL_ARGS, 'split': split,
                                   'name': name + '_' + split})
            print(f'{split.upper()} mAP50={metrics.box.map50:.6f}, '
                  f'mAP50-95={metrics.box.map:.6f}\nOUTPUT: {metrics.save_dir}',
                  flush=True)
    else:
        if run.exists():
            raise FileExistsError(f'{run} exists. --resume is for interrupted runs only.')
        P2DetailSpatialYOLO(str(model_path), task='detect').train(**settings)


if __name__ == '__main__':
    main()
