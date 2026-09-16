"""Repeat the two P2 head experiments with matched random seeds."""
import argparse
from pathlib import Path
import ultralytics
from train_yolov8n_s0 import TRAIN_ARGS, DATA
from train_yolov8n_mobilenetv2_p2_task_s0 import MODEL as COMBO_MODEL
from train_yolov8n_mobilenetv2_p2_local_light_s0 import MODEL as LOCAL_MODEL
from p2_task_structure import P2TaskYOLO
from p2_local_light_structure import P2LocalLightYOLO

VARIANTS = {
    'combo': (COMBO_MODEL, P2TaskYOLO, 'p2_task_combo', 'p2_task_design', 'adjacent_task_fusion_v1'),
    'local_light': (LOCAL_MODEL, P2LocalLightYOLO, 'p2_local_light', 'p2_local_light_design', 'local_only_task_towers_v1'),
}


def configuration(variant, seed):
    if variant not in VARIANTS or seed not in (0, 1, 2):
        raise ValueError('Expected combo/local_light and seed 0, 1 or 2.')
    model_path, model_class, suffix, design_key, design_value = VARIANTS[variant]
    name = f'yolov8n_mobilenetv2_w0625_{suffix}_640_s{seed}_scratch'
    settings = {**TRAIN_ARGS, 'seed': seed, 'name': name}
    return model_path, model_class, design_key, design_value, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--seed', required=True, type=int, choices=(0, 1, 2))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='Print settings without training or creating a run.')
    modes.add_argument('--resume', action='store_true', help='Resume the selected interrupted run.')
    args = parser.parse_args()
    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('Use ultralytics 8.4.89, matching seed 0.')
    path, cls, key, value, settings = configuration(args.variant, args.seed)
    run = Path(settings['project']) / settings['name']
    if not path.is_file():
        raise FileNotFoundError(path)
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    print(f'Variant: {args.variant}\nSeed: {args.seed}\nModel: {path}\nRun: {run}', flush=True)
    print('imgsz=640, epochs=200, patience=50, batch=8, workers=4, SGD, scratch, AMP', flush=True)
    if args.dry_run:
        print(f'DRY RUN: directory exists={run.exists()}; no training started.')
        return
    if args.resume:
        checkpoint = run / 'weights' / 'last.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = cls(str(checkpoint), task='detect')
        if model.model.yaml.get(key) != value:
            raise RuntimeError('Checkpoint architecture does not match the selected variant.')
        if model.model.args.get('seed') != args.seed:
            raise RuntimeError('Checkpoint seed does not match the selected seed.')
        model.train(resume=True)
    else:
        if run.exists():
            raise FileExistsError(f'{run} already exists. Use --resume only for interrupted training.')
        cls(str(path), task='detect').train(**settings)


if __name__ == '__main__':
    main()
