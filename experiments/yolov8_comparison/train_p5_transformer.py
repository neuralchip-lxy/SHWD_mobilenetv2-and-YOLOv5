"""P5 Transformer experiment: default scratch training, independent of original runs."""
import argparse
from pathlib import Path
import ultralytics
from p5_transformer_structure import P5TransformerYOLO
from train_yolov8n_s0 import TRAIN_ARGS, DATA
from train_yolov8n_mobilenetv2_w0625_s0 import VAL_ARGS

MODEL = Path(__file__).with_name('yolov8n_mobilenetv2_w0625_p2_task_p5_transformer.yaml')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--val', action='store_true')
    args = parser.parse_args()
    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('Use ultralytics 8.4.89.')
    name = f'yolov8n_mobilenetv2_w0625_p2_task_p5_transformer_640_s{args.seed}_scratch'
    settings = {**TRAIN_ARGS, 'name': name, 'seed': args.seed}
    run = Path(settings['project']) / name
    if not DATA.is_file() or not MODEL.is_file():
        raise FileNotFoundError('Dataset YAML or model YAML missing.')
    print(f'Model: {MODEL}\nRun: {run}\n640, batch8, epochs200, patience50, SGD, scratch, seed={args.seed}', flush=True)
    print('NOTE: default model GFLOPs may omit attention matrix products; use --check profiling report.', flush=True)
    if args.dry_run:
        from ultralytics.cfg import get_cfg
        get_cfg(overrides=settings)
        print('DRY RUN: no training started.')
    elif args.check:
        from check_p5_transformer import check
        check(MODEL, settings)
    elif args.resume or args.val:
        checkpoint = run / 'weights' / ('last.pt' if args.resume else 'best.pt')
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = P5TransformerYOLO(str(checkpoint), task='detect')
        if model.ckpt['train_args']['seed'] != args.seed:
            raise RuntimeError('Checkpoint seed mismatch.')
        if model.model.yaml.get('p5_transformer_design') != 'post_sppf_d96_h4_ffn192_v1':
            raise RuntimeError('Checkpoint design mismatch.')
        if args.resume:
            model.train(resume=True)
        else:
            metrics = model.val(**{**VAL_ARGS, 'split': 'val', 'name': name + '_val'})
            print(f'VAL mAP50={metrics.box.map50:.6f}, mAP50-95={metrics.box.map:.6f}')
    else:
        if run.exists():
            raise FileExistsError(f'{run} exists. Use --resume only for interrupted training.')
        P5TransformerYOLO(str(MODEL), task='detect').train(**settings)


if __name__ == '__main__':
    main()
