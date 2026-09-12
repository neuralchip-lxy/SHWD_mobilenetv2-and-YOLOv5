"""C + shallow detail; standalone scratch training, validation, test and resume."""
import argparse
from pathlib import Path
import ultralytics
from ultralytics import YOLO
from shallow_detail import register_shallow
from train_yolov8n_s0 import TRAIN_ARGS
from train_yolov8n_mobilenetv2_w0625_s0 import VAL_ARGS

register_shallow()
MODEL = Path(__file__).with_name('yolov8n_mobilenetv2_w0625_realloc_p3_detail.yaml')
NAME = 'yolov8n_mobilenetv2_w0625_realloc_p3_detail_640_s0_scratch'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    for flag in ('check', 'resume', 'val', 'test'):
        group.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('Use ultralytics 8.4.89 to match the baseline.')
    train_args = {**TRAIN_ARGS, 'name': NAME}
    run = Path(train_args['project']) / NAME
    print(f'Model: {MODEL}\nRun: {run}', flush=True)
    if args.check:
        from check_shallow_detail import check
        check(MODEL, train_args)
    elif args.resume or args.val or args.test:
        checkpoint = run / 'weights' / ('last.pt' if args.resume else 'best.pt')
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        yolo = YOLO(str(checkpoint), task='detect')
        if yolo.model.yaml['backbone'][0][2] != 'MobileNetV2ShallowBackbone':
            raise RuntimeError('Checkpoint does not belong to the detail experiment.')
        if args.resume:
            yolo.train(resume=True)
        else:
            split = 'test' if args.test else 'val'
            metrics = yolo.val(**{**VAL_ARGS, 'split': split, 'name': NAME + '_' + split})
            print(f'{split} mAP50: {metrics.box.map50:.6f}; mAP50-95: {metrics.box.map:.6f}')
            print(f'OUTPUT: {metrics.save_dir}')
    else:
        if run.exists():
            raise FileExistsError(f'{run} exists. Use --resume for an interrupted run.')
        YOLO(str(MODEL), task='detect').train(**train_args)


if __name__ == '__main__':
    main()
