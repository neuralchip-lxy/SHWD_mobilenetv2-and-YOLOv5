"""Full P2 + task-specific fusion experiment. Default: scratch training."""
import argparse
from pathlib import Path
import ultralytics
from p2_task_structure import P2TaskYOLO
from train_yolov8n_s0 import TRAIN_ARGS, DATA
from train_yolov8n_mobilenetv2_w0625_s0 import VAL_ARGS

MODEL = Path(__file__).with_name('yolov8n_mobilenetv2_w0625_p2_task_combo.yaml')
NAME = 'yolov8n_mobilenetv2_w0625_p2_task_combo_640_s0_scratch'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    for flag in ('check','resume','val','test'):
        mode.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if ultralytics.__version__!='8.4.89':
        raise RuntimeError('Use ultralytics 8.4.89, matching the baseline.')
    train_args={**TRAIN_ARGS,'name':NAME}
    run=Path(train_args['project'])/NAME
    print(f'Model: {MODEL}\nRun: {run}',flush=True)
    if args.check:
        from check_p2_task import check
        check(MODEL,train_args)
        return
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    if args.resume or args.val or args.test:
        checkpoint=run/'weights'/('last.pt' if args.resume else 'best.pt')
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model=P2TaskYOLO(str(checkpoint),task='detect')
        if model.model.yaml.get('p2_task_design')!='adjacent_task_fusion_v1':
            raise RuntimeError('Checkpoint design does not match this experiment.')
        if args.resume:
            model.train(resume=True)
        else:
            split='test' if args.test else 'val'
            metrics=model.val(**{**VAL_ARGS,'split':split,'name':NAME+'_'+split})
            print(f'{split}: mAP50={metrics.box.map50:.6f}, mAP50-95={metrics.box.map:.6f}')
            print(f'OUTPUT: {metrics.save_dir}')
    else:
        if run.exists():
            raise FileExistsError(f'{run} exists; --resume is for interrupted training.')
        P2TaskYOLO(str(MODEL),task='detect').train(**train_args)


if __name__=='__main__':
    main()
