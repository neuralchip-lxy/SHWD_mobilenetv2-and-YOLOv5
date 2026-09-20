"""Matched six-class SHEL5K experiments; no changes to existing SHWD runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from p2_task_structure import P2TaskYOLO, P2TaskModel, P2TaskTrainer
from train_yolov8n_s0 import TRAIN_ARGS
from train_yolov8n_mobilenetv2_w0625_s0 import VAL_ARGS

ROOT=Path(__file__).resolve().parent
DATA=Path('E:/experiment_M2Y5/datasets/SHEL5K6_YOLO_v1/shel5k.yaml')
MODEL=ROOT/'yolov8n_mobilenetv2_w0625_p2_task_shel5k6.yaml'
NAMES=['helmet','head_with_helmet','person_with_helmet','head','person_no_helmet','face']


def check(data_path):
    torch.set_num_threads(4)
    for variant in ('baseline','combo'):
        if variant=='baseline':
            model=DetectionModel(copy.deepcopy(YOLO('yolov8n.yaml').model.yaml),nc=6,verbose=False)
        else:
            model=P2TaskModel(str(MODEL),nc=6,verbose=False)
        model.eval()
        with torch.no_grad():
            pred=model(torch.rand(1,3,128,128))[0]
        assert pred.shape[1]==10 and torch.isfinite(pred).all()
        model.train(); model.args=get_cfg(overrides={**TRAIN_ARGS,'data':str(data_path)})
        batch=dict(img=torch.rand(2,3,128,128),batch_idx=torch.tensor([0,1]),
                   cls=torch.tensor([[4.],[5.]]),bboxes=torch.tensor([[.5,.5,.2,.2],[.4,.4,.3,.3]]))
        loss,_=model(batch)
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        for name,p in model.named_parameters():
            if p.requires_grad: assert p.grad is not None and torch.isfinite(p.grad).all(),name
        if variant=='combo':
            trainer=object.__new__(P2TaskTrainer); trainer.data=dict(nc=6,channels=3)
            rebuilt=trainer.get_model(cfg=copy.deepcopy(model.yaml),weights=model,verbose=False)
            assert rebuilt.model[-1].nc==6
        print(f'CHECK PASSED: {variant}, nc=6, 10-channel decoded output, six-class loss/gradients. CPU only.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('baseline','combo'),default='baseline')
    parser.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--project',type=Path,default=Path('E:/experiment_M2Y5/analysis_runs/shel5k6'))
    mode=parser.add_mutually_exclusive_group()
    for option in ('dry-run','check','resume','val','test'): mode.add_argument('--'+option,action='store_true')
    args=parser.parse_args()
    if ultralytics.__version__!='8.4.89': raise RuntimeError('Use ultralytics 8.4.89.')
    data=yaml.safe_load(args.data.read_text(encoding='utf-8'))
    if data['nc']!=6 or data['names']!=NAMES: raise ValueError('SHEL5K six-class mapping mismatch.')
    data_root=Path(data['path'])
    report=json.loads((data_root/'preparation_report.json').read_text(encoding='utf-8'))
    for split in ('train','val','test'):
        count=len(list((data_root/data[split]).glob('*.png')))
        if count!=report['split'][split]['images']: raise ValueError(f'{split} image count changed')
    name=f'shel5k6_{args.variant}_640_s{args.seed}_scratch'
    settings={**TRAIN_ARGS,'data':str(args.data.resolve()),'project':str(args.project),'name':name,'seed':args.seed}
    get_cfg(overrides=settings)
    run=args.project/name
    fingerprint=dict(data_yaml_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                     split_manifest_sha256=hashlib.sha256((data_root/'split_manifest.json').read_bytes()).hexdigest(),
                     variant=args.variant,seed=args.seed,names=NAMES)
    print(f'Data: {args.data}\nVariant: {args.variant}, seed={args.seed}\nRun: {run}\n640, batch8, epochs200, patience50, SGD, AMP, scratch',flush=True)
    if args.dry_run: print('DRY RUN: configuration and image counts checked. No training.'); return
    if args.check: check(args.data); return
    cls=YOLO if args.variant=='baseline' else P2TaskYOLO
    if args.resume or args.val or args.test:
        checkpoint=run/'weights'/('last.pt' if args.resume else 'best.pt')
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        saved=json.loads((run/'dataset_identity.json').read_text(encoding='utf-8'))
        if saved!=fingerprint: raise RuntimeError('Dataset or run identity changed.')
        model=cls(str(checkpoint),task='detect')
        if model.ckpt['train_args']['seed']!=args.seed or model.model.model[-1].nc!=6:
            raise RuntimeError('Wrong checkpoint seed or class count.')
        if args.variant=='combo' and model.model.yaml.get('p2_task_design')!='adjacent_task_fusion_v1':
            raise RuntimeError('Wrong combo checkpoint design.')
        if args.resume: model.train(resume=True)
        else:
            split='test' if args.test else 'val'
            metrics=model.val(**{**VAL_ARGS,'data':settings['data'],'split':split,'project':str(args.project),'name':name+'_'+split})
            print(f'{split}: mAP50={metrics.box.map50:.6f}, mAP50-95={metrics.box.map:.6f}\nOUTPUT: {metrics.save_dir}')
        return
    if run.exists(): raise FileExistsError(f'{run} exists. --resume is only for interrupted training.')
    model=cls('yolov8n.yaml' if args.variant=='baseline' else str(MODEL),task='detect')
    def record_identity(trainer):
        (Path(trainer.save_dir)/'dataset_identity.json').write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    model.add_callback('on_pretrain_routine_start',record_identity)
    model.train(**settings)


if __name__=='__main__': main()
