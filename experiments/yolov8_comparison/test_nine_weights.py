"""Evaluate nine frozen SHWD checkpoints. Never trains a model."""
import argparse
import csv
import gc
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from experiment_paths import resolve_path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_checked(cls, checkpoint, seed, key, value):
    model = cls(str(checkpoint), task='detect')
    # YOLO._load filters model.args; original training metadata remains in ckpt.
    actual_seed = (model.ckpt.get('train_args') or {}).get('seed')
    if actual_seed != seed:
        raise RuntimeError(f'Wrong checkpoint seed: expected={seed}, saved={actual_seed}: {checkpoint}')
    if key and model.model.yaml.get(key) != value:
        raise RuntimeError(f'Wrong architecture: {checkpoint}')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, help='Reuse a previous output to continue interrupted evaluation.')
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from p2_task_structure import P2TaskYOLO
    from p2_local_light_structure import P2LocalLightYOLO
    from train_yolov8n_s0 import PROJECT, DATA
    from train_yolov8n_mobilenetv2_w0625_s0 import VAL_ARGS
    if ultralytics.__version__ != '8.4.89':
        raise RuntimeError('Use ultralytics 8.4.89, matching training.')
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    settings = {k: v for k, v in VAL_ARGS.items() if k not in ('project', 'name', 'exist_ok')}
    settings.update(split='test', seed=0, deterministic=True)
    get_cfg(overrides=settings)
    variants = [
        ('baseline', YOLO, 'yolov8n_640_s{seed}_scratch', None, None),
        ('local_light', P2LocalLightYOLO, 'yolov8n_mobilenetv2_w0625_p2_local_light_640_s{seed}_scratch',
         'p2_local_light_design', 'local_only_task_towers_v1'),
        ('combo', P2TaskYOLO, 'yolov8n_mobilenetv2_w0625_p2_task_combo_640_s{seed}_scratch',
         'p2_task_design', 'adjacent_task_fusion_v1'),
    ]
    jobs = []
    for variant, cls, template, key, value in variants:
        for seed in (0, 1, 2):
            checkpoint = Path(PROJECT) / template.format(seed=seed) / 'weights' / 'best.pt'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            jobs.append((variant, seed, cls, checkpoint, key, value))
            print(f'FOUND {variant} seed={seed}: {checkpoint}', flush=True)
    print('Shared evaluation settings:', json.dumps(settings, ensure_ascii=False), flush=True)
    if args.dry_run:
        for variant, seed, cls, checkpoint, key, value in jobs:
            model = load_checked(cls, checkpoint, seed, key, value)
            print(f'CHECKED {variant} seed={seed}: checkpoint loaded, metadata valid.', flush=True)
            del model
            gc.collect()
        print('DRY RUN PASSED: all nine weights loaded; seeds, designs and configuration valid. No evaluation started.')
        return
    output = resolve_path(args.output or Path(PROJECT) / 'test_nine_weights' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    print(f'OUTPUT: {output}', flush=True)
    records = []
    for index, (variant, seed, cls, checkpoint, key, value) in enumerate(jobs, 1):
        name = f'{variant}_s{seed}'
        result_file = output / f'{name}.json'
        identity = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), settings=settings,
                        data_yaml_sha256=digest(DATA), ultralytics=ultralytics.__version__)
        if result_file.exists():
            record = json.loads(result_file.read_text(encoding='utf-8'))
            if record['identity'] != identity:
                raise RuntimeError(f'Checkpoint/config changed: use a new --output. {result_file}')
            print(f'[{index}/9] SKIP completed {name}', flush=True)
        else:
            print(f'[{index}/9] TEST {name}', flush=True)
            model = load_checked(cls, checkpoint, seed, key, value)
            metrics = model.val(**settings, project=str(output), name=name, exist_ok=False)
            per_class = []
            for pos, class_id in enumerate(metrics.box.ap_class_index):
                p, r, ap50, ap = metrics.box.class_result(pos)
                per_class.append(dict(class_id=int(class_id), name=model.names[int(class_id)],
                                      precision=float(p), recall=float(r), map50=float(ap50), map5095=float(ap)))
            record = dict(identity=identity, variant=variant, seed=seed,
                          precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                          map50=float(metrics.box.map50), map5095=float(metrics.box.map),
                          per_class=per_class, speed=metrics.speed, save_dir=str(metrics.save_dir))
            temp = result_file.with_suffix('.tmp')
            temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(result_file)
            del model, metrics
            gc.collect()
            torch.cuda.empty_cache()
        records.append(record)
        fields = ['variant', 'seed', 'precision', 'recall', 'map50', 'map5095', 'save_dir']
        with (output / 'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
    lines = ['# SHWD test: nine checkpoints', '', 'Values are percent; SD is sample SD across three training seeds.',
             'Timing in individual JSON files is validation timing, not a dedicated latency benchmark.', '',
             '| Model | mAP50 mean +/- SD | mAP50-95 mean +/- SD |', '|---|---:|---:|']
    for variant, *_ in variants:
        subset = [r for r in records if r['variant'] == variant]
        cells = []
        for metric in ('map50', 'map5095'):
            values = [r[metric] * 100 for r in subset]
            cells.append(f'{statistics.mean(values):.3f} +/- {statistics.stdev(values):.3f}')
        lines.append(f'| {variant} | {cells[0]} | {cells[1]} |')
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'ALL 9 TESTS COMPLETE\nOUTPUT: {output}', flush=True)


if __name__ == '__main__':
    main()
