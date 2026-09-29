"""Stream a local video through the existing P2 task combo checkpoint.

Paths are resolved relative to the repository. Output is a silent MP4;
the source video and checkpoint are never changed. No training is performed.
"""
import argparse
from datetime import datetime
import json
import math
import time

import cv2
from experiment_paths import path_for, resolve_path
from p2_task_structure import P2TaskYOLO


def decoding_status(count, reported_total, reached_eof):
    """Container frame counts may be rounded estimates, especially for WebM.

    Allow at most two trailing frames of discrepancy, with an explicit warning;
    this does not certify that the input has no decode errors.
    """
    missing = reported_total - count
    if reached_eof and reported_total > 0 and missing > 0:
        if missing > 2 or count == 0:
            raise RuntimeError(f'Video decoding ended early: {count}/{reported_total}; output is partial.')
        return 'completed_with_warning', (
            f'Decoded {count} frames; container reports {reported_total}. '
            f'Trailing difference of {missing} frame(s) tolerated; output retained.')
    return 'completed', None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', help='New output directory; must not already exist.')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-frames', type=int, default=0, help='0: entire video.')
    args = parser.parse_args()
    if args.max_frames < 0 or args.imgsz <= 0 or not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error('Invalid size, frame limit or threshold.')
    weights, source = resolve_path(args.weights), resolve_path(args.source)
    for file in (weights, source):
        if not file.is_file():
            raise FileNotFoundError(file)
    model = P2TaskYOLO(str(weights), task='detect')
    if model.model.yaml.get('p2_task_design') != 'adjacent_task_fusion_v1':
        raise ValueError('Expected a P2 task combo checkpoint.')
    cap = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {source}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise RuntimeError('Cannot read first frame or valid source FPS.')
        height, width = frame.shape[:2]
        output = resolve_path(args.output) if args.output else (
            path_for('runs') / 'video_predict' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True, exist_ok=False)
        target = output / 'detections.mp4'
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError('MP4 video writer is unavailable.')
        info = dict(weights=str(weights), source=str(source), names=model.names,
                    source_fps=fps, source_frames=total, width=width, height=height,
                    settings=vars(args), audio=False, status='running')
        report = output / 'run_info.json'
        report.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Classes: {model.names}\nOUTPUT: {output}\nSource: {total} frames, {fps:.3f} FPS', flush=True)
        count, started = 0, time.perf_counter()
        try:
            while ok:
                result = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                                       iou=args.iou, device=args.device, quantize=32,
                                       max_det=300, save=False, verbose=False)[0]
                writer.write(result.plot())
                count += 1
                if count % 300 == 0:
                    print(f'{count}/{total} frames; processing {count/(time.perf_counter()-started):.1f} FPS', flush=True)
                if args.max_frames and count >= args.max_frames:
                    break
                ok, frame = cap.read()
            info['status'], warning = decoding_status(count, total, reached_eof=not ok)
            info['reported_frame_count_difference'] = total - count if not ok else None
            info['stop_reason'] = 'end_of_stream' if not ok else 'frame_limit'
            if warning:
                info['warning'] = warning
                print(f'WARNING: {warning}', flush=True)
        finally:
            info.update(processed_frames=count, elapsed_seconds=time.perf_counter()-started)
            if info['status'] not in ('completed', 'completed_with_warning'):
                info['status'] = 'interrupted_or_failed'
            report.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Done: {count} frames\nVIDEO: {target}', flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    main()
