"""Detect people first, then run the SHWD P2 task combo model on person crops.

This is an inference-only pilot. The SHWD class named ``person`` means an
unhelmeted head; it is not used for whole-body detection here.
"""

import argparse
from datetime import datetime
import json
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO

from experiment_paths import path_for, resolve_path
from p2_task_structure import P2TaskYOLO


def expanded_roi(box, width, height, expand):
    """Return a clipped, nonempty person crop in integer xyxy coordinates."""
    x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = (x2 - x1) * expand, (y2 - y1) * expand
    left = max(0, math.floor(x1 - dx))
    top = max(0, math.floor(y1 - dy))
    right = min(width, math.ceil(x2 + dx))
    bottom = min(height, math.ceil(y2 + dy))
    return (left, top, right, bottom) if right > left and bottom > top else None


def suppress_duplicates(detections, threshold):
    """Class-aware NMS after mapping crop detections into the full frame."""
    if not detections:
        return []
    ordered = sorted(detections, key=lambda det: det['confidence'], reverse=True)
    kept = []
    for candidate in ordered:
        box = candidate['box']
        duplicate = False
        for selected in kept:
            if candidate['class_id'] != selected['class_id']:
                continue
            other = selected['box']
            x1, y1 = max(box[0], other[0]), max(box[1], other[1])
            x2, y2 = min(box[2], other[2]), min(box[3], other[3])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            area2 = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
            union = area1 + area2 - intersection
            if union > 0 and intersection / union > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def draw_detection(frame, box, label, color):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--person-weights', required=True,
                        help='COCO-pretrained whole-person detector .pt; class 0 must be person.')
    parser.add_argument('--helmet-weights', required=True, help='SHWD P2 task combo best.pt.')
    parser.add_argument('--source', required=True, help='Input video.')
    parser.add_argument('--output', help='New output directory; must not already exist.')
    parser.add_argument('--person-imgsz', type=int, default=640)
    parser.add_argument('--helmet-imgsz', type=int, default=640)
    parser.add_argument('--person-conf', type=float, default=0.25)
    parser.add_argument('--helmet-conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--merge-iou', type=float, default=0.5)
    parser.add_argument('--expand', type=float, default=0.1,
                        help='Fraction of person-box width/height added on every side.')
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-frames', type=int, default=0, help='0 processes the entire video.')
    parser.add_argument('--show-person', action='store_true', help='Draw stage-one person boxes.')
    args = parser.parse_args()
    if (args.person_imgsz <= 0 or args.helmet_imgsz <= 0 or args.max_frames < 0
            or not 0 <= args.person_conf <= 1 or not 0 <= args.helmet_conf <= 1
            or not 0 <= args.iou <= 1 or not 0 <= args.merge_iou <= 1
            or not 0 <= args.expand <= 1):
        parser.error('Invalid image size, frame limit, confidence, IoU or expansion.')

    person_weights = resolve_path(args.person_weights)
    helmet_weights = resolve_path(args.helmet_weights)
    source = resolve_path(args.source)
    for item in (person_weights, helmet_weights, source):
        if not item.is_file():
            raise FileNotFoundError(item)

    person_model = YOLO(str(person_weights), task='detect')
    if person_model.names.get(0) != 'person':
        raise ValueError('Person checkpoint must have class 0 named person (COCO).')
    helmet_model = P2TaskYOLO(str(helmet_weights), task='detect')
    if helmet_model.model.yaml.get('p2_task_design') != 'adjacent_task_fusion_v1':
        raise ValueError('Expected the SHWD P2 task combo checkpoint.')
    if list(helmet_model.names.values()) != ['hat', 'person']:
        raise ValueError(f'Expected SHWD [hat, person] classes, got {helmet_model.names}.')

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
            path_for('runs') / 'video_predict_person_roi' /
            datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True, exist_ok=False)
        target = output / 'detections.mp4'
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError('MP4 video writer is unavailable.')
        report = output / 'run_info.json'
        info = dict(person_weights=str(person_weights), helmet_weights=str(helmet_weights),
                    source=str(source), source_fps=fps, source_frames=total,
                    width=width, height=height, settings=vars(args), audio=False,
                    status='running', shwd_person_class='unhelmeted head, not full body')
        report.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'OUTPUT: {output}\nSource: {total} frames, {fps:.3f} FPS', flush=True)

        count = person_count = head_count = 0
        person_seconds = helmet_seconds = 0.0
        started = time.perf_counter()
        try:
            while ok:
                stage_started = time.perf_counter()
                people = person_model.predict(
                    source=frame, imgsz=args.person_imgsz, conf=args.person_conf,
                    iou=args.iou, classes=[0], device=args.device, max_det=300,
                    save=False, verbose=False)[0]
                person_seconds += time.perf_counter() - stage_started
                person_boxes = people.boxes.xyxy.cpu().numpy()
                rois = [expanded_roi(box, width, height, args.expand) for box in person_boxes]
                rois = [roi for roi in rois if roi is not None]
                person_count += len(rois)
                crops = [np.ascontiguousarray(frame[y1:y2, x1:x2])
                         for x1, y1, x2, y2 in rois]
                detections = []
                if crops:
                    stage_started = time.perf_counter()
                    results = helmet_model.predict(
                        source=crops, imgsz=args.helmet_imgsz, conf=args.helmet_conf,
                        iou=args.iou, device=args.device, quantize=32,
                        max_det=300, save=False, verbose=False)
                    helmet_seconds += time.perf_counter() - stage_started
                    if len(results) != len(rois):
                        raise RuntimeError('Second-stage result count does not match person crops.')
                    for (x1, y1, _, _), result in zip(rois, results):
                        boxes = result.boxes
                        for box, score, class_id in zip(boxes.xyxy.cpu().numpy(),
                                                        boxes.conf.cpu().numpy(),
                                                        boxes.cls.int().cpu().numpy()):
                            translated = [float(box[0] + x1), float(box[1] + y1),
                                          float(box[2] + x1), float(box[3] + y1)]
                            detections.append(dict(box=translated, confidence=float(score),
                                                   class_id=int(class_id)))
                detections = suppress_duplicates(detections, args.merge_iou)
                head_count += len(detections)
                rendered = frame.copy()
                if args.show_person:
                    for roi in rois:
                        draw_detection(rendered, roi, 'person ROI', (255, 190, 0))
                for det in detections:
                    is_hat = det['class_id'] == 0
                    label = 'hat' if is_hat else 'no_helmet_head'
                    color = (0, 200, 0) if is_hat else (0, 60, 255)
                    draw_detection(rendered, det['box'],
                                   f"{label} {det['confidence']:.2f}", color)
                writer.write(rendered)
                count += 1
                if count % 300 == 0:
                    elapsed = time.perf_counter() - started
                    print(f'{count}/{total} frames; processing {count/elapsed:.1f} FPS; '
                          f'{person_count} person ROIs', flush=True)
                if args.max_frames and count >= args.max_frames:
                    break
                ok, frame = cap.read()
            missing = total - count
            if not ok and missing > 2:
                raise RuntimeError(f'Video decoding ended early: {count}/{total}; output is partial.')
            info['status'] = 'completed_with_warning' if not ok and missing > 0 else 'completed'
            if info['status'] == 'completed_with_warning':
                info['warning'] = f'Decoded {count} frames; container reports {total}.'
                print(f"WARNING: {info['warning']}", flush=True)
            info['stop_reason'] = 'end_of_stream' if not ok else 'frame_limit'
        finally:
            elapsed = time.perf_counter() - started
            info.update(processed_frames=count, person_rois=person_count,
                        head_detections=head_count, elapsed_seconds=elapsed,
                        processing_fps=count / elapsed if elapsed > 0 else None,
                        person_inference_seconds=person_seconds,
                        helmet_inference_seconds=helmet_seconds)
            if info['status'] not in ('completed', 'completed_with_warning'):
                info['status'] = 'interrupted_or_failed'
            report.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Done: {count} frames; {count/elapsed:.1f} FPS\nVIDEO: {target}', flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    main()
