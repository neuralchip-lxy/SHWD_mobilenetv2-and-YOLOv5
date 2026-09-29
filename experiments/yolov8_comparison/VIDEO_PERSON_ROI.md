# Person-first video detection pilot

`video_predict_person_roi.py` runs a COCO person detector on each full frame, expands
each whole-person box, then runs the existing SHWD P2 task combo checkpoint on
those crops. Crop-relative head boxes are translated back to the original frame
and overlapping results are merged. The original video and both weights remain
unchanged. The output is a silent MP4 plus `run_info.json`.

The SHWD checkpoint's class `person` denotes an unhelmeted **head**, not a
whole-person box. The first stage therefore needs a separate COCO-pretrained
person checkpoint. The local `yolo26n.pt` can be used for this pilot. It is an
untracked local weight, so pass its path explicitly on each machine.

From PowerShell in the `yolov8-research` repository root:

```powershell
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\experiments\yolov8_comparison\video_predict_person_roi.py" `
  --person-weights "experiments/yolov8_comparison/yolo26n.pt" `
  --helmet-weights "E:/experiment_M2Y5/analysis_runs/yolov8n_mobilenetv2_w0625_p2_task_combo_640_s0_scratch/weights/best.pt" `
  --source "E:/datasets/construction_15min.webm" `
  --max-frames 300 --show-person
```

Remove `--max-frames 300` to process the whole video. `--show-person` draws
first-stage person boxes for diagnosis. Outputs are placed in a new timestamped
folder under `outputs/analysis_runs/video_predict_person_roi/` by default.
Paths passed on the command line are resolved from the repository root.

Useful controls: `--person-conf` and `--helmet-conf` set separate thresholds;
`--expand` adds a fraction of the person-box size on every side (default 0.1);
`--helmet-imgsz 320` may reduce the cost of many crops but must be checked for
missed small heads. `--device cpu` runs without CUDA. The output video keeps the
source FPS metadata; `processing_fps` in `run_info.json` is the actual measured
throughput, including reading and writing.

For a fair comparison, run `video_predict_combo.py` on the same video and frame
limit, then inspect identical frames for false positives and missed heads. A
person missed by stage one can never be recovered by stage two. This pilot is
not a trained new model or proof of improved accuracy.
