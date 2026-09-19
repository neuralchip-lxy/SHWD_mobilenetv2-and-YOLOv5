# P5 Transformer exploratory ablation

This is a new candidate, not a change to the frozen paper model. No accuracy or novelty claim is established by adding the block.

## Architecture

Use `train_p5_transformer.py` and its P5TransformerYOLO/Trainer/Model wrappers. Plain YOLO loading the YAML would not install this module or the custom task head.

The original 30 graph indices are retained. Layer 7 is wrapped as Sequential(original SPPF, P5Transformer), so both its top-down upsampling consumer (layer 10) and bottom-up skip consumer (layer 27) see enhanced features. The original P2 task-fusion head and all other layers retain their topology and channel dimensions.

At 640 square input, the new block receives Bx192x20x20. A bias-free 1x1 projection reduces channels to 96. Spatial locations become 400 tokens. Dynamic 2D sine/cosine position encoding works for rectangular inputs too. One pre-normalized attention/FFN block uses 4 heads, QKV projections, FP32 softmax under AMP, a 96->192->96 GELU FFN, and residual connections. The output is projected to 192 channels and added to the block input. No dropout, no pretrained Transformer, and no additional loss is used. Position embeddings are included in the shared QKV input. This is an experimental implementation, not an exact reproduction of RT-DETR or MobileViT.

## Matched training

All training settings are inherited from train_yolov8n_s0.TRAIN_ARGS, changing only name and seed. SHWD split unchanged: 640, batch 8, epochs 200, patience 50, SGD, AMP, scratch. Start seed 0. Screen on validation data, then repeat seeds 1/2 if warranted. Do not select modules using test-set metrics. The existing full-model seed-0 run is the control; no retraining of that control is required for this initial screen.

PowerShell, in experiments/yolov8_comparison:

```powershell
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_p5_transformer.py" --seed 0
```

Interrupted training only: add `--resume`. Independent validation of best.pt: add `--val`. Configuration-only check: `--dry-run`. Synthetic checks: `--check`.

Outputs: E:/experiment_M2Y5/analysis_runs/yolov8n_mobilenetv2_w0625_p2_task_p5_transformer_640_s0_scratch

## Verified synthetic checks

Square 640 and rectangular 320x512 forward outputs, original-layer shape contracts, trainer reconstruction, save/reload, CPU and CUDA AMP native loss/backward, and Conv-BN fusion equivalence passed. Fixed DFL parameters are correctly excluded from trainable-gradient assertions. No optimizer steps or dataset training were run during checks.

| Model | Fused parameters | Profiled GFLOPs at 640 |
|---|---:|---:|
| Existing full P2 combo | 2,085,464 | 6.5270 |
| + P5 Transformer | 2,197,112 | 6.6776 |

Profiling uses full 640 input and includes QK and AV matrix products through an explicit THOP hook; softmax, positional encoding and elementwise overhead are not fully counted. Default Ultralytics FLOPs may omit attention products and should not replace this report. Synthetic AMP batch=8, 640, 80 targets/image: peak allocated 3.67 GiB, reserved 3.81 GiB. This excludes full training/optimizer overhead and is not a runtime guarantee or latency benchmark.

Actual accuracy, training time and inference latency remain to be measured. Adding a Transformer is not by itself evidence of a novel algorithm.
