# P2 detail preservation and spatial task fusion

This is a research-branch SHWD experiment built on the existing `p2_task_combo`
model. It does not modify that checkpoint, the paper worktree, Ultralytics, the
four output strides (4/8/16/32), DFL/CIoU, or task-aligned assignment.

## Hypothesis

At 640 input, the SHWD training split contains 6,422 `hat` and 79,778
`person` instances. After aspect-preserving resize to 640, roughly 29% of
`hat` and 82% of `person` boxes have area below 32x32 pixels. The proposal
tests whether preserving some 2x2 spatial phase information before the
stride-4 feature, and selecting adjacent-scale context per location, helps
without replacing the entire backbone or neck. The size/class association
is a potential dataset shortcut, not a proven model behavior.

## Variants

- `detail`: Replace only the first stride-4-producing MobileNetV2 depthwise
  downsample. Three quarters of expanded channels keep stride-2 depthwise
  convolution; one quarter use PixelUnshuffle(2) and a 1x1 projection.
- `spatial`: Keep the original backbone. At the P2 and P3 box/class towers,
  replace fixed local/context concatenation with separate per-position gains.
  Gains start at 1, so the new fusion initially equals the old one.
- `combo`: Both changes. This is the default variant.

Each variant has its own YAML and distinct output name. All train from
scratch with the existing seed-0, 640, batch-8, 200-epoch policy. The
`--check` mode only runs synthetic forward/backward/serialization checks.

Run from `experiments/yolov8_comparison` in PowerShell:

```powershell
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_p2_detail_spatial.py" --variant combo --check
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_p2_detail_spatial.py" --variant combo --seed 0
```

For component ablations, replace `combo` with `detail` or `spatial`. Use
`--resume` only after interruption, `--val` for a saved best checkpoint, and
`--test` only after the configuration has been selected on validation data.
The paths come from the worktree's `experiment_paths.local.json`, so default
outputs remain under `yolov8-research/outputs/analysis_runs`.

Do not describe this as an established accuracy improvement until the same
validation protocol and repeated seeds support it. PixelUnshuffle may need
operator conversion or a different implementation for a specific FPGA toolchain.
