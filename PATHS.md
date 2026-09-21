# 实验路径配置

适用范围：experiments/yolov8_comparison 内的 YOLOv8/SHWD/SHEL5K 训练、测试、数据审计与转换入口。旧 YOLOv5 独立工具、历史日志和旧说明中的路径未批量修改。

## 规则

所有命令行路径和 JSON 配置相对路径都以仓库根目录为基准，不依赖 PowerShell 当前目录。传给 Ultralytics 的 project 始终是解析后的绝对路径，因此不会再次被自动拼到 runs/detect 下。

仓库默认配置示例为 experiment_paths.example.json。没有本地配置时：

- runs：outputs/analysis_runs
- shwd_yaml：SHWD_YOLO/shwd.yaml
- shwd_root：SHWD_YOLO（该目录下应有 images/train 等）
- shel5k_yolo：datasets/SHEL5K6_YOLO_v1
- shel5k_source：datasets/SHEL5K_original/9rcv8mm682-4/Safety Helmet Wearing Dataset
- shel5k_audit：outputs/shel5k_preparation

## 在另一台电脑上使用

在仓库根执行：

```powershell
Copy-Item experiment_paths.example.json experiment_paths.local.json
```

编辑 experiment_paths.local.json：每个值都可以填写相对仓库根目录的路径，或本机绝对路径。没有文件则使用默认值；不要覆盖已有本地配置。此文件已加入 .gitignore，不应上传。可以通过环境变量 M2Y5_PATHS_CONFIG 指定另一份 JSON（相对路径仍基于仓库根）。

SHEL5K 入口的 --data/--project 显式参数优先于默认值，相对路径也以仓库根解析。例如 --project outputs/analysis_runs/shel5k6。

已有数据集仍需单独复制。新转换生成的 shel5k.yaml 使用 path: .，表示 YAML 所在目录。入口会生成解析为绝对路径的运行副本，放在 runs/_resolved_data，不改写原 YAML。不要绕过入口把这种相对 YAML 直接交给不同工作目录下的原生 yolo 命令。

SHWD 默认入口使用配置中的 shwd_root；SHEL5K 默认入口使用 shel5k_yolo，因此可以迁移带有旧盘符的 YAML 而不修改原文件。手动 --data 指定 YAML 时，其 path 若为相对路径，以 YAML 所在目录解析；若写死旧盘符，则需自行改正或使用默认配置入口。

## 当前电脑保持原位置

本机已设置忽略的 experiment_paths.local.json，仍指向：

- 训练结果：E:/experiment_M2Y5/analysis_runs
- SHEL5K 结果：E:/experiment_M2Y5/analysis_runs/shel5k6
- SHEL5K 数据：E:/experiment_M2Y5/datasets/SHEL5K6_YOLO_v1
- SHWD 数据：项目上一级的 SHWD_YOLO

原有数据、权重、结果目录未移动；已有 YAML 字节不变，原断点身份校验保持有效。旧的 shel5k6 combo seed0 特殊 runs/detect 目录也没有移动，不会被自动纳入新目录。

正在运行的 Python 进程已加载的路径配置不会自动刷新；下一次启动脚本时读取配置。已有绝对路径命令仍有效。不要在训练中途手动更换数据或输出配置。

## 验证

已核对：修改前后 SHWD TRAIN_ARGS 完全一致；当前 SHEL5K YAML 路径不变；从其他工作目录执行两个训练入口的 --dry-run 仍指向原 E 盘输出；模拟新项目副本时使用项目内目录；相对 YAML 生成运行副本并保留原文。未启动训练。

本次改动备份位于 E:/experiment_M2Y5/path_migration_20260921。Git 只提交脚本、示例配置、本说明和 .gitignore；不提交本地配置、outputs、datasets。
