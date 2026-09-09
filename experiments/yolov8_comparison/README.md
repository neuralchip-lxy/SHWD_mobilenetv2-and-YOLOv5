# YOLOv8n 对照实验（准备于2026-09-09）

用途：帮助判断是否保留MobileNetV2-W0.625作为研究主线。只准备脚本与检查，用户自行启动训练。

## 操作

PyCharm解释器：`C:/Users/liuji/.conda/envs/pyenv/python.exe`。
脚本：`E:/experiment_M2Y5/yolov8_comparison/train_yolov8n_s0.py`。
工作目录：`E:/experiment_M2Y5/yolov8_comparison`。首次训练参数留空；中断后恢复参数填`--resume`；仅检查填`--check`。
训练输出：`E:/experiment_M2Y5/analysis_runs/yolov8n_640_s0_scratch`。存在目录时首次训练会拒绝启动，避免覆盖/混淆。

## 对照口径

- 当前环境Ultralytics8.4.89、PyTorch2.11.0+cu130，未升级/降级环境。
- yolov8n.yaml随机初始化，pretrained=False，不从COCO预训练权重开始。
- 同一SHWD YAML与图像划分：5457/607/1517，hat/person两类。
- 640输入、batch8、seed0、最多200epochs、patience50、workers4。
- SGD、lr0=.01/lrf=.1、momentum=.937、weight_decay=.0005，warmup3epochs。
- 对齐可对应的数据增强参数，包括scale=.9、mixup=.1、mosaic=1，close_mosaic=0（与原YOLOv5全程保留Mosaic一致）。
- YOLOv8原生loss/正样本分配保留，box7.5/cls.5/dfl1.5；不复制YOLOv5的obj/anchor参数或loss权重。
- 验证conf=.001、NMS IoU=.6、max_det300，与原val.py默认阈值一致。模型内置验证实现、最佳轮次选择、增强实现、AMP等细节仍存在跨框架差异，不称为严格仅改网络结构的实验，也不代表各架构独立调参后的最优成绩。
- 数据划分检查与两类模型640 CPU前向已通过。融合模型3,006,038参数、约8.1GFLOPs（当前Ultralytics统计口径）；与旧项目最终比较应统一profile及评估口径。
- Ultralytics可能下载yolo26n.pt作AMP兼容性检查；这是环境检查辅助模型，不是本次YOLOv8n的训练初始化。脚本中训练模型固定yolov8n.yaml、pretrained=False。
- 当前只运行seed0作初筛，不能据此称某个架构稳定优于另一个。按验证集决定后续方案，test用于固定方案报告。

## Git

脚本位于实验工作区，尚不在原Git仓库。正式保留时将脚本与说明指定加入仓库；不要加入权重、数据集或整个analysis_runs。本轮没有执行commit/push。
