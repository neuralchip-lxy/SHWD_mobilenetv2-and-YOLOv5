# SHWD MobileNetV2-YOLOv5

本项目面向施工/工业场景的安全帽佩戴检测。模型以 YOLOv5 的 FPN、PAN 和 Detect Head 为基础，使用 MobileNetV2 倒残差网络替换原始 YOLOv5s Backbone，在保持检测能力的同时降低参数量、计算量和模型体积，适合资源受限的边缘部署场景。

## 模型说明

| 项目 | 内容 |
|---|---|
| 检测类别 | `hat`（安全帽）、`person`（人员） |
| 输入尺寸 | 640 × 640 |
| Backbone | MobileNetV2 Inverted Residual Blocks（ReLU6） |
| Neck / Head | YOLOv5 FPN + PAN / Detect |
| 最新权重 | [`best.pt`](yolov5-master/runs/train/shwd_mobilenetv24/weights/best.pt) |

网络配置位于 [`models/yolov5s_mobilenetv2.yaml`](yolov5-master/models/yolov5s_mobilenetv2.yaml)，MobileNetV2 模块实现在 [`models/common.py`](yolov5-master/models/common.py)。

## 验证集性能

在 SHWD 验证集（607 张图像、9,925 个实例）上的结果如下：

| 类别 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| hat | 0.940 | 0.877 | 0.935 | 0.678 |
| person | 0.916 | 0.855 | 0.913 | 0.455 |
| all | **0.928** | **0.866** | **0.924** | **0.567** |

## 轻量化效果

| 模型 | 参数量 | GFLOPs | 权重大小 | mAP@0.5 |
|---|---:|---:|---:|---:|
| YOLOv5s 基线 | 7.016 M | 15.8 | 14.4 MB | 0.947 |
| MobileNetV2-YOLOv5 | 2.369 M | 6.4 | 5.1 MB | 0.924 |

与基线相比，融合模型的参数量减少 **66.2%**，计算量减少 **59.5%**，权重文件减小 **64.6%**。当前比较使用相同验证集；基线使用预训练权重而融合模型从头训练，严格消融实验应进一步统一初始化方式与 batch size。

## 环境安装

```powershell
cd yolov5-master
pip install -r requirements.txt
```

推荐使用支持 CUDA 的 PyTorch 环境。

## 数据集结构

数据集不包含在仓库中。准备为 YOLO 格式后，目录应为：

```text
SHWD_YOLO/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

在 [`SHWD_YOLO/shwd.yaml`](SHWD_YOLO/shwd.yaml) 中设置数据集根目录。每个图像应在对应的 `labels` 目录中有同名 `.txt` 标注文件。

## 训练

在 `yolov5-master` 目录中执行：

```powershell
python train.py --img 640 --batch 16 --epochs 100 --data ..\SHWD_YOLO\shwd.yaml --cfg models\yolov5s_mobilenetv2.yaml --weights "" --device 0 --name shwd_mobilenetv2 --workers 0
```

`--workers 0` 可避免 Windows 多进程数据加载导致的兼容性问题。若环境稳定，可尝试提高该数值以加快读取速度。

## 推理

使用最新模型检测单张图像、文件夹或视频：

```powershell
python detect.py --weights runs\train\shwd_mobilenetv24\weights\best.pt --source path\to\image_or_video --img 640 --conf 0.25 --device 0
```

结果默认保存到 `runs/detect/`。

## 训练产物与报告

- 最新权重：`yolov5-master/runs/train/shwd_mobilenetv24/weights/best.pt`
- 训练曲线：`yolov5-master/runs/train/shwd_mobilenetv24/results.png`
- 详细对比报告：`yolov5-master/runs/train/shwd_mobilenetv24/baseline_vs_mobilenetv2_yolov5_report.md`
