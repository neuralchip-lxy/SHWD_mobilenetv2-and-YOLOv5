# YOLOv8n / MobileNetV2 W0.625 骨干替换消融

日期：2026-09-10。代码适配现有 Ultralytics 8.4.89、Python 3.11 环境。

## 本轮只研究什么

对照组 A 是已完成的 YOLOv8n，实验组 B 将其 SPPF 之前的骨干替换为
MobileNetV2（width_mult=0.625），并用三个 1×1 Conv-BN-SiLU 适配层连接原网络。
两组的 SPPF、PAN/FPN、C2f、Detect 的结构与通道保持一致；损失函数、增强、
训练预算、数据划分和初始化策略保持一致。适配层计入 B 的参数量和计算量。

这轮检验骨干替换的精度/计算量取舍，不能提前宣称精度提高或已经形成创新。
之后再在 B 上逐项验证精度恢复方法，并最终补组合实验和多个随机种子。
开发阶段根据验证集选择方案，固定方案后再报告测试集结果。

## 宽度系数的具体含义

- 使用标准 MobileNetV2 的阶段重复次数 1/2/3/4/3/3/1、倒残差和 ReLU6；
  去除分类用的最后 1280 通道扩展层与分类器，从头训练，不下载 ImageNet 权重。
- 0.625 仅用于 MobileNetV2 骨干。通道按 MobileNet 的 8 对齐及 10% 防缩减规则取整。
- P3/P4/P5 输出为 24/64/200 通道，步长为 8/16/32；适配为 64/128/256 后连接原 YOLOv8n。
- YAML 中全局 width_multiple=1.0，因为其余部分已经写成 nano 实际通道数。
  不要把全局 width_multiple 改成 0.625，否则会同时改变颈部和检测头。
- 旧 YOLOv5 工程采用简化的 MobileNetV2 风格骨干，阶段与通道设置不同。
  本轮的 0.625 与旧工程同名系数不能视为完全相同的网络。
- 本 YAML 固定为 W0.625。如果以后更换宽度，还需同步 Index 的通道声明和实验名称。

## 控制变量与现有结果

训练参数直接复用同目录 train_yolov8n_s0.py 的 TRAIN_ARGS，仅改变输出名称。
640 输入、batch=8、seed=0、SGD、200 epochs、patience=50、pretrained=False、AMP=True。
已核对原基线 args.yaml 的这些设置。数据集仍为 train=5457、val=607、test=1517，
类别为 hat/person。测试设置为 640、batch=4、conf=0.001、iou=0.6、max_det=300、FP32。

两组均使用本机同一 Ultralytics get_flops 工具，在 640 输入下统计融合后完整检测器：

| 配置 | 融合后参数 | GFLOPs@640 | 验证 mAP50-95 | 测试 mAP50 | 测试 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A：YOLOv8n | 3,006,038 | 8.0863 | 0.604 | 0.927051 | 0.590507 |
| B：MobileNetV2 W0.625 + 原颈部/头 | 2,743,198 | 7.5541 | 待训练 | 待测试 | 待测试 |

B 未融合参数为 2,757,158；融合后参数减少约 8.7%，计算量减少约 6.6%。
此次压缩幅度有限，因为原颈部与检测头仍保留。这是为了隔离骨干替换的影响。
GFLOPs 下降不能直接等同于设备时延下降，时延需统一设备和测试方法另行测量。

## PyCharm 运行配置

新建 Python 运行配置，选择 script：

| 字段 | 值 |
| --- | --- |
| 名称 | paper_yolov8n_mobilenetv2_w0625_s0 |
| Python 解释器 | C:\Users\liuji\.conda\envs\pyenv\python.exe |
| 脚本路径 | E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison\train_yolov8n_mobilenetv2_w0625_s0.py |
| 脚本参数 | 留空即训练 |
| 工作目录 | E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison |
| 环境变量 | PYTHONUNBUFFERED=1;PYTHONIOENCODING=utf-8 |

运行按钮启动训练。输出目录：

```text
E:\experiment_M2Y5\analysis_runs\yolov8n_mobilenetv2_w0625_640_s0_scratch
```

同一脚本的其他模式（互斥，按需在“脚本参数”中填写）：

| 参数 | 用途 |
| --- | --- |
| --check | 数据数量、网络与合成张量检查，无训练、无测试集评估 |
| --resume | 仅对中断的训练，从本实验 last.pt 恢复 |
| --test | 训练完成后，使用本实验 best.pt 评估 test，输出到同名目录加 _test |

运行此入口会先注册自定义骨干；训练、测试、恢复均应使用此入口。
后续独立加载自定义权重时，要确保同目录模块可导入，并先调用
mobilenetv2_backbone.register_backbone()。无需修改已安装的 ultralytics 源码。

## 已完成的代码检查

- 数据数量与类别一致，训练配置除输出名称外与基线一致。
- SPPF 与每个颈部/检测头模块的类型和 state_dict 张量形状与 YOLOv8n 一致。
- 640 前向输出为 (1, 6, 8400)，三个骨干输出分辨率为 80/40/20。
- 原生检测损失与反向传播有限，所有可训练参数收到有限梯度；未执行优化器更新。
- 自定义权重保存、重新加载和按 YAML 重建通过。
- Conv-BN 融合后无残留 BN，融合前后输出在浮点误差范围内一致。

## 手动提交 Git

在 yolov5-paper 仓库根目录的终端执行；不需要提交训练权重或数据。

```bash
git add -- experiments/yolov8_comparison/mobilenetv2_backbone.py experiments/yolov8_comparison/yolov8n_mobilenetv2_w0625.yaml experiments/yolov8_comparison/train_yolov8n_mobilenetv2_w0625_s0.py experiments/yolov8_comparison/ABLATION_MOBILENETV2.md
git commit -m "feat: add YOLOv8n MobileNetV2 w0625 backbone ablation"
git push origin paper/model-improvement-2026
```

本地直接训练不依赖 Git 提交。需要在另一台电脑运行时，先推送，再在另一台电脑拉取代码；
解释器和数据目录如不同，需要按那台电脑的路径配置。
