# MobileNetV2 W0.625 + 通道重分配消融

2026-09-11。本轮从头训练，不加载上一轮 best.pt。Python/Ultralytics 环境、数据划分、
训练参数与上一轮一致，仅改变模型的通道分配和实验目录。

## 对比设计

- A：原 YOLOv8n，融合后 3,006,038 参数，8.0863 GFLOPs@640。
- B：标准 MobileNetV2 W0.625 + 原 YOLOv8n 颈部/检测头，2,743,198 参数，7.5541 GFLOPs。
- C（本轮）：B 的 P3 保持 64 通道，P4/P5 从 128/256 缩至 96/192。
  同步调整适配层、SPPF、颈部和 Detect 输入通道，连接关系与模块类型不变。
  Detect 的分类/回归分支结构与隐藏宽度不变，输入卷积随输入通道变化。

C 未融合参数 2,160,486，融合后 2,147,134，6.6790 GFLOPs@640。
相对 A 参数减少约 28.6%，计算量减少约 17.4%；尚无 C 的实测精度。
计算量统一采用 Ultralytics/THOP 统计方式，乘加计为两次浮点运算。

假设：保留高分辨率 P3 的通道容量，更多压缩低分辨率分支，争取保持检测精度。
保持通道数不代表 P3 特征完全不受影响，因为其融合输入也发生了改变。
本轮不增加注意力、上下文模块或新损失函数，先验证通道分配这一项设计。

当前 seed=0 的已完成记录：

| 组别 | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.941 | 0.604 | 0.927051 | 0.590507 |
| B | 0.938 | 0.599 | 0.929262 | 0.592854 |
| C | 待训练 | 待训练 | 待测试 | 待测试 |

验证集用于后续方案选择；最终方案固定后再报告测试结果和多个种子的波动。

## PyCharm 配置

复制上一轮 Python 配置，修改名称与脚本参数即可。

| 字段 | 填写内容 |
| --- | --- |
| 名称 | paper_yolov8n_mobilenetv2_w0625_realloc_s0 |
| Python 解释器 | C:\Users\liuji\.conda\envs\pyenv\python.exe |
| 脚本路径 | E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison\train_yolov8n_mobilenetv2_w0625_s0.py |
| 脚本参数 | --realloc |
| 工作目录 | E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison |
| 环境变量 | PYTHONUNBUFFERED=1;PYTHONIOENCODING=utf-8 |

输入 640、batch=8、epochs=200、patience=50、seed=0、SGD、AMP=True、pretrained=False。
数据保持 train=5457、val=607、test=1517，类别 hat/person。

所有模式都沿用上表的同一个脚本。原版不带 --realloc 时行为不变。

| 操作 | 脚本参数 |
| --- | --- |
| 本轮训练 | --realloc |
| 本轮检查，不训练 | --realloc --check |
| 本轮中断后恢复 | --realloc --resume |
| 本轮训练结束后测试 best.pt | --realloc --test |

新训练目录：

```text
E:\experiment_M2Y5\analysis_runs\yolov8n_mobilenetv2_w0625_realloc_640_s0_scratch
```

测试目录为上述路径加 `_test`，测试输入为 640、batch=4、FP32、conf=0.001、
iou=0.6、max_det=300，与已完成 B 的测试设置一致。

启动时终端应打印包含 `realloc` 的 Model 和 Run 路径。
训练初始摘要约 216.0 万参数；最终融合摘要约 214.7 万参数、6.7 GFLOPs。
初始与最终参数量差异来自 Conv-BN 融合。

## 实现与检查

只增加新的结构 YAML，并给原入口增加 --realloc 选项，共用训练、测试、恢复逻辑。
新旧两种配置均检查数据数量、拓扑/通道、640 前向、合成数据上的原生检测损失和反向传播、
权重保存/重载/模型重建、融合前后输出一致性；不执行训练或优化器更新。

## Git 上传

在 yolov5-paper 仓库根目录终端执行：

```bash
git add -- experiments/yolov8_comparison/train_yolov8n_mobilenetv2_w0625_s0.py experiments/yolov8_comparison/yolov8n_mobilenetv2_w0625_realloc.yaml experiments/yolov8_comparison/ABLATION_REALLOC.md
git commit -m "feat: add MobileNetV2 YOLOv8n channel reallocation ablation"
git push origin paper/model-improvement-2026
```

本地训练不依赖 Git 提交；上述命令只提交本轮代码和说明。
