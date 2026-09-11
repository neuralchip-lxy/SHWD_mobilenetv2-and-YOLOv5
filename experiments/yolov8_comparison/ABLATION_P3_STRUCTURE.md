# 方案一：P3 跨层门控融合 + 可重参数化特征块

2026-09-11。已实现并进行合成张量检查，未启动正式训练。

## 本轮改动

基于已完成的 MobileNetV2 W0.625 + realloc（P3/P4/P5=64/96/192）。
G：替换第 12 层 Concat。两路信息生成 2×sigmoid 空间门控，仅缩放上采样语义特征，浅层侧路保留；输出依旧按原顺序 [96 通道语义，64 通道侧路] 拼接。
门控为 1×1 Conv(160→16)、DWConv3×3、1×1 Conv(16→1)，最后一层零初始化，使初始门控为 1。
R：保留第 13 层 C2f 外层结构，将其中一个 Bottleneck 换成 PW(32→64) → 并行 DW5/DW3/DW1 → 求和后激活 → PW(64→32) → 残差。
并行分支内部是线性 Conv-BN；评估时将三支合并为一个 DW5。门控、点卷积、激活和残差仍然保留。
P3 输出仍供 Detect 以及后续 P4 下采样共同使用。

这些是受加权跨层融合与结构重参数化启发的设计候选，不是已经证实的精度提升或新颖性结论。
参考：[RepVGG](https://arxiv.org/abs/2101.03697)、[RepViT](https://github.com/THU-MIG/RepViT)、[EfficientDet/BiFPN](https://arxiv.org/abs/1911.09070)。

## 当前统计

统计完整两类别检测器，640 输入，与前轮一致采用 Ultralytics/THOP 口径。GFLOPs 是算子计数估计，不能替代延迟实测，部分逐元素操作并不由 THOP 完整计数。

| 版本 | 训练结构参数量（未融合） | 推理结构参数量（融合后） | 推理 GFLOPs@640 |
| --- | ---: | ---: | ---: |
| 上一轮 realloc | 2,160,486 | 2,147,134 | 6.6790 |
| 仅 G | 2,163,271 | 2,149,887 | 6.7138 |
| 仅 R | 2,148,838 | 2,134,494 | 6.5160 |
| G+R 组合（本轮先跑） | 2,151,623 | 2,137,247 | 6.5508 |

组合版更小，是因为 R 替换了原来的密集卷积；不代表新增模块必然提高精度。
新三组精度均待训练。旧 realloc 验证 mAP50-95=0.599，测试=0.588909；原 YOLOv8n 验证=0.604，测试=0.590507。

## PyCharm：本轮组合版训练

新建或复制 Python 运行配置：

| 项目 | 内容 |
| --- | --- |
| 名称 | paper_yolov8n_mobilenetv2_p3_combo_s0 |
| 运行方式 | script |
| Python 解释器 | C:\Users\liuji\.conda\envs\pyenv\python.exe |
| 脚本路径 | E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison\train_yolov8n_mobilenetv2_p3_s0.py |
| 脚本参数 | --variant combo |
| 工作目录 | E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison |
| 环境变量 | PYTHONUNBUFFERED=1;PYTHONIOENCODING=utf-8 |

注意使用新的 p3 入口，不是旧的 train_yolov8n_mobilenetv2_w0625_s0.py；这里也不用 --realloc。
参数留空默认为 combo，但建议明确写 --variant combo，便于识别。

本轮仍是 640、batch8、200 epochs、patience50、seed0、SGD、AMP=True、从头训练，不加载前轮 best.pt。
训练参数直接复用 train_yolov8n_s0.py，测试参数复用原 MobileNetV2 入口，只有模型和结果名称变化。
数据 train/val/test=5457/607/1517，hat/person 两类别。

训练目录：

```text
E:\experiment_M2Y5\analysis_runs\yolov8n_mobilenetv2_w0625_realloc_p3_combo_640_s0_scratch
```

启动终端应显示 Variant: combo，以及 P3 design: gate=True, rep=True。
初始未融合摘要参数为 2,151,623；最终融合摘要参数为 2,137,247，约 6.6 GFLOPs。
训练期多分支会带来额外计算，不能把融合后的开销写成训练开销。

## 模式与消融

| 操作 | 同一新脚本的参数 |
| --- | --- |
| 本轮组合训练 | --variant combo |
| 组合检查，不训练 | --variant combo --check |
| 组合中断后恢复 | --variant combo --resume |
| 组合训练结束后测试 | --variant combo --test |
| 后续仅门控消融训练 | --variant gate |
| 后续仅特征块消融训练 | --variant rep |

gate/rep 同样可以搭配 --check、--resume、--test，输出目录分别含 p3_gate / p3_rep。
不自动连续启动多个实验。所有测试都读取所选版本自己的 best.pt，测试结果目录加 _test。
后续用验证集选择方案，补齐 C/G/R/G+R 消融；最终候选再补 seed1、2及机制对照。

## 模型重建和转换

三个 YAML 通过 p3_design 开关明确记录 G/R 是否启用。
由于标准 Ultralytics YAML 解析器不直接支持这两个自定义结构，P3DetectionModel 在解析标准图后应用开关，P3DetectionTrainer 保证训练及恢复重建时使用相同模型类。
因此，创建这些 YAML 对应的模型必须使用本目录的 p3_structure.P3YOLO，而不能用普通 YOLO 直接创建，否则普通构造器不会应用 p3_design。
保存的 .pt 包含自定义模型类，复制到其他机器时需保留本目录的模块并保证可以导入。
本次未修改 site-packages，也未改动旧实验入口。

自定义 fuse() 显式完成 RepDepthwise5 的多分支合并，再调用原生 Conv-BN 融合。不是仅删除 BatchNorm。
后续导出/上板仍需单独验证算子支持、精度与时延；目前完成的是 PyTorch 路径的等价转换。

## 已完成检查

三种开关配置均通过：

- 数据划分和训练参数一致；除第 12/13 层以外，模块类型和参数形状保持不变。
- 门控恒等初始化、浅层通路不被缩放、非零门控对语义通路的实际影响。
- 非默认 BN 统计下、非方形输入的分支融合等价性及重复融合。
- 640 前向输出 (1,6,8400)，Detect 输入通道与分辨率不变。
- CPU 与 CUDA AMP 上的原生损失/反向传播；全部可训练参数梯度有限，新模块梯度非零。
- 训练器实际 get_model() 路径重建，权重逐项一致。
- 临时 .pt 保存/加载、恢复时重建开关和权重结构一致。
- 使用更新过的 BN 统计和非恒等门控检查整网融合前后输出等价，无残留 BN。

检查没有优化器更新，没有读取测试集图像作评估，没有启动正式训练。

## Git 手动提交

在 yolov5-paper 根目录执行：

```bash
git add -- experiments/yolov8_comparison/p3_structure.py experiments/yolov8_comparison/check_p3_structure.py experiments/yolov8_comparison/train_yolov8n_mobilenetv2_p3_s0.py
git add -- experiments/yolov8_comparison/yolov8n_mobilenetv2_w0625_realloc_p3_combo.yaml experiments/yolov8_comparison/yolov8n_mobilenetv2_w0625_realloc_p3_gate.yaml experiments/yolov8_comparison/yolov8n_mobilenetv2_w0625_realloc_p3_rep.yaml experiments/yolov8_comparison/ABLATION_P3_STRUCTURE.md
git commit -m "feat: add gated P3 fusion and reparameterized block ablations"
git push origin paper/model-improvement-2026
```

上传不是本地训练的前置条件。权重与数据不在本轮提交范围内。
