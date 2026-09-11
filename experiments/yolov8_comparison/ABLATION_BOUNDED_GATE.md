# 本轮固定方案：P3 门控幅度限制对比

本轮以 MobileNetV2 W0.625 + 通道重分配版 C 为基础，P3/P4/P5=64/96/192。
只使用门控 G，不加入 R。推荐的是可检验的候选方案，并非已经验证的最优模型。

## 明确比较对象

| 配置 | 融合方式 | 推理参数 | GFLOPs@640 |
| --- | --- | ---: | ---: |
| C，已完成 | Concat(U,L)，不门控 | 2,147,134 | 6.6790 |
| G，现有普通门控 | Concat(2*sigmoid(z)*U,L) | 2,149,887 | 6.7138 |
| 本轮 gate_bounded | Concat((1+0.5*tanh(z))*U,L) | 2,149,887 | 6.7138 |

U 为上采样语义特征，96通道；L 为浅层侧路，64通道。
z 仍由同一网络根据 Concat(U,L) 生成：1×1 Conv(160→16)、DWConv3×3、1×1 Conv(16→1)。
隐藏宽度、初始化、作用位置、被调制的分支、原C2f、骨干、检测头都不变。
本轮只是改变门控映射函数；范围从0～2变为0.5～1.5，同时也改变远离原点时的饱和特性，不能把实验解释为与函数形状完全无关的“纯范围变化”。

两种映射在z=0时均为1，对z的导数均为0.5；最后卷积零初始化保持这一条件。
本轮不是照搬SGDF，不引入其双路径混合结构。参考其“防止过度抑制”的动机：[SGDF论文第3.3节](https://link.springer.com/article/10.1007/s43684-026-00141-4)。
限制门控乘法的衰减幅度，不代表保证小目标信息不会丢失，是否改善检测精度需要实际验证。

GFLOPs 使用与前轮一致的 Ultralytics/THOP 估计，逐元素运算可能未完整统计；tanh 与 sigmoid 实测时延也不一定相同。不能据两组GFLOPs相同认定实际速度相同。

## PyCharm 本轮配置

| 字段 | 填写内容 |
| --- | --- |
| 名称 | paper_yolov8n_mobilenetv2_gate_bounded_s0 |
| 运行方式 | script |
| Python 解释器 | C:\Users\liuji\.conda\envs\pyenv\python.exe |
| 脚本路径 | E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison\train_yolov8n_mobilenetv2_p3_s0.py |
| 脚本参数 | --variant gate_bounded |
| 工作目录 | E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison |
| 环境变量 | PYTHONUNBUFFERED=1;PYTHONIOENCODING=utf-8 |

仍是640、batch8、epochs200、patience50、seed0、SGD、AMP=True、从头训练。
数据5457/607/1517，hat/person两类别。不加载旧best.pt微调，不修改损失或增强。

启动应显示：

```text
Variant: gate_bounded
P3 design: gate=True, rep=False, gate_mode=bounded; channels=64/96/192
```

结果目录：

```text
E:\experiment_M2Y5\analysis_runs\yolov8n_mobilenetv2_w0625_realloc_p3_gate_bounded_640_s0_scratch
```

| 用途 | 同一脚本的参数 |
| --- | --- |
| 本轮训练 | --variant gate_bounded |
| 本轮检查，不训练 | --variant gate_bounded --check |
| 中断后恢复本轮 | --variant gate_bounded --resume |
| 本轮完成后测试 | --variant gate_bounded --test |
| 普通G训练对照 | --variant gate |

不要用 --variant combo 与本轮直接比较后把差异全部归因于门控范围，因为combo还有R。原G若已训练完成可以复用；若未完成，需要补这一组，C无需重跑。

## 结果怎么判断

先比较三组验证集最佳mAP50-95、mAP50、类别指标和训练曲线。C的训练记录最佳mAP50-95=0.59891；组合G+R为0.59920，但组合不是本轮幅度对照。
如果gate_bounded仅优于普通G却仍不如C，不能说总体改进有效。
若gate_bounded优于C和普通G，也只能先判断单次实验有利，之后需要补关键配置多个种子。
同时可以在相同验证图像上记录门控权重分布，检查普通G是否发生极端抑制；不要预先把它写成已证实的问题。连续轮次均值不能代替多种子实验。
新颖性表述应落在可验证的设计取舍上；更换激活函数或范围本身不足以保证创新成立。

## 实现与检查

新增 BoundedGatedP3Concat 子类，普通 G 的映射保持原样。
YAML 新增 gate_mode: bounded，保存在 checkpoint 中；缺省 gate_mode 仍使用旧 standard 模式，兼容旧G/组合/R配置。
训练入口增加 gate_bounded 选项，恢复和测试按所选YAML核对设计标记。
已检查权重边界、初始值与梯度、原始门控回归行为、浅层侧路不变、640前向、CPU/CUDA AMP损失反向、训练器重建、checkpoint保存重载与融合等价。
未启动正式训练，未执行优化器更新，未提交或推送Git。

## 手动 Git 上传

在 yolov5-paper 仓库根目录执行：

```bash
git add -- experiments/yolov8_comparison/p3_structure.py experiments/yolov8_comparison/check_p3_structure.py experiments/yolov8_comparison/train_yolov8n_mobilenetv2_p3_s0.py experiments/yolov8_comparison/yolov8n_mobilenetv2_w0625_realloc_p3_gate_bounded.yaml experiments/yolov8_comparison/ABLATION_BOUNDED_GATE.md
git commit -m "feat: add bounded P3 gate comparison"
git push origin paper/model-improvement-2026
```

本地训练不依赖Git上传。
