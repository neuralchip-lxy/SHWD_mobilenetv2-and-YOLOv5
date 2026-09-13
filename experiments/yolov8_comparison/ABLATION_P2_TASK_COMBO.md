# 完整组合版：高分辨率 P2 与分类/定位分路特征融合

## 目的及实验范围

在C（MobileNetV2 W0.625 + 通道重分配）基础上同时改颈部和P2/P3检测头，先验证完整结构，再决定是否补组件消融。已有样本诊断提示低置信度候选与定位不足同时存在，但不证明某个分支已经被确认有缺陷。本方案是对此的可检验设计，尚无精度结果；结构改动幅度不等于原创性，也不保证论文录用。

本轮不沿用G、R或上一版P2下采样拼接到P3的detail路径。复用shallow_detail.py只是为了取得骨干P2输出，不会带入detail版的颈部。

## 改动一：四尺度特征金字塔

骨干宽度、各stage及权重形状保持原样，输出顺序仍为[P3,P4,P5,P2]：原始通道24/64/200/16。1x1适配后P2/P3/P4/P5通道为32/64/96/192。

原C为P5→P4→P3的自顶向下融合，再P3→P4→P5自底向上；新结构延长到P2，并从P2向上重建P3/P4/P5。P2到P3的下采样采用3x3 stride2深度卷积+1x1投影。其他P3→P4、P4→P5下采样仍为原类型普通卷积。

```text
MobileNetV2 P5 → 适配/SPPF ────────────────┐
                    ↓ 上采样             │
MobileNetV2 P4 → 融合C2f96 ──────────┐    │
                    ↓ 上采样        │    │
MobileNetV2 P3 → 融合C2f64 ────┐    │    │
                    ↓ 上采样   │    │    │
MobileNetV2 P2 → 融合C2f32     │    │    │
                    ↓         │    │    │
                P2输出F2      │    │    │
                    ↓ DW下采样+PW │    │
                   拼接 ──────┘    │    │
                    ↓ C2f64        │    │
                P3输出F3           │    │
                    ↓ 下采样+拼接 ─┘    │
                    ↓ C2f96             │
                P4输出F4                │
                    ↓ 下采样+拼接 ──────┘
                    ↓ C2f192
                P5输出F5
```

640输入下，输出网格为160²/80²/40²/20²，步长4/8/16/32。预测位置从8400增至34000；新增位置实际参加原生标签分配和检测，不是仅借用浅层特征。候选增加可能增加NMS/激活开销。

## 改动二：P2/P3分类与回归采用独立的相邻尺度融合

对P2使用(F2,F3)，对P3使用(F3,F4)。每个任务分别学习两套1x1投影：本层投影与相邻更深层投影；较深层先压通道，再最近邻上采样到本层网格；拼接后用1x1混合，再经DW3x3、PW1x1和输出卷积。

对任务t∈{box,cls}及尺度i∈{P2,P3}：

```text
Z_i^t = PW_mix^t(Concat(PW_local^t(F_i), Up(PW_semantic^t(F_{i+1}))))
Output_i^t = Conv_out^t(PW^t(DW3x3^t(Z_i^t)))
```

分类和回归参数互不共享，输出同一网格的一一对应的类别与框分布。它不是置信度后乘门控，也没有额外质量分数分支。

|尺度/任务|本层投影宽度|较深层投影宽度|混合输出宽度|最终输出通道|
|---|---:|---:|---:|---:|
|P2回归|24|8|32|64（4×reg_max16）|
|P2分类|16|32|32|2|
|P3回归|48|16|64|64|
|P3分类|24|48|64|2|

回归给本层路径更多通道；分类给较深层路径更多通道。通道数量是容量分配，不是固定注意力权重，不能说模型必然更依赖某一路。定位也使用语义、分类也使用本层细节。

P4/P5保留C的普通分类和回归塔结构/通道形状（权重均重新初始化）。YOLOv8本来就是解耦头，因此这里的可比较变化是任务分别接收相邻尺度信息、不同通道容量分配及轻量塔，而非“首次分类回归解耦”。

## 损失和训练

保留当前Ultralytics 8.4.89的原生DFL(reg_max=16)、框解码、标签分配、分类/回归损失；四个检测尺度都会进入原生分配流程。源码中新增了forward_head，但返回原生boxes/scores/feats结构。未实现TOOD、DyHead或额外任务对齐损失。

完整继承train_yolov8n_s0.TRAIN_ARGS，仅模型/实验name不同：640、200 epochs、patience50、batch8、seed0、从头训练、SGD、原增强配置。数据划分5457/607/1517不变。正式训练用原workers4；测试/验证参数继承既有VAL_ARGS。

## 资源与检查

|项目|C|完整组合|
|---|---:|---:|
|融合后参数|2,147,134|2,085,464|
|640 GFLOPs|约6.6790|6.5270|
|预测位置|8400|34000|

组合版未融合参数2,099,768。相比C融合参数约下降2.87%，GFLOPs约下降2.28%。增加P2而总成本下降，主要来自P2/P3分类和回归塔的轻量化；并不说明新增P2没有成本。GFLOPs不等于实际速度。

合成检查使用当前GPU、CUDA AMP、batch8/640、每图80个大小混合目标，真实原生损失反传：峰值allocated约3.63GiB，reserved约3.76GiB。未调用optimizer.step，没有训练新权重。此数值不包含完整训练所有开销，不保证任意真实批次峰值相同。

检查项目：四尺度输入和34000输出；矩形输入；P2头只直接使用P2/P3输入；输入不被原地修改；P4/P5塔形状与C一致；各新分类/回归本层和语义投影均有非零有限梯度；原生trainer重建和strict形状状态核对；临时checkpoint保存/恢复；非默认BN统计的融合一致性；原生tensor-only导出路径。未验证ONNX或板端部署。

## PowerShell运行

```powershell
cd "E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUNBUFFERED="1"
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_yolov8n_mobilenetv2_p2_task_s0.py"
```

输出目录：E:/experiment_M2Y5/analysis_runs/yolov8n_mobilenetv2_w0625_p2_task_combo_640_s0_scratch。

可选参数：--check（合成检查）、--resume（last.pt中断续训）、--val（best.pt验证集）、--test（best.pt测试集）。已有目录禁止默认启动覆盖。无参数即完整组合训练，不需要--variant。

必须使用本入口/P2TaskYOLO加载YAML和训练：普通YOLO(YAML)不会执行头替换。新类在独立p2_task_structure.py中定义，trainer在训练重建和续训时保留设计；不修改site-packages或既有实验脚本。

先观察完整方案与C的验证集mAP50-95、各类AP、小目标匹配与误检，再补P2/任务融合/通道与轻量塔消融及多种子。完整方案同时改变多个因素，单次结果不能归因于其中一个组件。

## 文件

- p2_task_structure.py：任务融合、定制检测头、模型、trainer与YOLO入口。
- yolov8n_mobilenetv2_w0625_p2_task_combo.yaml：四尺度颈部与设计标记。
- train_yolov8n_mobilenetv2_p2_task_s0.py：训练/续训/验证/测试入口。
- check_p2_task.py：合成检查。
- 本说明。
