# P2 + 普通检测头对照

## 对照目的

保留P2完整组合版的全部骨干、适配和四尺度颈部，替换最终检测头，比较完整新检测头与传统C风格检测头的精度/效率。已有完整版保持不变，本轮从头训练，不载入其权重。

本轮“普通头”明确指原C的YOLOv8风格检测塔：每个尺度独立读取本层特征；分类和回归各自两层普通3x3 Conv-BN-SiLU加最终1x1输出卷积，隐藏通道64。P2输入32，P3/P4/P5输入64/96/192；分类输出2通道，回归输出64=4×reg_max16。没有检测头内部的相邻尺度融合、任务不对称通道投影和深度可分离塔。分类与回归仍解耦，这本来就是YOLOv8原有结构。

不能直接使用默认四层Detect的推导宽度：新增P2的32输入会使默认分类隐藏宽度全变为32。本实现显式保留C的64隐藏通道，避免额外缩窄P3/P4/P5普通分类塔。P3/P4/P5塔类型及参数形状与C一致。颈部仍有完整版原先的跨尺度融合，“无融合”仅指没有新增任务分路头内的融合。

YAML的backbone/head连接与完整版逐项相同，仅设计标记不同。使用P2PlainYOLO构造时替换尾部Detect，独立trainer确保训练重建及续训保留普通头。

## 指标和边界

|配置|融合参数|640 GFLOPs|
|---|---:|---:|
|C|2,147,134|约6.6790|
|P2 + 普通头（本轮）|2,307,520|13.5115|
|P2完整新头（已有）|2,085,464|6.5270|

本轮未融合参数2,321,544。普通P2头在160x160特征上使用64通道普通卷积，计算量明显增大是预期结构差异，不代表误用了更大骨干。不要拿它的GFLOPs估计实际训练时间。

本轮比较“头整体设计”，同时改变跨尺度输入、分支通道和卷积类型，并不是等参数或等计算量的纯融合开关消融。若论文要进一步单独声称跨尺度融合有效，需要另做保持轻量塔/通道预算的控制。本轮先回答完整方法相对传统P2结构是否具有精度效率优势。若普通P2精度也高，要据实把新增头贡献表述为效率或精度效率取舍，而不是预设它一定贡献精度。

训练沿用原640、200 epochs、patience50、batch8、seed0、SGD、从头训练、原增强/损失及5457/607/1517数据划分。仅模型和实验名称/输出目录变化。先分析验证集；当前不重复测试集调参。

## 检查

已通过：与完整版颈部YAML/模块类型/参数形状一致；P3/P4/P5普通塔形状与C一致；头只直接读取本层输入；640及矩形输入，四尺度输出34000位置；trainer及临时checkpoint恢复；640 batch8、每图80个混合尺寸目标的CUDA AMP原生损失反传（无optimizer.step）；非默认BN统计融合数值一致。合成峰值allocated约3.57GiB、reserved约3.65GiB，不代表完整实际训练最高开销。未开始训练。

## PowerShell

```powershell
cd "E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUNBUFFERED="1"
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_yolov8n_mobilenetv2_p2_plain_s0.py"
```

结果：E:/experiment_M2Y5/analysis_runs/yolov8n_mobilenetv2_w0625_p2_plain_640_s0_scratch。

参数：--check 合成检查；--resume 从本轮last.pt继续中断训练；--val 对本轮best.pt评估验证集；--test 最终测试集评估。无参数是本轮从头训练。已存在输出目录会阻止意外覆盖。

新增文件：p2_plain_structure.py、yolov8n_mobilenetv2_w0625_p2_plain.yaml、train_yolov8n_mobilenetv2_p2_plain_s0.py、check_p2_plain.py、本说明。依赖原有backbone/shallow_detail/公共训练设置。--check还引用完整版定义进行结构对比，正式训练不读取完整版权重。
