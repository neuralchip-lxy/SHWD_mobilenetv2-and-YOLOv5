# P2 纯本层轻量检测头对照（2026-09-15）

## 实验问题

在接近完整版的计算预算下，检测头额外读取相邻深层特征是否有价值？

已有普通四尺度头同时改变卷积类型、容量和融合方式，不能单独回答此问题。本对照保留完整版的骨干、颈部、四尺度输出、DW/PW 检测塔和两路投影宽度，仅将检测头第二路输入改成本层特征。不将其称为标准 YOLO 普通头。

## 具体实现

- P2 头原读取 F2/F3，本版两路独立投影均读取 F2。
- P3 头原读取 F3/F4，本版两路独立投影均读取 F3。
- 取消头内相邻深层输入及其上采样。颈部已经进行的跨尺度融合保持不变。
- 保留两路投影再拼接的结构，使模块深度和容量接近完整版。
- P2 定位两路投影为 24/8，分类为 16/32，融合输出 32。
- P3 定位两路投影为 48/16，分类为 24/48，融合输出 64。
- 保留 P4/P5 检测头、原生 DFL、损失函数、框解码与匹配规则。

第二路投影输入通道及空间大小改变，所以这不是参数和 FLOPs 完全相等的严格单因素实验。结果只能在已报告的预算差异下解释额外跨尺度输入的价值，也不能单独证明非对称通道分配的作用。

## 模型大小（融合后，640 输入）

|模型|参数|GFLOPs|
|---|---:|---:|
|P2 普通头|2,307,520|13.5115|
|P2 完整任务分路头|2,085,464|6.5270|
|P2 纯本层轻量头|2,082,136|6.5926|

新对照比完整版参数少约 0.16%，计算量高约 1.0%。第二路投影在更大的本层网格执行，因此少量参数下降不代表 FLOPs 下降。

未融合参数 2,096,440。这里不预估最终精度，也不以 GFLOPs 推断部署速度。

## 训练与检查

脚本：train_yolov8n_mobilenetv2_p2_local_light_s0.py

默认开始从头训练。支持 --check、--resume、--val、--test，互斥。

沿用 TRAIN_ARGS，仅修改实验名称：640、最多200轮、patience50、batch8、workers4、seed0、SGD、AMP、相同增强与原数据划分。正式输出目录：

`E:/experiment_M2Y5/analysis_runs/yolov8n_mobilenetv2_w0625_p2_local_light_640_s0_scratch`

目录已存在时阻止重新训练，防止误覆盖。中断续训使用 --resume，并要求存在有效 last.pt。

必须使用专用 P2LocalLightYOLO 包装类，不能用原生 YOLO 直接构建该 YAML。YAML 的 Detect 层由专用模型类替换，训练器重建和 checkpoint 加载保留新头。checkpoint 依赖本实验 Python 模块，保存权重时同时保留代码。

已检查：YAML骨干/颈部与完整版相同、P4/P5形状、独立扰动各头输入确认只影响本尺度预测、四尺度推理、矩形输入、训练器重建、保存重载、原生损失反传、融合前后数值与tensor导出一致。

CUDA AMP 合成 batch8/640、每图80个标注，峰值 allocated3.65 GiB、reserved3.82 GiB。检查不含优化器更新，不等于完整训练内存上限。没有读取真实测试集或开始正式训练。

正式训练后先记录 best.pt 的验证结果，与两组已有P2实验比较。单种子小差异需重复验证，固定最终方案后再统一测试。

## PowerShell

```powershell
cd "E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUNBUFFERED="1"
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_yolov8n_mobilenetv2_p2_local_light_s0.py"
```

检查命令为同一行末尾加 --check；中断续训加 --resume。
