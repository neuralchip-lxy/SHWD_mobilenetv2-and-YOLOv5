# P2 随机种子重复实验

比较完整跨尺度任务头 combo 与纯本层轻量头 local_light。seed0 已完成，补 seed1/2，共四次新训练。随机种子是重复实验条件，不是新结构模块。

统一入口 train_p2_seed_repeat.py 导入现有模型路径和专用 YOLO 类，仅覆盖 TRAIN_ARGS 的 seed/name，保留原有骨干、颈部、检测头、数据划分、增强、SGD、640、batch8、最多200轮和patience50。从头训练，不能将seed0权重用于seed1/2续训。相同seed不意味着不同架构逐个随机数和初始化权重完全相同。

命令：

```powershell
cd "E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUNBUFFERED="1"
foreach ($trialSeed in 1,2) {
    foreach ($trialVariant in "combo","local_light") {
        & "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_p2_seed_repeat.py" --variant $trialVariant --seed $trialSeed
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $trialVariant seed=$trialSeed. Queue stopped." }
    }
}
```

顺序为combo s1、local_light s1、combo s2、local_light s2，串行占用GPU。不要重复启动该队列。中断后对当前一组使用相同variant/seed加 --resume，完成后只启动剩余组，避免整个队列从头执行。

正式目录为 E:/experiment_M2Y5/analysis_runs/yolov8n_mobilenetv2_w0625_p2_{task_combo或local_light}_640_s{seed}_scratch。每组独立目录，已有目录阻止重跑。

--dry-run 只检查版本、模型/数据配置文件存在并显示目录，不加载模型训练，也不完整扫描数据集。seed0保留原来的目录名称，与原脚本兼容。

最终收集两种结构各seed0/1/2的best.pt验证指标，报告mAP50、mAP50-95的均值和样本标准差（ddof=1），并列出每个种子及同种子差值。P/R及各类AP可附表。不能只挑表现最佳的种子，也不能把训练末段多个epoch当独立重复。保留现有early stopping和best checkpoint选择规则。三次重复提供初步稳定性证据，不能自动宣称统计显著。

暂不重复评估测试集。结构确定后统一测试。当前队列不包含普通P2头或三尺度C，因此仅验证跨尺度与纯本层两种轻量头的相对表现，不能据此宣称其相对所有基线的多种子优势。
