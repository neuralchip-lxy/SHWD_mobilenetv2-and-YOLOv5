# C + 浅层细节补充到 P3

实验假设：验证集误差诊断显示小目标定位值得优先研究。增加从stride-4到P3的短路径，检验保留浅层空间信息能否改善检测。该方案是待验证结构，尚未证明有效或原创。

结构：MobileNetV2 width=0.625，原骨干阶段和参数不变，额外输出stage 1末尾stride-4的16通道特征。经过3x3 stride-2 depthwise Conv-BN-SiLU，再经过1x1 16->16 Conv-BN-SiLU，与96通道上采样语义、64通道P3适配特征拼接。P3 C2f输入160->176、输出64不变；P4/P5为96/192，保持三个检测头。关闭G/R，不改损失。这里浅层通道本来仅16，因此采用保持16通道的投影，并未进一步压缩。

原始骨干文件、C及G/R配置均未修改。新类在本地注册，不修改site-packages。新骨干输出顺序为[P3,P4,P5,P2]，确保原三个Index语义一致。

融合后参数2,148,590，比C的2,147,134增加1456（约0.068%）；640下6.6972 GFLOPs，比C约6.6790增加0.0182（约0.27%）。这些统计不代表真实设备延迟；浅层特征还会增加激活存储和数据访问。

## 运行

在experiments/yolov8_comparison目录，使用原pyenv环境：

```powershell
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_yolov8n_mobilenetv2_detail_s0.py"
```

默认从头训练，完全继承train_yolov8n_s0.TRAIN_ARGS，仅改变name和模型：640、200 epochs、patience50、batch8、seed0、SGD、原数据划分。结果目录E:/experiment_M2Y5/analysis_runs/yolov8n_mobilenetv2_w0625_realloc_p3_detail_640_s0_scratch。

可选参数：--check（合成检查）、--resume（中断续训）、--val（best验证集）、--test（best测试集）。已有结果目录禁止默认启动覆盖。先比较验证集，不据反复测试集结果调参。

检查已通过：同权重骨干原三路输出完全一致，P2和检测输出尺寸正确，额外16通道确实到达P3，CUDA AMP原生损失反传梯度有限且新分支梯度非零，原生trainer重建和权重加载，临时checkpoint保存恢复，非默认BN统计的融合数值一致性。未启动训练或更新优化器。

比较C与本配置：整体mAP50-95、两类AP、固定阈值的小目标匹配/定位不足、资源开销。先做seed0筛选，有合理收益后再多种子验证。整个新增路径是一个消融单元；本轮结果不能单独区分短路径、算子与少量容量增加的贡献。
