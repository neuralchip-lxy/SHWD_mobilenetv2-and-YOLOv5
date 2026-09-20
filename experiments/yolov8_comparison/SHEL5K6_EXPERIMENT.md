# SHEL5K 六类独立训练实验

## 数据来源与处理

来源：https://data.mendeley.com/datasets/9rcv8mm682/4 ，SHEL5K V4，CC BY 4.0。论文应引用数据集及 SHEL5K 原论文。

下载包 SHA256：dfba1d3ce01af69d791020cdfdfdbc25904b41724d11160361e7a4cd164e7a7a，与官网一致。

原始目录：E:/datasets/SHEL5K_original/9rcv8mm682-4/Safety Helmet Wearing Dataset

原始 5000 PNG + 5000 XML，共 75578 个框。官方包中另残留 8 个 person 标签，分布在 hard_hat_workers1016、1303、1571、1595、380 这 5 张图中。因无法仅凭类别名确定佩戴状态，本实验排除这 5 张整图，不猜测重标、不只删除框，原始文件保持不变。清洗后 4995 图、75482 个目标。

固定映射：0 helmet；1 head_with_helmet；2 person_with_helmet；3 head；4 person_no_helmet；5 face。

转换后的图片和标签：E:/experiment_M2Y5/datasets/SHEL5K6_YOLO_v1

| 划分 | 图片 |
|---|---:|
| train | 3596 |
| val | 400 |
| test | 999 |

这是自定义 72/8/20 划分，不是作者官方 80/20 划分，不能直接与使用不同划分的文献数字排名。固定划分随机种子为 20260920；改变训练种子不会改变数据划分。

使用解码后 RGB 像素 SHA256 与 63 位 DCT pHash（汉明距离 <=4）审计图片。未检出与本地 SHWD 的完全相同图片或达到阈值的近似候选，这不是没有任何重叠的绝对保证。数据内部有 98 对近似候选；相似候选连通组整体划分，清洗后共 4902 组，最大组 4 图。没有组跨越训练/验证/测试集。pHash 是启发式筛查，不能保证找全裁剪、旋转和连拍场景。

XML/image 尺寸一致、框坐标有效。按原 XML 边界坐标归一化，不任意减一个像素；转换实现有边界裁剪和同类重复框处理，本次两者均未触发。每份数据都覆盖全部六类。审计清单、排除清单、XML 哈希、划分清单和汇总保存在数据目录中。

## 实验范围

比较原始 YOLOv8n 和已验证的 P2 task combo 完整版。两者均从头训练六类任务；不加载 SHWD 权重，不加入新 P5 Transformer。这验证结构在第二数据集上的有效性，不是冻结 SHWD 权重的跨数据集直接测试。

除 data/project/name/seed 外，沿用 train_yolov8n_s0.TRAIN_ARGS：640，batch8，epochs200，patience50，SGD，AMP，scratch。两种模型自动按 nc=6 构建输出；六类模型参数量需重新统计，不能直接套用两类模型的参数量。

## PowerShell

先进入项目目录：

```powershell
cd "E:\codex_work\experiment of mobilenet2 and YOLO5\yolov5-paper\experiments\yolov8_comparison"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUNBUFFERED="1"
```

原始模型 seed0：

```powershell
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_shel5k6.py" --variant baseline --seed 0
```

完整模型 seed0（等前一任务结束再运行）：

```powershell
& "C:\Users\liuji\.conda\envs\pyenv\python.exe" ".\train_shel5k6.py" --variant combo --seed 0
```

后续种子将 0 改为 1 或 2。所有重复实验使用相同划分。每次结果独立保存：

E:/experiment_M2Y5/analysis_runs/shel5k6/shel5k6_{baseline|combo}_640_s{0|1|2}_scratch

配置预检查加 --dry-run；CPU 合成检查加 --check；中断续训加 --resume；独立验证 best.pt 加 --val；固定方案后最终测试加 --test。测试集不能用于调结构和选择超参数。测试使用 batch4/640/FP32/conf0.001/iou0.6/max_det300。

本次已完成：全部图片/标签配对与框合法性检查、类别计数核对、相似组无跨集、两种模型六类前向与损失梯度、完整版训练器重建。检查仅用 CPU 合成数据，无训练或测试集评估。

## 复现与迁移

数据集与权重不提交 Git。提交 audit_shel5k.py、prepare_shel5k_yolo.py、train_shel5k6.py、yolov8n_mobilenetv2_w0625_p2_task_shel5k6.yaml 和本说明。

另一台电脑可用 audit_shel5k.py --source 指定含 Annotations/Images 的原始目录，--shwd-images 指定 SHWD 图片根，--output 指定审计输出；然后 prepare_shel5k_yolo.py --audit-dir 指定审计目录 --output 指定新数据目录。转换脚本拒绝覆盖现有目录。训练支持 --data 和 --project 指定迁移后的路径。若直接复制转换后的数据，需更新 shel5k.yaml 中的 path，同时保留 split_manifest.json 和 preparation_report.json。

原 SHWD 训练入口、模型 YAML、数据划分和既有权重未修改。
