# 快速实验：温度异常「预测」基础模型选型

统一口径比较 机器学习 / 深度学习 / 神经网络 / 强化学习，含监督 / 无监督(无标签) /
自监督 / 少样本 / 半监督 / 零样本 / 迁移 / 预训练-微调 / 联邦 等数据设定，
共 **77 个**基础模型方法（编号 00–77，缺 71），选 **test F1 最高** 者。
任务是 **预测事故（12h 早期预警）**，不是检测事故。

> 2026-08-09 校正：本文此前写「47 个」「36 个」两处互相打架的数字，均为早期快照。
> 以 `基础模型/` 目录下的编号文件为准（现为 77 个）。

> ⚠️ **结果解读警告（2026-08-09）**：在 `real_fault_wl` 主口径下，test 段
> `tier1_n = 0` —— 全部 12 个 Tier-1 真过温事件都落在 2016-2021 训练窗，测试段正例
> 100% 是 Tier-2 液压保护跳闸（`Missing gear oil` / `Low gearbox oil pressure`）。
> 用温度残差预测秒级液压事件属目标错配，因此本目录下 77 个模型的近零指标
> **不可表述为「温度早期预警性能」**。详见 `docs/项目问题诊断_2026-08-09.md`
> 与 `tier1_leoo.py`（LOEO 替代协议）。

## 口径（一句话）

三风场 kelmarsh / penmanshiel / hill_of_towie（训 2016-21 / 验 2022 / 测 2023-24），
标签 = 过温事件起点前 72 步（12h）；事件进行期剔除；特征严格因果；
阈值只在 val 上选（最大F1）；test 只评一次。

## 怎么运行（IDE 或命令行）

1. 解释器选 **chuangxin 环境**：`E:\ancoda\chuangxin\python.exe`
2. 环境变量 `FASTEXP_FARM` 选风场（kelmarsh/penmanshiel/hill_of_towie，缺省 kelmarsh），
   打开 `基础模型/` 里任意模型 py，点运行 —— 自动 训练→选阈→测试→打印结果，
   并存到 `快速实验结果/<farm>/<模型名>/metrics.json`
3. 跑 `汇总.py` 出各 farm 排名表 + `快速实验结果/<farm>/汇总.csv` + 跨 farm `全指标汇总.csv`
4. 一键三 farm 全跑：`python 跑三farm全部快速实验.py`（46 在 penmanshiel 跳过：源=目标域退化）

（`准备数据.py --farm <farm>` 只需在数据缺失时每 farm 跑一次，从 v2 预处理产物生成
`快速实验数据/<farm>/`；train 下采样步长按 farm 规模自适应锚定 kelmarsh 历史口径
（kel=4/12、pen=3/9、hot=14/42），val/test 一律全量。）

## 指标（2026-07-18 全指标扩展，见 `基础模型/_common.py`）

- 点级：loss / accuracy / precision / recall / F1 / AUC / AUPRC
- 事件族：range P/R/F1（Tatbul 2018）、affiliation P/R/F1（Huet 2022，A′预注册口径）、
  far_per_day（误报点/机组日）、pa_point_adjust_f1（PA 附录口径，虚高，不进主表）
- 段级（池化轴近似量，跨模型可比）：seg_n_events / seg_n_detected / seg_event_recall /
  seg_event_f1（点级P×段级R 调和，与 事件级评测.py 主口径同形式）/ seg_lead_rows_median

## 模型清单（36 个）

| 类别 | 模型 |
|---|---|
| 参照基线 | 00 残差能量（不训练） |
| 监督机器学习 | 01 逻辑回归 02 感知机 03 朴素贝叶斯 04 LDA 05 QDA 06 K近邻 07 决策树 08 随机森林 09 ExtraTrees 10 AdaBoost 11 直方图梯度提升 12 XGBoost 13 LightGBM 14 CatBoost 15 线性SVM 16 核SVM 17 高斯过程 38 软投票集成 39 Stacking堆叠 |
| 无监督机器学习 | 18 孤立森林 19 一类SVM 20 LOF 21 高斯混合 36 PCA重构误差 37 KMeans距离 40 马氏距离 |
| 自监督(无标签) | 41 自监督预测误差(GRU一步预测) |
| 少样本/半监督(100标签) | 42 逻辑回归 43 梯度提升 44 原型法 45 半监督自训练 |
| 零样本(目标域无标签) | 46 跨场迁移(Pen→Kel, 6维农场无关特征); 00 基线也属零样本物理规则 |
| 监督神经网络 | 22 MLP 23 一维CNN 24 简单RNN 25 LSTM 26 GRU 27 TCN 28 Transformer |
| 无监督神经网络 | 29 自编码器 30 变分自编码器 31 LSTM自编码器 |
| 强化学习 | 32 DQN 33 REINFORCE 34 ActorCritic 35 PPO |

共享代码：`_common.py`（数据/阈值/指标/落盘）、`_torch.py`（深度模型训练循环）。

## 诚实备注

- 快速实验：train 下采样提速，val/test 全量；单 seed；结论用于**模型排序**，
  冠军的确证数字应回全量 train + 5 seed 重跑。
- 无监督模型只用 train 正常样本、不看标签；强化学习中报警不改变风机状态，
  MDP 退化为上下文老虎机（代码内有注释）。
- 标签源自 NBM 残差（轻度循环）；非平凡处在 12h 提前量。
