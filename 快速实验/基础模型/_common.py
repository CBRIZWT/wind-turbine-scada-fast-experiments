# -*- coding: utf-8 -*-
"""_common.py — 全部模型共用: 数据加载 / 选阈值 / 算指标 / 存结果 (保证口径一致可比)

口径(预测非检测): y=1 表示未来72步(12h)内将发生过温事件的起始; 事件进行期已剔除。
防泄漏: 时序切分(训2016-21/验2022/测2023-24); 阈值只在val上选(最大F1); test只评一次。

多 farm (2026-07-18): 环境变量 FASTEXP_FARM ∈ {kelmarsh, penmanshiel, hill_of_towie}
  (默认 kelmarsh) 选择 快速实验数据/<farm>/ 与 快速实验结果/<farm>/。

全指标 (2026-07-18, 与项目逐模型指标清单对齐):
  点级:   loss(=逐点二元NLL) / accuracy / precision / recall / f1 / auc / auprc
  回归/概率族 (归一分数p vs 标签y, Brier式转译): mse(=Brier) / mae / rmse / r2 / nrmse
          —— 对应 TriTrackNet/wt-transformer 的 mse/mae/rmse 与复现论文的 R²/NRMSE 在
          二分类报警分数上的可计算等价物 (温度回归原义需预测型输出, 见诚实注记)。
  事件族: range P/R/F1 (Tatbul 2018) / affiliation P/R/F1 (Huet 2022, A′预注册主口径)
          / far_per_day (误报点/机组日) / pa_point_adjust_f1 (PA附录, 不作主表)
  段级:   seg_n_events / seg_n_detected / seg_event_recall / seg_event_precision / seg_event_f1
          (2026-07-19 改口径: 『命中一段即命中』段级 P & R 双侧检出F1) / seg_lead_rows_median
诚实注记: 行轴为池化多机组按时间交错 (与 v2 战役 4 深度模型消费方式一致), 标签连续段
  是真实早警窗被交错切出的碎片 —— 段级/事件族指标在此轴上是跨模型可比的近似量,
  不是逐机组真实事件数; 真实事件级口径见 事件级评测.py (带 timestamps/turbines)。
不可诚实计算的项目指标 (NA_METRICS, 逐条落盘声明, 不伪造):
  HitRate@k / NDCG@k (需逐通道根因分数, 快速模型只输出标量分), CP95/PICP (需概率区间输出),
  温度回归 MAE°C/NRMSE% (需温度预测输出, 属 NBM/预测型基线), HI(t)曲线/RUL/MK单调性
  (需 run-to-failure 轨迹与逐机组时间轴), UHH (需舰队多机组归一), CARE 四分 (需 CARE 协议)。
"""
import json                      # 落盘 metrics.json 用
import os                        # 读 FASTEXP_FARM 环境变量
import sys                       # 用于重配置标准输出编码 / 导入项目根模块
import time                      # 计时(perf_counter)
from pathlib import Path         # 跨平台路径拼接

import numpy as np               # 数值计算/数组
from sklearn.metrics import (accuracy_score, average_precision_score,             # 评测指标族
                             balanced_accuracy_score, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

if hasattr(sys.stdout, "reconfigure"):                       # 若运行环境支持重配置标准输出
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 强制UTF-8, cmd中文不乱码不报错

_ROOT = Path(__file__).resolve().parent.parent               # 本文件在 基础模型/, 上一级=快速实验/
if str(_ROOT.parent) not in sys.path:                        # 项目根 (E:\创新) 进 sys.path
    sys.path.insert(0, str(_ROOT.parent))
from 事件指标 import (compute_affiliation_metrics, compute_range_metrics,  # noqa: E402  事件族指标(向量化)
                      extract_events, false_alarms_per_day, point_adjust_f1)
from 事件级评测 import alarm_segments  # noqa: E402  Hill无标签报警负担
from 统一评测 import evaluate_all, select_operating_points  # noqa: E402  metrics-v3唯一评测入口

FARM = os.environ.get("FASTEXP_FARM", "kelmarsh")            # 当前farm (环境变量注入, 默认kelmarsh)
VARIANT = os.environ.get("FASTEXP_VARIANT", "").strip()       # 非空variant物理隔离真实故障快速数据
RESULT_SET = os.environ.get("FASTEXP_RESULT_SET", "快速实验结果").strip()
PROTOCOL = os.environ.get("FASTEXP_PROTOCOL", "").strip()
_DATA_KEY = f"{FARM}__{VARIANT}" if VARIANT else FARM
DATA = _ROOT / "快速实验数据" / _DATA_KEY
RESULT = _ROOT / RESULT_SET / FARM / PROTOCOL if PROTOCOL else _ROOT / RESULT_SET / FARM
now = time.perf_counter                                      # 高精度计时器别名; 用法: t0=now()...now()-t0

# 项目指标清单中对标量报警分数不可诚实计算的指标 → 逐条声明落盘, 与 metrics-v3 的
# metric_status=None 哲学一致 (不把不适用指标伪造成 0), 见模块 docstring。
NA_METRICS = {
    "hitrate_at_k": "需逐通道根因分数 (TranAD诊断路线); 快速模型仅输出标量报警分",
    "ndcg_at_k": "同上",
    "cp95_picp": "需概率区间/分位输出 (PMLP路线); 逐点二元NLL已由 loss 覆盖",
    "temp_mae_degc": "温度回归原义需温度预测输出 (NBM/FL-LSTM路线); 本任务为二分类报警",
    "temp_nrmse_pct": "同上; 分数-标签口径的 nrmse 已单列",
    "hi_prognostic_lead_h": "HI(t)/预后提前小时需逐机组时间轴; 经典A′池化轴无侧车, 段级lead_rows已近似",
    "rul_mk_monotonicity": "RUL/MK单调性需 run-to-failure 轨迹 (cGAN仿真路线)",
    "uhh": "需舰队多机组归一 (舰队AE路线)",
    "care_score": "需 CARE 协议 (Coverage/Accuracy/Reliability/Earliness) 多数据集设定",
    "true_event_family": "真实事件级 (event_recall/lead_minutes/FAR段级) 需 timestamps/turbines 侧车, 见 metrics-v3 真实故障管线",
}

PROBABILITY_SCORE_MODELS = {
    "01_逻辑回归", "03_朴素贝叶斯", "04_LDA", "05_QDA", "06_K近邻",
    "07_决策树", "08_随机森林", "09_ExtraTrees", "10_AdaBoost",
    "11_直方图梯度提升", "12_XGBoost", "13_LightGBM", "14_CatBoost",
    "17_高斯过程", "33_REINFORCE", "34_ActorCritic", "35_PPO",
    "38_软投票集成", "39_Stacking堆叠", "42_少样本_逻辑回归",
    "43_少样本_梯度提升", "45_少样本_半监督自训练", "46_零样本_跨场迁移",
}


def fit_regularized_qda(X, y):
    """以收缩协方差的eigen求解器拟合QDA，支持正例数小于特征数。

    [修复 2026-07-26] 补 priors=[0.5,0.5]: 原实现用数据经验先验, 在 0.13% 正例率下
    后验几乎恒判负类, 严重低估 QDA。等先验与项目其余监督模型的 balanced 口径一致
    (阈值仍由下游 val 选, 故此处只是去掉极端先验的偏置)。
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    return QuadraticDiscriminantAnalysis(
        solver="eigen", shrinkage="auto", reg_param=0.1, priors=[0.5, 0.5]
    ).fit(X, y)


def pca_component_count(n_samples, n_features, requested=16):
    """返回PCA在当前协议维度下可辨识的主成分数。"""
    return max(1, min(int(requested), int(n_samples), int(n_features)))


def cross_farm_source(target_farm):
    """模型46的双真风场互为源域映射。"""
    mapping = {"kelmarsh": "penmanshiel", "penmanshiel": "kelmarsh"}
    if target_farm not in mapping:
        raise ValueError(f"模型46仅支持双真风场目标，收到: {target_farm}")
    return mapping[target_farm]


def quick_data_dir(farm, variant=None):
    """返回指定farm/variant的快速数据目录。"""
    variant = VARIANT if variant is None else str(variant).strip()
    key = f"{farm}__{variant}" if variant else str(farm)
    return _ROOT / "快速实验数据" / key


def needs_external_train_scores():
    """Hill外部无标签协议是否需要训练分数分位阈值。"""
    return FARM == "hill_of_towie" and "external" in VARIANT


def load_flat():
    """扁平特征 (ML/MLP用): 返回 Xtr,ytr, Xva,yva, Xte,yte —— X:(N,D+6), y:0/1 (D=farm通道数)"""
    f = lambda n: np.load(DATA / n)                          # 小工具: 按文件名加载 .npy
    return (f("X_flat_train.npy"), f("y_flat_train.npy"), f("X_flat_val.npy"),   # 训练特征/标签, 验证特征
            f("y_flat_val.npy"), f("X_flat_test.npy"), f("y_flat_test.npy"))     # 验证标签, 测试特征/标签


def load_seq():
    """序列窗口 (CNN/RNN等用): 每个split返回 (基座数组(T,D), 窗口末端索引, 标签0/1)。
    61万测试窗一次物化要7.8GB, 所以只存索引、用 take_windows 按批取窗。"""
    d = {s: (np.load(DATA / f"X_base_{s}.npy", mmap_mode="r"),   # 基座序列(内存映射, 不全load进内存)
             np.load(DATA / f"idx_seq_{s}.npy"), np.load(DATA / f"y_seq_{s}.npy"))  # 窗口末端索引, 窗口标签
         for s in ("train", "val", "test")}                     # 对三个split各建一份
    W = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))["W_seq"]  # 从meta读窗宽(=36)
    return d, W                                                 # 返回 {split:(base,idx,y)} 和窗宽W


def take_windows(base, idx, W):
    """按窗口末端索引取一批窗口 → (B, W, D) float32 (零拷贝视图+花式索引, 只物化这一批)"""
    v = np.lib.stride_tricks.sliding_window_view(base, W, axis=0)  # 沿时间轴构造滑窗视图(不复制内存)
    return np.ascontiguousarray(v[np.asarray(idx) - W + 1].swapaxes(1, 2)).astype(np.float32)
    # idx是窗口末端 → 起点=idx-W+1; 取出后把(通道,时间)轴换回(时间,通道), 连续化并转float32


def standardize(Xtr, *rest):
    """标准化: 均值/方差只用train算 (防泄漏); 给尺度敏感模型(线性/KNN/SVM/神经网络)用"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8                   # 仅在train上求每列均值与标准差(+1e-8防除零)
    return tuple(((X - mu) / sd).astype(np.float32) for X in (Xtr, *rest))  # 用同一组mu/sd变换所有split


def _seg_metrics(y, pred):
    """段级事件检出指标 (2026-07-19 改口径: 『只要命中一段就相当于命中』—— 段级 P & R 双侧)。

    - 召回侧: 真实事件段内有 ≥1 个报警点 → 该事件检出 (命中一段即算命中); R = 检出段/事件段。
    - 精确侧: 报警段与 ≥1 真实事件段相交 → 该报警段算真命中 (命中一段即算命中);
              P = 命中报警段/报警段总数。 (旧口径为逐点精确率, 现按用户指示改为段级。)
    - seg_event_f1 = 2·P_seg·R_seg/(P_seg+R_seg); lead_rows = 段末−段内首报警行+1, 报检出段中位数。
    诚实注记: 行轴为池化多机组按时间交错, 段为近似量; 段级精确率对"常开报警"会退化为高值,
      但 val 阈值取"最大逐点F1"已抑制常开; 该口径与 事件指标.affiliation_f1 (存在性检出) 同族。"""
    y_arr = np.asarray(y).astype(int)
    pred_arr = np.asarray(pred).astype(int)
    true_segs = extract_events(y_arr)                        # 标签连续正例段(闭区间)=真实事件段
    if not true_segs:                                        # 无正例段(理论上不发生)
        return {"seg_n_events": 0, "seg_n_detected": 0, "seg_event_recall": float("nan"),
                "seg_event_precision": float("nan"), "seg_event_f1": float("nan"),
                "seg_n_alarm_segments": 0, "seg_n_alarm_hit": 0,
                "seg_lead_rows_median": float("nan")}
    arr = np.asarray(true_segs, dtype=np.int64)              # (n,2) starts/ends
    s, e = arr[:, 0], arr[:, 1]
    pos = np.flatnonzero(pred_arr == 1)                      # 全部报警行索引(升序)
    # ---- 召回侧: 每个真实事件段内是否有 ≥1 报警点 (命中一段即检出) ----
    k = np.searchsorted(pos, s, side="left")                 # 每段内第一个≥start的报警位置
    k = np.minimum(k, max(len(pos) - 1, 0))
    hit = (len(pos) > 0) & (pos[k] >= s) & (pos[k] <= e) if len(pos) else np.zeros(len(s), bool)
    n_det = int(hit.sum())
    rec = n_det / len(s)                                     # 段级召回
    # ---- 精确侧: 每个报警段是否与 ≥1 真实事件段相交 (命中一段即算真命中) ----
    pred_segs = extract_events(pred_arr)                     # 报警连续段
    if pred_segs:
        cy = np.concatenate([[0], np.cumsum(y_arr == 1).astype(np.int64)])  # 真实正例前缀和
        ps = np.asarray(pred_segs, dtype=np.int64)
        overlap_true = cy[ps[:, 1] + 1] - cy[ps[:, 0]]      # 每报警段内真实正例点数
        n_alarm = int(len(pred_segs))
        n_alarm_hit = int((overlap_true > 0).sum())         # 与真实事件相交的报警段数
        prec = n_alarm_hit / n_alarm
    else:
        n_alarm = n_alarm_hit = 0
        prec = 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)  # 段级P×段级R调和
    leads = (e[hit] - pos[k[hit]] + 1) if n_det else np.empty(0, dtype=np.int64)     # 检出段提前行数
    return {"seg_n_events": int(len(s)), "seg_n_detected": n_det,
            "seg_event_recall": float(rec), "seg_event_precision": float(prec),
            "seg_event_f1": float(f1),
            "seg_n_alarm_segments": n_alarm, "seg_n_alarm_hit": n_alarm_hit,
            "seg_lead_rows_median": float(np.median(leads)) if n_det else float("nan")}


def _metrics(y, s, thr):
    """按阈值算全套指标; 非[0,1]的分数先归一再算log loss"""
    pred = (s >= thr).astype(int)                            # 分数≥阈值判为正类(报警)
    p = s if (s.min() >= 0 and s.max() <= 1) else (s - s.min()) / (np.ptp(s) + 1e-12)  # log_loss需概率, 非[0,1]先归一
    prec = float(precision_score(y, pred, zero_division=0))  # 点级精确率(段级F1也要用)
    y_arr = np.asarray(y).astype(int)
    tp = int(((pred == 1) & (y_arr == 1)).sum()); fp = int(((pred == 1) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum()); tn = int(((pred == 0) & (y_arr == 0)).sum())
    # 回归/概率族 (Brier式转译, 归一分数p vs 标签y): mse=Brier分数; r2=1−mse/var(y);
    #   nrmse=rmse/std(y)。对应 TriTrackNet/wt-transformer mse/mae/rmse 与复现论文
    #   R²/NRMSE 在二分类报警分数上的可计算等价物 (温度回归原义见 NA_METRICS)。
    err = p.astype(float) - y_arr
    mse = float(np.mean(err ** 2)); rmse = float(np.sqrt(mse))
    var_y = float(np.var(y_arr))
    out = {"loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),    # 对数损失(裁剪防log0)
           "accuracy": float(accuracy_score(y, pred)),       # 准确率
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),   # 平衡准确率(治基率)
           "precision": prec,                                # 精确率(报警中真为正的比例)
           "recall": float(recall_score(y, pred, zero_division=0)),        # 召回率(真正例中被报警的比例)
           "f1": float(f1_score(y, pred, zero_division=0)),                # F1=精确率与召回率的调和均值
           "mcc": float(matthews_corrcoef(y, pred)),         # Matthews相关系数(四格全用)
           "auc": float(roc_auc_score(y, s)),                # ROC曲线下面积(阈值无关判别力)
           "auprc": float(average_precision_score(y, s)),    # PR曲线下面积(不平衡下更敏感)
           "tp": tp, "tn": tn, "fp": fp, "fn": fn,           # 混淆矩阵计数
           "valid_count": int(len(y_arr)),                   # 参评行数(ignore已在数据侧剔除)
           "mse": mse, "mae": float(np.mean(np.abs(err))), "rmse": rmse,   # Brier回归族
           "r2": float(1.0 - mse / var_y) if var_y > 0 else float("nan"),
           "nrmse": float(rmse / np.sqrt(var_y)) if var_y > 0 else float("nan"),
           "threshold": float(thr)}                          # 记录所用阈值
    out.update(compute_range_metrics(y, pred))               # Tatbul 2018 range P/R/F1
    out.update(compute_affiliation_metrics(y, pred))         # Huet 2022 affiliation P/R/F1 (A′预注册口径)
    out["far_per_day"] = float(false_alarms_per_day(y, pred))            # 误报点/机组日(10min行距)
    out["pa_point_adjust_f1"] = float(point_adjust_f1(y, pred))          # PA附录口径(虚高, 不进主表)
    out.update(_seg_metrics(y, pred))                        # 段级事件检出: 命中一段即命中(段级P&R)
    # ---- 2026-08-09 十项指标补齐 (Accuracy/AUC/Recall/F1/R2/MAE/RMSE/Precision 已在上方 out 里) ----
    # MAPE: y∈{0,1} 时 y=0 行的相对误差发散(除零), 按标准做法只在 y≠0 (真正例) 上算,
    #   等价于 mean(|1-p|) —— 真正例上的分数亏欠; y=0 行的误差由 Brier 族 mse/mae 覆盖。
    pos = (y_arr == 1)
    out["mape_on_positives"] = float(np.mean(np.abs(1.0 - p[pos]))) if pos.any() else float("nan")
    out["mape_n_positives"] = int(pos.sum())
    # LeadTime: 段级检出提前行数 → 小时 (行距 10min)。
    _lr = out.get("seg_lead_rows_median", float("nan"))
    out["lead_time_hours_median"] = float(_lr * 10.0 / 60.0) if np.isfinite(_lr) else float("nan")
    # 判别力提升倍数 = AUPRC / 基率。基率即随机分类器的 AUPRC 期望, 故该比值剥离了基率影响,
    #   是极端不平衡下排序"模型对异常的判别力"的正确依据 (F1/精确率在 0.1% 基率下有结构性上限)。
    _base = float(pos.sum()) / max(len(y_arr), 1)
    out["base_rate"] = _base
    out["auprc_lift"] = float(out["auprc"] / _base) if _base > 0 else float("nan")
    return out


def report(name, yva, sva, yte, ste, elapsed, extra=None, train_scores=None):
    """统一收尾(每个模型文件最后调这一句): val扫250个分位取最大F1阈值 → 存指标 → 打印"""
    extra = dict(extra or {})
    if name in PROBABILITY_SCORE_MODELS:
        extra["scores_are_probabilities"] = True
    if VARIANT:
        representation = _infer_representation(len(yva), len(yte))
        return report_v3(name, yva, sva, yte, ste, elapsed, extra=extra,
                         representation=representation, train_scores=train_scores)
    sva, ste = np.asarray(sva, float), np.asarray(ste, float)     # 保证分数是float数组
    qs = np.append(np.linspace(0.50, 0.999, 250), 0.0)           # 候选阈值=val分数的250个分位+0分位
    thr = max((float(np.quantile(sva, q)) for q in qs),         # 在val上遍历候选阈值
              key=lambda t: f1_score(yva, (sva >= t).astype(int), zero_division=0))  # 取val F1最大者
    mv, mt = _metrics(yva, sva, thr), _metrics(yte, ste, thr)    # 用同一阈值算val与test指标(test只评这一次)
    out = RESULT / name                                         # 该模型的结果子目录
    out.mkdir(parents=True, exist_ok=True)                      # 不存在则创建
    d = {"model": name, "farm": FARM, "elapsed_sec": round(elapsed, 1), "val": mv, "test": mt,
         "na_metrics": NA_METRICS}                             # 不适用指标逐条声明(不伪造)
    if extra:                                                   # 若有额外信息(范式/标签数等)
        d["extra"] = extra                                     # 一并记录
    (out / "metrics.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")  # 落盘
    print(f"[{FARM}/{name}] test F1={mt['f1']:.4f}  AUC={mt['auc']:.4f}  AUPRC={mt['auprc']:.4f}  "
          f"affF1={mt['affiliation_f1']:.4f}\n"
          f"  精确率={mt['precision']:.4f}  召回率={mt['recall']:.4f}  准确率={mt['accuracy']:.4f}  "
          f"rangeF1={mt['range_f1']:.4f}  segEvtF1={mt['seg_event_f1']:.4f}  "
          f"FAR/日={mt['far_per_day']:.2f}  PA-F1(附录)={mt['pa_point_adjust_f1']:.4f}\n"
          f"  loss={mt['loss']:.3f}  耗时={elapsed:.1f}s  → {out / 'metrics.json'}")


def _infer_representation(n_val, n_test):
    """按侧车长度判定模型消费的是flat还是seq表示。"""
    matches = []
    for rep in ("flat", "seq"):
        vp = DATA / f"timestamps_{rep}_val.npy"
        tp = DATA / f"timestamps_{rep}_test.npy"
        if vp.exists() and tp.exists():
            if len(np.load(vp, mmap_mode="r")) == int(n_val) and len(np.load(tp, mmap_mode="r")) == int(n_test):
                matches.append(rep)
    if len(matches) != 1:
        raise ValueError(f"无法唯一判定评测表示: n_val={n_val}, n_test={n_test}, matches={matches}")
    return matches[0]


def _eval_sidecars(split, representation):
    """读取与模型分数逐行对齐的时间/机组侧车。"""
    ts = np.load(DATA / f"timestamps_{representation}_{split}.npy")
    turb = np.load(DATA / f"turbines_{representation}_{split}.npy")
    return np.asarray(ts, dtype=np.int64), np.asarray(turb).astype(str)


def report_v3(name, yva, sva, yte, ste, elapsed, extra=None, *, representation="flat",
              val_sidecars=None, test_sidecars=None, event_table=None,
              val_event_table=None, test_event_table=None, train_scores=None):
    """metrics-v3统一收尾：验证集冻结三个工作点，测试集只按冻结阈值评估。"""
    yva = np.asarray(yva).astype(int)
    yte = np.asarray(yte).astype(int)
    sva = np.asarray(sva, dtype=float)
    ste = np.asarray(ste, dtype=float)
    if not (len(yva) == len(sva) and len(yte) == len(ste)):
        raise ValueError("val/test labels 与 scores 必须分别等长")
    ts_val, turb_val = val_sidecars or _eval_sidecars("val", representation)
    ts_test, turb_test = test_sidecars or _eval_sidecars("test", representation)
    if not (len(ts_val) == len(turb_val) == len(yva)):
        raise ValueError("val score 与 timestamps/turbines 侧车未对齐")
    if not (len(ts_test) == len(turb_test) == len(yte)):
        raise ValueError("test score 与 timestamps/turbines 侧车未对齐")
    default_event_source = Path(event_table) if event_table is not None else DATA / "event_table.csv"
    val_event_source = Path(val_event_table) if val_event_table is not None else default_event_source
    test_event_source = Path(test_event_table) if test_event_table is not None else default_event_source
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if bool(meta.get("external_unlabeled", False)):
        return _report_external_v3(
            name,
            yva,
            sva,
            yte,
            ste,
            elapsed,
            representation=representation,
            ts_val=ts_val,
            turb_val=turb_val,
            ts_test=ts_test,
            turb_test=turb_test,
            meta=meta,
            extra=extra,
            train_scores=train_scores,
        )
    selection = select_operating_points(
        yva,
        sva,
        ts_val,
        turb_val,
        val_event_source,
        lead_steps=int(meta.get("lead_steps", 72)),
    )
    polarity = selection["score_polarity"]
    oriented_val = -sva if polarity == "negative" else sva
    oriented_test = -ste if polarity == "negative" else ste
    extra = dict(extra or {})
    common_context = {
        "model": name,
        "farm": FARM,
        "seed": 0,
        "label_mode": meta.get("label_mode", "real_fault_wl"),
        "preprocess_variant": meta.get("preprocess_variant", VARIANT),
        "score_polarity": polarity,
        "scores_are_probabilities": bool(extra.get("scores_are_probabilities", False)),
    }
    if "objective_loss" in extra:
        common_context["objective_loss"] = extra["objective_loss"]

    out_dir = RESULT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "score_val.npy", sva.astype(np.float32))
    np.save(out_dir / "score_test.npy", ste.astype(np.float32))
    np.save(out_dir / "labels_val.npy", yva.astype(np.int8))
    np.save(out_dir / "labels_test.npy", yte.astype(np.int8))
    np.save(out_dir / "timestamps_val.npy", np.asarray(ts_val, dtype=np.int64))
    np.save(out_dir / "timestamps_test.npy", np.asarray(ts_test, dtype=np.int64))
    np.save(out_dir / "turbines_val.npy", np.asarray(turb_val))
    np.save(out_dir / "turbines_test.npy", np.asarray(turb_test))

    workpoints = {}
    for wp_name, chosen in selection["workpoints"].items():
        threshold = float(chosen["threshold"])
        pred_val = (oriented_val >= threshold).astype(np.int8)
        pred_test = (oriented_test >= threshold).astype(np.int8)
        np.save(out_dir / f"pred_val_{wp_name}.npy", pred_val)
        np.save(out_dir / f"pred_test_{wp_name}.npy", pred_test)
        val_ctx = {
            **common_context,
            "split": "val",
            "workpoint": wp_name,
            "threshold_source": chosen["threshold_source"],
        }
        test_ctx = {**val_ctx, "split": "test"}
        val_eval = evaluate_all(
            yva,
            sva,
            predictions=pred_val,
            timestamps=ts_val,
            turbines=turb_val,
            event_table=val_event_source,
            threshold=threshold,
            context=val_ctx,
        )
        test_eval = evaluate_all(
            yte,
            ste,
            predictions=pred_test,
            timestamps=ts_test,
            turbines=turb_test,
            event_table=test_event_source,
            threshold=threshold,
            context=test_ctx,
        )
        workpoints[wp_name] = {"selection": chosen, "val": val_eval, "test": test_eval}

    balanced = workpoints["balanced"]
    rec = {
        "schema_version": "metrics-v3",
        "model": name,
        "farm": FARM,
        "seed": 0,
        "representation": representation,
        "elapsed_sec": round(float(elapsed), 3),
        "label_mode": common_context["label_mode"],
        "preprocess_variant": common_context["preprocess_variant"],
        "threshold_selection": selection,
        "workpoints": workpoints,
        # 兼容旧汇总消费者；仅指向balanced工作点。
        "val": balanced["val"]["metrics"],
        "test": balanced["test"]["metrics"],
        "extra": extra,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    bm = balanced["test"]["metrics"]
    print(
        f"[{FARM}/{name}] metrics-v3 balanced: event_f1={bm['event_f1']} "
        f"event_recall={bm['event_recall']} AUPRC={bm['auprc']} "
        f"FAR={bm['false_alarm_segments_per_turbine_day']} → {out_dir / 'metrics.json'}",
        flush=True,
    )
    return rec


def _external_monitoring(scores, pred, timestamps, turbines):
    scores = np.asarray(scores, dtype=float)
    pred = np.asarray(pred).astype(int)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    turbines = np.asarray(turbines).astype(str)
    finite = scores[np.isfinite(scores)]
    turbine_days = max(len(timestamps) * 10.0 / (60 * 24), 1e-9)
    segs = alarm_segments(pred, turbines, timestamps)
    return {
        "score_mean": float(np.mean(finite)) if len(finite) else None,
        "score_std": float(np.std(finite)) if len(finite) else None,
        "score_q50": float(np.quantile(finite, 0.5)) if len(finite) else None,
        "score_q99": float(np.quantile(finite, 0.99)) if len(finite) else None,
        "alarm_point_rate": float(np.mean(pred == 1)) if len(pred) else None,
        "n_alarm_segments": int(len(segs)),
        "observed_turbine_days": float(turbine_days),
        "alarm_segments_per_turbine_day": float(len(segs) / turbine_days),
    }


def _report_external_v3(name, yva, sva, yte, ste, elapsed, *, representation,
                        ts_val, turb_val, ts_test, turb_test, meta, extra, train_scores):
    """Hill无标签收尾：只用训练分数分位数定阈，不读取val/test标签性能。"""
    if train_scores is None:
        raise ValueError("external_unlabeled 协议必须显式提供 train_scores")
    train_scores = np.asarray(train_scores, dtype=float)
    finite_train = train_scores[np.isfinite(train_scores)]
    if finite_train.size == 0:
        raise ValueError("external_unlabeled 的 train_scores 没有有限值")
    out_dir = RESULT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "score_train.npy", train_scores.astype(np.float32))
    np.save(out_dir / "score_val.npy", np.asarray(sva, dtype=np.float32))
    np.save(out_dir / "score_test.npy", np.asarray(ste, dtype=np.float32))
    np.save(out_dir / "timestamps_val.npy", np.asarray(ts_val, dtype=np.int64))
    np.save(out_dir / "timestamps_test.npy", np.asarray(ts_test, dtype=np.int64))
    np.save(out_dir / "turbines_val.npy", np.asarray(turb_val))
    np.save(out_dir / "turbines_test.npy", np.asarray(turb_test))
    (out_dir / "unlabeled.json").write_text(
        json.dumps({"external_unlabeled": True, "labels_used_for_evaluation": False},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quantiles = {"q99": 0.99, "q995": 0.995, "q999": 0.999}
    selection = {
        "score_polarity": "positive",
        "threshold_source": "train_score_quantiles",
        "workpoints": {},
    }
    workpoints = {}
    for wp_name, q in quantiles.items():
        threshold = float(np.quantile(finite_train, q))
        selection["workpoints"][wp_name] = {"threshold": threshold, "quantile": q}
        pred_val = (np.asarray(sva, dtype=float) >= threshold).astype(np.int8)
        pred_test = (np.asarray(ste, dtype=float) >= threshold).astype(np.int8)
        np.save(out_dir / f"pred_val_{wp_name}.npy", pred_val)
        np.save(out_dir / f"pred_test_{wp_name}.npy", pred_test)
        context = {
            "model": name,
            "farm": FARM,
            "seed": 0,
            "label_mode": "external_unlabeled",
            "preprocess_variant": meta.get("preprocess_variant", VARIANT),
            "external_unlabeled": True,
            "score_polarity": "positive",
            "threshold_source": "train_score_quantiles",
            "workpoint": wp_name,
        }
        val_eval = evaluate_all(None, sva, predictions=pred_val, threshold=threshold,
                                context={**context, "split": "val"})
        test_eval = evaluate_all(None, ste, predictions=pred_test, threshold=threshold,
                                 context={**context, "split": "test"})
        workpoints[wp_name] = {
            "selection": selection["workpoints"][wp_name],
            "val": val_eval,
            "test": test_eval,
            "external_monitoring_val": _external_monitoring(sva, pred_val, ts_val, turb_val),
            "external_monitoring_test": _external_monitoring(ste, pred_test, ts_test, turb_test),
        }
    reference = workpoints["q995"]
    rec = {
        "schema_version": "metrics-v3",
        "model": name,
        "farm": FARM,
        "seed": 0,
        "representation": representation,
        "elapsed_sec": round(float(elapsed), 3),
        "label_mode": "external_unlabeled",
        "preprocess_variant": meta.get("preprocess_variant", VARIANT),
        "external_protocol": meta.get("external_protocol"),
        "threshold_selection": selection,
        "workpoints": workpoints,
        "val": reference["val"]["metrics"],
        "test": reference["test"]["metrics"],
        "extra": dict(extra or {}),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[{FARM}/{name}] external_unlabeled q99/q99.5/q99.9 → {out_dir / 'metrics.json'}",
          flush=True)
    return rec
