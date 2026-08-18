"""시간순 Walk-forward(확장 윈도우) 검증 + 주요 성능 지표의 부트스트랩 신뢰구간.

chronological_validation.py는 "앞 80% 학습 / 뒤 20% 평가"라는 단일 분할
결과였다(Recall 0). 이건 극단적인 한 시점의 스냅샷일 뿐이라, 정말 이
데이터가 통째로 시간에 못 버티는 것인지 아니면 특정 구간만 유독
어려운 것인지 구분이 안 된다. TimeSeriesSplit으로 여러 개의 확장
윈도우 fold를 만들어(과거 -> 다음 구간을 반복) 시점마다 어떻게
달라지는지 확인한다.

또한 지금까지 보고한 모든 성능 차이(랜덤 vs 시간순, 파생변수 유무 등)는
단일 실행 결과였다. 314개(혹은 그 이하) 테스트 샘플의 부트스트랩
신뢰구간을 계산해 "그 차이가 노이즈보다 큰가"를 확인한다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from data import load_raw
from train import RANDOM_STATE, add_time_features, build_base_steps

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(RANDOM_STATE)


def make_pipe():
    steps = build_base_steps() + [
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]
    return ImbPipeline(steps)


def bootstrap_ci(y_true, y_pred, y_proba, metric_fn, n=N_BOOTSTRAP, needs_proba=False):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    n_samples = len(y_true)
    scores = []
    for _ in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        yt, yp = y_true[idx], y_pred[idx]
        if yt.sum() == 0:  # 불량이 하나도 없는 리샘플은 recall/precision 정의 불가 -> 스킵
            continue
        if needs_proba:
            scores.append(metric_fn(yt, y_proba[idx]))
        else:
            scores.append(metric_fn(yt, yp, zero_division=0))
    scores = np.array(scores)
    return {
        "point": None,  # 호출부에서 채움
        "ci_lower": round(float(np.percentile(scores, 2.5)), 4),
        "ci_upper": round(float(np.percentile(scores, 97.5)), 4),
        "n_bootstrap_used": int(len(scores)),
    }


def evaluate_with_ci(X_train, y_train, X_test, y_test, label):
    pipe = make_pipe()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    point = {
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4)
        if y_test.sum() > 0 else None,
    }

    ci = {}
    ci["recall"] = bootstrap_ci(y_test.values, y_pred, y_proba, recall_score)
    ci["recall"]["point"] = point["recall"]
    ci["precision"] = bootstrap_ci(y_test.values, y_pred, y_proba, precision_score)
    ci["precision"]["point"] = point["precision"]
    ci["f1"] = bootstrap_ci(y_test.values, y_pred, y_proba, f1_score)
    ci["f1"]["point"] = point["f1"]
    if point["pr_auc"] is not None:
        ci["pr_auc"] = bootstrap_ci(y_test.values, y_pred, y_proba, average_precision_score, needs_proba=True)
        ci["pr_auc"]["point"] = point["pr_auc"]

    print(f"\n=== {label} ===")
    for k, v in ci.items():
        print(f"  {k}: {v['point']}  (95% CI: {v['ci_lower']} ~ {v['ci_upper']}, n={v['n_bootstrap_used']})")

    return {"point_estimate": point, "bootstrap_ci": ci, "n_test": int(len(y_test)), "n_test_fail": int(y_test.sum())}


def main():
    X, y, ts = load_raw()
    X_fe = add_time_features(X, ts)

    results = {}

    # 1) 기존 랜덤 분할 최종 모델에 신뢰구간 추가
    X_train, X_test, y_train, y_test = train_test_split(
        X_fe, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    results["random_split"] = evaluate_with_ci(X_train, y_train, X_test, y_test, "랜덤 분할 (기존 최종 모델)")

    # 2) 기존 단일 시간순 분할(80/20)에 신뢰구간 추가
    order = np.argsort(ts.values)
    split_point = int(len(order) * 0.8)
    tr_idx, te_idx = order[:split_point], order[split_point:]
    results["single_chronological_split"] = evaluate_with_ci(
        X_fe.iloc[tr_idx], y.iloc[tr_idx], X_fe.iloc[te_idx], y.iloc[te_idx],
        "시간순 단일 분할 (80/20)"
    )

    # 3) Walk-forward: TimeSeriesSplit으로 여러 확장 윈도우 fold
    print("\n=== Walk-forward (TimeSeriesSplit, 4 folds) ===")
    X_sorted = X_fe.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)
    ts_sorted = ts.iloc[order].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=4)
    fold_results = []
    for i, (tr, te) in enumerate(tscv.split(X_sorted), start=1):
        pipe = make_pipe()
        y_tr_fold, y_te_fold = y_sorted.iloc[tr], y_sorted.iloc[te]
        if y_tr_fold.sum() == 0 or y_te_fold.sum() == 0:
            print(f"fold {i}: train 또는 test에 불량이 0건이라 스킵")
            continue
        pipe.fit(X_sorted.iloc[tr], y_tr_fold)
        y_pred = pipe.predict(X_sorted.iloc[te])
        y_proba = pipe.predict_proba(X_sorted.iloc[te])[:, 1]
        fold_metrics = {
            "fold": i,
            "train_period": [str(ts_sorted.iloc[tr].min()), str(ts_sorted.iloc[tr].max())],
            "test_period": [str(ts_sorted.iloc[te].min()), str(ts_sorted.iloc[te].max())],
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_test_fail": int(y_te_fold.sum()),
            "recall": round(float(recall_score(y_te_fold, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_te_fold, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_te_fold, y_pred, zero_division=0)), 4),
            "pr_auc": round(float(average_precision_score(y_te_fold, y_proba)), 4),
        }
        fold_results.append(fold_metrics)
        print(f"fold {i}: test={fold_metrics['test_period']}  n_fail={fold_metrics['n_test_fail']}  "
              f"recall={fold_metrics['recall']}  pr_auc={fold_metrics['pr_auc']}")

    results["walk_forward_folds"] = fold_results
    if fold_results:
        results["walk_forward_summary"] = {
            "mean_recall": round(float(np.mean([f["recall"] for f in fold_results])), 4),
            "mean_pr_auc": round(float(np.mean([f["pr_auc"] for f in fold_results])), 4),
            "std_recall": round(float(np.std([f["recall"] for f in fold_results])), 4),
        }

    with open(ROOT / "results" / "walk_forward_and_ci.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 시각화: 세 가지 평가 방식의 recall 비교 (신뢰구간 포함)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Random split", "Single chronological\n(80/20)"] + [f"Walk-forward\nfold {f['fold']}" for f in fold_results]
    points = [results["random_split"]["bootstrap_ci"]["recall"]["point"],
              results["single_chronological_split"]["bootstrap_ci"]["recall"]["point"]] + \
             [f["recall"] for f in fold_results]
    lowers = [results["random_split"]["bootstrap_ci"]["recall"]["ci_lower"],
              results["single_chronological_split"]["bootstrap_ci"]["recall"]["ci_lower"]] + [None] * len(fold_results)
    uppers = [results["random_split"]["bootstrap_ci"]["recall"]["ci_upper"],
              results["single_chronological_split"]["bootstrap_ci"]["recall"]["ci_upper"]] + [None] * len(fold_results)
    x = np.arange(len(labels))
    colors = ["#4C72B0", "#C44E52"] + ["#DD8452"] * len(fold_results)
    ax.bar(x, points, color=colors)
    for i, (lo, hi) in enumerate(zip(lowers, uppers)):
        if lo is not None:
            ax.errorbar(i, points[i], yerr=[[points[i]-lo], [hi-points[i]]], color="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Recall")
    ax.set_title("평가 방식별 Recall (막대=Random/시간순 분할은 95% 부트스트랩 CI 포함)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "18_walk_forward_recall.png", dpi=120)
    plt.close(fig)

    print("\n결과 저장 완료: results/walk_forward_and_ci.json, figures/18_walk_forward_recall.png")


if __name__ == "__main__":
    main()
