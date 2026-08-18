"""모델 선택을 walk-forward 기준으로 재검증.

train.py의 "Logistic Regression이 최고"라는 결론은 무작위 분할 5-fold CV로
정해졌다. 하지만 walk_forward_validation.py에서 무작위 분할 성능이 시간적
일반화를 대변하지 못한다는 걸 확인했으므로, 모델을 고르는 기준 자체도
walk-forward로 다시 검증해야 앞뒤가 맞는다. 4개 모델(Logistic Regression,
Random Forest, XGBoost, SVM)을 동일한 walk-forward fold에서 비교한다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data import load_raw
from train import RANDOM_STATE, add_time_features, build_base_steps

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def build_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=300, random_state=RANDOM_STATE, eval_metric="logloss"),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def main():
    X, y, ts = load_raw()
    X_fe = add_time_features(X, ts)
    order = np.argsort(ts.values)
    X_sorted = X_fe.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)
    ts_sorted = ts.iloc[order].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=4)
    fold_indices = [(tr, te) for tr, te in tscv.split(X_sorted)
                    if y_sorted.iloc[tr].sum() > 0 and y_sorted.iloc[te].sum() > 0]

    all_results = {name: [] for name in build_models()}

    for fold_i, (tr, te) in enumerate(fold_indices, start=1):
        y_tr, y_te = y_sorted.iloc[tr], y_sorted.iloc[te]
        print(f"\n--- fold {fold_i}: test {ts_sorted.iloc[te].min()} ~ {ts_sorted.iloc[te].max()} "
              f"(n_fail={int(y_te.sum())}) ---")
        for name, clf in build_models().items():
            steps = build_base_steps() + [("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", clf)]
            pipe = ImbPipeline(steps)
            pipe.fit(X_sorted.iloc[tr], y_tr)
            y_pred = pipe.predict(X_sorted.iloc[te])
            y_proba = pipe.predict_proba(X_sorted.iloc[te])[:, 1]
            metrics = {
                "fold": fold_i,
                "recall": round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
                "pr_auc": round(float(average_precision_score(y_te, y_proba)), 4),
            }
            all_results[name].append(metrics)
            print(f"  {name:22s} recall={metrics['recall']:.3f}  pr_auc={metrics['pr_auc']:.3f}")

    summary = {}
    for name, folds in all_results.items():
        summary[name] = {
            "mean_recall": round(float(np.mean([f["recall"] for f in folds])), 4),
            "mean_pr_auc": round(float(np.mean([f["pr_auc"] for f in folds])), 4),
            "mean_f1": round(float(np.mean([f["f1"] for f in folds])), 4),
            "folds": folds,
        }

    print("\n=== Walk-forward 기준 모델별 평균 성능 ===")
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["mean_recall"], reverse=True)
    for name, s in ranked:
        print(f"  {name:22s} mean_recall={s['mean_recall']:.4f}  mean_pr_auc={s['mean_pr_auc']:.4f}")

    # 참고: train.py의 랜덤 분할 CV 기준 순위
    random_cv_ranking = {
        "logistic_regression": 0.387, "svm_rbf": 0.120, "xgboost": 0.072, "random_forest": 0.036,
    }

    result = {
        "walk_forward_model_summary": summary,
        "walk_forward_ranking_by_recall": [name for name, _ in ranked],
        "random_split_cv_ranking_reference": random_cv_ranking,
    }
    with open(ROOT / "results" / "walk_forward_model_selection.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(summary.keys())
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, [summary[n]["mean_recall"] for n in names], width, label="Walk-forward mean recall")
    ax.bar(x + width/2, [random_cv_ranking[n] for n in names], width, label="Random-split CV recall (기존)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Recall")
    ax.set_title("모델 선택: 무작위 분할 CV vs Walk-forward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "20_model_selection_walk_forward.png", dpi=120)
    plt.close(fig)

    print("\n결과 저장 완료: results/walk_forward_model_selection.json, figures/20_model_selection_walk_forward.png")


if __name__ == "__main__":
    main()
