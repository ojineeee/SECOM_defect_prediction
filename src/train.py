"""SECOM 모델링: 불균형 처리 비교 + 모델 비교 + 최종 모델 평가.

데이터 누수를 피하기 위해 train/test 분리를 가장 먼저 하고,
imputer/scaler/feature-selection/resampling은 전부 Pipeline 안에서
train fold에만 적용되도록 구성한다.
"""
import json
from functools import partial
from pathlib import Path

from plot_style import plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data import load_raw

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def add_time_features(X, ts, *, reference_date):
    """학습 구간의 기준일을 사용해 타임스탬프 파생변수를 생성한다.

    EDA에서 수집 기간(2008-07-19~10-17, 약 3개월) 동안 불량률이 앞 절반 8.56%
    -> 뒤 절반 4.72%로 뚜렷하게 감소하는 시간 드리프트가 발견되어, 이를 피처화한다.

    ``reference_date``는 반드시 train timestamp에서 정하고 validation/test/추론에도
    그대로 재사용한다. split마다 ``ts.min()``을 다시 계산하면 동일한 값이 서로 다른
    날짜를 뜻하게 되어 학습과 추론의 시간축이 달라진다.
    """
    X = X.copy()
    ts = pd.to_datetime(ts)
    reference_date = pd.Timestamp(reference_date)
    X["days_since_start"] = (
        (ts - reference_date).dt.total_seconds().to_numpy() / 86400
    )
    # numpy 배열로 할당해 train_test_split 뒤에도 pandas index 정렬로 값이 어긋나지 않게 한다.
    X["hour_of_day"] = ts.dt.hour.to_numpy()
    X["day_of_week"] = ts.dt.dayofweek.to_numpy()
    return X


def build_base_steps():
    """결측치 처리 -> 상수 피처 제거 -> 스케일링 -> 상위 피처 선택 (누수 없이 fold 안에서 fit)"""
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("var_thresh", VarianceThreshold(threshold=1e-6)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=partial(mutual_info_classif, random_state=RANDOM_STATE), k=60)),
    ]


def compare_imbalance_strategies(X_train, y_train):
    """같은 모델(RandomForest)로 불균형 처리 방법만 바꿔가며 CV 비교."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1", "pr_auc": "average_precision"}

    strategies = {}

    steps = build_base_steps() + [("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300))]
    strategies["baseline_no_resampling"] = ImbPipeline(steps)

    steps = build_base_steps() + [
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300, class_weight="balanced"))
    ]
    strategies["class_weight_balanced"] = ImbPipeline(steps)

    steps = build_base_steps() + [
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300)),
    ]
    strategies["smote"] = ImbPipeline(steps)

    rows = []
    for name, pipe in strategies.items():
        result = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "strategy": name,
                "recall": result["test_recall"].mean(),
                "precision": result["test_precision"].mean(),
                "f1": result["test_f1"].mean(),
                "pr_auc": result["test_pr_auc"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("recall", ascending=False)


def compare_models(X_train, y_train, best_imbalance_strategy):
    """가장 좋았던 불균형 처리 방식을 고정하고 모델만 바꿔가며 CV 비교."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1", "pr_auc": "average_precision"}

    def with_resampling(clf):
        steps = build_base_steps()
        if best_imbalance_strategy == "smote":
            steps += [("smote", SMOTE(random_state=RANDOM_STATE))]
        if best_imbalance_strategy == "class_weight_balanced" and hasattr(clf, "class_weight"):
            clf.set_params(class_weight="balanced")
        steps += [("clf", clf)]
        return ImbPipeline(steps)

    models = {
        "logistic_regression": with_resampling(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "random_forest": with_resampling(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        "xgboost": with_resampling(
            XGBClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                scale_pos_weight=(1 if best_imbalance_strategy != "class_weight_balanced" else None),
            )
        ),
        "svm_rbf": with_resampling(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    }

    rows = []
    for name, pipe in models.items():
        result = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "recall": result["test_recall"].mean(),
                "precision": result["test_precision"].mean(),
                "f1": result["test_f1"].mean(),
                "pr_auc": result["test_pr_auc"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("recall", ascending=False), models


def main():
    X, y, ts = load_raw()
    X_train_sensors, X_test_sensors, y_train, y_test, ts_train, ts_test = train_test_split(
        X, y, ts, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    reference_date = ts_train.min()
    X_train = add_time_features(X_train_sensors, ts_train, reference_date=reference_date)
    X_test = add_time_features(X_test_sensors, ts_test, reference_date=reference_date)

    print("=== 0) 시간 파생변수 효과 검증 (SMOTE+LogisticRegression 고정, 5-fold CV) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1", "pr_auc": "average_precision"}
    time_fe_rows = []
    for label, data in [("sensors_only", X_train_sensors), ("sensors_plus_time_features", X_train)]:
        steps = build_base_steps() + [
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
        pipe = ImbPipeline(steps)
        result = cross_validate(pipe, data, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        time_fe_rows.append(
            {
                "feature_set": label,
                "recall": result["test_recall"].mean(),
                "precision": result["test_precision"].mean(),
                "f1": result["test_f1"].mean(),
                "pr_auc": result["test_pr_auc"].mean(),
            }
        )
    time_fe_df = pd.DataFrame(time_fe_rows)
    print(time_fe_df.to_string(index=False))
    use_time_features = time_fe_df.set_index("feature_set").loc["sensors_plus_time_features", "recall"] > \
        time_fe_df.set_index("feature_set").loc["sensors_only", "recall"]
    print(f"\n시간 파생변수 채택 여부: {use_time_features}")
    time_fe_df.to_csv(ROOT / "results" / "time_feature_comparison.csv", index=False)

    if not use_time_features:
        X_train, X_test = X_train_sensors, X_test_sensors

    print("\n=== 1) 불균형 처리 전략 비교 (RandomForest 고정, 5-fold CV) ===")
    imbalance_df = compare_imbalance_strategies(X_train, y_train)
    print(imbalance_df.to_string(index=False))
    best_strategy = imbalance_df.iloc[0]["strategy"]
    print(f"\n선택된 전략: {best_strategy}")

    print("\n=== 2) 모델 비교 (선택된 불균형 처리 고정, 5-fold CV) ===")
    model_df, models = compare_models(X_train, y_train, best_strategy)
    print(model_df.to_string(index=False))
    best_model_name = model_df.iloc[0]["model"]
    print(f"\n선택된 모델: {best_model_name}")

    # 최종 모델을 train 전체로 학습 후 held-out test로 평가
    final_pipe = models[best_model_name]
    final_pipe.fit(X_train, y_train)
    y_pred = final_pipe.predict(X_test)
    y_proba = final_pipe.predict_proba(X_test)[:, 1]

    test_metrics = {
        "recall": round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
    }
    print("\n=== 3) 최종 모델 held-out test 성능 ===")
    print(json.dumps(test_metrics, indent=2))
    print("\n", classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Pass", "Fail"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_model_name} ({best_strategy})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_confusion_matrix.png", dpi=120)
    plt.close(fig)

    # PR curve
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, name=best_model_name, ax=ax)
    ax.set_title("Precision-Recall Curve (held-out test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_pr_curve.png", dpi=120)
    plt.close(fig)

    # Threshold 조정 실험 (recall 우선 시 어떻게 바뀌는지)
    thresholds = np.arange(0.1, 0.95, 0.05)
    thresh_rows = []
    for t in thresholds:
        pred_t = (y_proba >= t).astype(int)
        thresh_rows.append(
            {
                "threshold": round(t, 2),
                "recall": round(recall_score(y_test, pred_t), 3),
                "precision": round(precision_score(y_test, pred_t, zero_division=0), 3),
                "f1": round(f1_score(y_test, pred_t, zero_division=0), 3),
            }
        )
    thresh_df = pd.DataFrame(thresh_rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresh_df["threshold"], thresh_df["recall"], marker="o", label="Recall")
    ax.plot(thresh_df["threshold"], thresh_df["precision"], marker="o", label="Precision")
    ax.plot(thresh_df["threshold"], thresh_df["f1"], marker="o", label="F1")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("score")
    ax.set_title("Threshold sweep — recall vs precision trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_threshold_sweep.png", dpi=120)
    plt.close(fig)

    # Feature importance: 트리 계열은 feature_importances_, 선형 모델은 |계수| 사용
    clf = final_pipe.named_steps["clf"]
    selected_mask = final_pipe.named_steps["select"].get_support()
    var_mask = final_pipe.named_steps["var_thresh"].get_support()
    cols_after_var = X_train.columns[var_mask]
    selected_cols = cols_after_var[selected_mask]

    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=selected_cols)
        importance_label = "importance"
    elif hasattr(clf, "coef_"):
        importances = pd.Series(np.abs(clf.coef_.ravel()), index=selected_cols)
        importance_label = "|coefficient|"
    else:
        importances = pd.Series(dtype=float)
        importance_label = None

    if not importances.empty:
        importances = importances.sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(7, 6))
        importances.iloc[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_xlabel(importance_label)
        ax.set_title(f"Top 15 feature importances — {best_model_name}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "09_feature_importance.png", dpi=120)
        plt.close(fig)
        top_features_out = importances.to_dict()
    else:
        top_features_out = {}

    # 결과 저장
    imbalance_df.to_csv(ROOT / "results" / "imbalance_strategy_comparison.csv", index=False)
    model_df.to_csv(ROOT / "results" / "model_comparison.csv", index=False)
    thresh_df.to_csv(ROOT / "results" / "threshold_sweep.csv", index=False)

    final_report = {
        "best_imbalance_strategy": best_strategy,
        "best_model": best_model_name,
        "time_reference_date": reference_date.isoformat(),
        "held_out_test_metrics": test_metrics,
        "top_feature_importances": {k: round(float(v), 4) for k, v in top_features_out.items()},
    }
    with open(ROOT / "results" / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print("\n결과 저장 완료: results/*.csv, results/final_report.json, results/figures/*.png")


if __name__ == "__main__":
    main()
