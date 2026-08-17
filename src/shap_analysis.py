"""최종 모델(SMOTE + Logistic Regression)에 대한 SHAP 설명력 분석.

계수 크기(|coefficient|) 기반 피처 중요도는 모델 전체 관점의 평균적인
영향력만 보여준다. SHAP은 (1) 피처 간 상관관계를 반영하고, (2) 개별
샘플 단위로 "왜 이 웨이퍼가 불량으로 예측됐는가"까지 설명할 수 있다.

주의: SHAP은 실제로 학습에 쓰인 입력(전처리+피처선택 이후의 60차원
공간)에 대해 계산해야 의미가 있다. 그래서 Pipeline에서 전처리 단계만
떼어내 transform한 뒤, 그 결과에 대해 LinearExplainer를 적용한다.
"""
import json
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import load_raw
from train import RANDOM_STATE, add_time_features, build_base_steps

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def main():
    X, y, ts = load_raw()
    X_fe = add_time_features(X, ts)
    X_train, X_test, y_train, y_test = train_test_split(
        X_fe, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # train.py의 최종 파이프라인과 동일하게 재구성 (SMOTE + LogisticRegression)
    preprocess = Pipeline(build_base_steps())
    preprocess.fit(X_train, y_train)
    X_train_t = preprocess.transform(X_train)
    X_test_t = preprocess.transform(X_test)

    var_mask = preprocess.named_steps["var_thresh"].get_support()
    cols_after_var = X_train.columns[var_mask]
    sel_mask = preprocess.named_steps["select"].get_support()
    feature_names = cols_after_var[sel_mask].tolist()

    X_train_sm, y_train_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_t, y_train)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(X_train_sm, y_train_sm)

    # 상관관계를 반영한 정확한 선형모델 SHAP (다중공선성이 있는 데이터이므로 중요)
    explainer = shap.LinearExplainer(
        clf, X_train_t, feature_perturbation="correlation_dependent"
    )
    shap_values = explainer.shap_values(X_test_t)

    X_test_df = pd.DataFrame(X_test_t, columns=feature_names)

    # 1. 전역 중요도 (mean |SHAP|) — 계수 기반 중요도와 비교할 대상
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
    top15_shap = mean_abs_shap.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(7, 6))
    top15_shap.iloc[::-1].plot(kind="barh", ax=ax, color="#55A868")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Top 15 features by SHAP importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_shap_importance_bar.png", dpi=120)
    plt.close(fig)

    # 2. Beeswarm summary plot — 값의 방향성(높을 때/낮을 때 영향)까지 확인
    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, X_test_df, max_display=15, show=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3. 개별 사례 설명 — 실제 불량을 불량으로 정확히 맞춘 사례 하나
    y_pred = clf.predict(X_test_t)
    y_proba = clf.predict_proba(X_test_t)[:, 1]
    correct_fail_idx = np.where((y_test.values == 1) & (y_pred == 1))[0]
    example_idx = int(correct_fail_idx[0]) if len(correct_fail_idx) > 0 else 0

    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[example_idx],
            base_values=explainer.expected_value,
            data=X_test_t[example_idx],
            feature_names=feature_names,
        ),
        max_display=12,
        show=False,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_shap_individual_example.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 계수 기반 중요도(이전 분석)와 비교
    coef_importance = pd.Series(np.abs(clf.coef_.ravel()), index=feature_names)
    top15_coef = set(coef_importance.sort_values(ascending=False).head(15).index)
    overlap = top15_coef & set(top15_shap.index)

    report = {
        "n_features_explained": len(feature_names),
        "example_wafer_test_index": example_idx,
        "example_wafer_true_label": "Fail",
        "example_wafer_predicted_proba_fail": round(float(y_proba[example_idx]), 4),
        "top15_shap_features": {k: round(float(v), 4) for k, v in top15_shap.items()},
        "top15_coef_vs_shap_overlap_count": len(overlap),
        "top15_overlap_features": sorted(overlap),
    }
    with open(ROOT / "results" / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
