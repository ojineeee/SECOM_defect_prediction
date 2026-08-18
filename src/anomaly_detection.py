"""라벨 없이(One-class 기반) 불량을 탐지할 수 있는지 비교하는 보조 실험.

지도학습(분류) 결과와 나란히 비교해서, "라벨이 부족한 상황을 가정하면
이상탐지가 대안이 될 수 있는가"를 확인한다.

주의: 완전한 비지도 학습이 아니라는 점을 명시해둔다.
- 피처 선택 단계는 라벨(y)을 사용하는 mutual_info_classif이므로 약한 지도 신호가
  들어간다. 다만 선택 자체는 "정상 vs 불량이 모두 섞인" 전체 train(X_train, y_train)에
  대해 한 번만 수행하고, 그 이후 이상탐지 모델 자체는 정상(Pass) 샘플만으로 학습한다
  (이전 버전은 정상 샘플만 걸러낸 뒤 피처 선택을 수행해 y가 전부 0인 상수 라벨에 대해
  mutual_info를 계산하는 버그가 있었다 — 사실상 무작위 선택과 다름없었다).
- contamination/nu는 실제 test 불량률을 그대로 넣지 않는다. 실전에서는 이 값을 몰라야
  정상적인 시나리오이므로, 일반적으로 쓰이는 보수적 기본값(5%)을 사용한다.
- One-class 모델의 기본 predict()(contamination=5%)와 지도학습의 기본 threshold(0.5)를
  그대로 비교하면, 두 방식이 실제로 플래깅하는 웨이퍼 개수가 서로 달라져 Recall을 그대로
  비교하는 게 불공정하다(이전 버전의 문제). 그래서 세 모델 모두 "이상치/불량 점수 기준
  테스트셋 상위 5%를 플래깅"하는 동일한 경고 예산으로 다시 비교한다.
"""
import json
from functools import partial
from pathlib import Path

from plot_style import plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from data import load_raw
from train import add_time_features, build_base_steps

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
RANDOM_STATE = 42
CONTAMINATION_DEFAULT = 0.05  # 실전 가정: 정확한 불량률을 모른다고 보고 흔히 쓰이는 기본값 사용
FLAG_FRACTION = 0.05  # 공정 비교용 경고 예산 (테스트셋 상위 5% 플래깅) — contamination과 동일하게 맞춤


def fit_preprocess(X_train, y_train):
    """전체 train(정상+불량 라벨 포함)에 대해 전처리를 fit — 피처 선택에 의미 있는 라벨 변동이 있도록."""
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)

    vt = VarianceThreshold(threshold=1e-6)
    X_train_v = vt.fit_transform(X_train_i)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_v)

    selector = SelectKBest(score_func=partial(mutual_info_classif, random_state=RANDOM_STATE), k=60)
    selector.fit(X_train_s, y_train)

    return imputer, vt, scaler, selector


def apply_preprocess(X, imputer, vt, scaler, selector):
    X_i = imputer.transform(X)
    X_v = vt.transform(X_i)
    X_s = scaler.transform(X_v)
    return selector.transform(X_s)


def flag_top_fraction(scores_most_anomalous_last, fraction, n):
    """scores가 클수록 더 이상(불량스럽다)일 때, 상위 fraction만큼을 플래깅한 인덱스를 반환."""
    n_flag = max(1, round(n * fraction))
    order = np.argsort(-scores_most_anomalous_last)  # 내림차순 (가장 의심스러운 것부터)
    return order[:n_flag], n_flag


def main():
    X, y, ts = load_raw()
    X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
        X, y, ts, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 전처리(결측치·분산·스케일·피처선택)는 정상+불량이 섞인 전체 train으로 fit
    imputer, vt, scaler, selector = fit_preprocess(X_train, y_train)

    # 이상탐지 모델 자체는 "정상 데이터의 분포"만 배우는 것이 목적이므로,
    # 학습에는 정상(Pass) 샘플만 사용한다 (실무에서 불량 라벨이 거의 없다고 가정).
    X_train_pass = X_train[y_train == 0]
    X_train_sel = apply_preprocess(X_train_pass, imputer, vt, scaler, selector)
    X_test_sel = apply_preprocess(X_test, imputer, vt, scaler, selector)

    contamination = CONTAMINATION_DEFAULT

    detectors = {
        "isolation_forest": IsolationForest(
            n_estimators=300, contamination=contamination, random_state=RANDOM_STATE
        ),
        "one_class_svm": OneClassSVM(nu=contamination, kernel="rbf", gamma="scale"),
        # EllipticEnvelope는 60차원 공분산 추정이 불안정해(수렴 경고 다발) 제외함
    }

    y_test_arr = y_test.values
    n_test = len(y_test_arr)
    anomaly_scores = {}  # 클수록 더 이상치(불량스럽다)로 정규화

    default_threshold_rows = []
    for name, model in detectors.items():
        model.fit(X_train_sel)
        raw_pred = model.predict(X_test_sel)  # 1=정상, -1=이상
        y_pred = (raw_pred == -1).astype(int)  # 1=불량으로 변환

        raw_scores = model.decision_function(X_test_sel)  # 클수록 "정상"에 가까움 (sklearn 관례)
        anomaly_scores[name] = -raw_scores  # 부호 반전 -> 클수록 이상치
        pr_auc = average_precision_score(y_test_arr, anomaly_scores[name])

        default_threshold_rows.append(
            {
                "detector": name,
                "recall": round(recall_score(y_test_arr, y_pred), 3),
                "precision": round(precision_score(y_test_arr, y_pred, zero_division=0), 3),
                "pr_auc": round(pr_auc, 3),
            }
        )

    default_threshold_df = pd.DataFrame(default_threshold_rows).sort_values("recall", ascending=False)
    print("=== One-class 이상탐지 결과 (모델 기본 predict(), contamination=5%) ===")
    print(default_threshold_df.to_string(index=False))

    # 지도학습 pipeline: train.py 최종 파이프라인과 동일(시간 파생변수 + SMOTE + LogisticRegression)
    X_train_fe = add_time_features(X_train, ts_train)
    X_test_fe = add_time_features(X_test, ts_test)
    steps = build_base_steps() + [
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]
    supervised_pipe = ImbPipeline(steps)
    supervised_pipe.fit(X_train_fe, y_train)
    y_proba = supervised_pipe.predict_proba(X_test_fe)[:, 1]
    anomaly_scores["supervised_logistic_regression"] = y_proba

    supervised_default_pred = (y_proba >= 0.5).astype(int)
    supervised_default_row = {
        "detector": "supervised_logistic_regression",
        "recall": round(recall_score(y_test_arr, supervised_default_pred), 3),
        "precision": round(precision_score(y_test_arr, supervised_default_pred, zero_division=0), 3),
        "pr_auc": round(average_precision_score(y_test_arr, y_proba), 3),
    }
    print("\n=== 지도학습 결과 (기본 threshold=0.5, 참고용) ===")
    print(pd.DataFrame([supervised_default_row]).to_string(index=False))

    # --- 공정 비교: 세 모델 모두 "테스트셋 상위 5%(경고 예산 고정)" 플래깅 기준으로 재계산 ---
    matched_rows = []
    for name, scores in anomaly_scores.items():
        flagged_idx, n_flag = flag_top_fraction(scores, FLAG_FRACTION, n_test)
        y_pred_matched = np.zeros(n_test, dtype=int)
        y_pred_matched[flagged_idx] = 1
        matched_rows.append(
            {
                "detector": name,
                "n_flagged": int(n_flag),
                "recall": round(recall_score(y_test_arr, y_pred_matched), 3),
                "precision": round(precision_score(y_test_arr, y_pred_matched, zero_division=0), 3),
            }
        )
    matched_df = pd.DataFrame(matched_rows).sort_values("recall", ascending=False)
    print(f"\n=== 공정 비교: 테스트셋 상위 {FLAG_FRACTION:.0%} 플래깅 (n={matched_df['n_flagged'].iloc[0]}) ===")
    print(matched_df.to_string(index=False))

    comparison = {
        "note": "matched_budget_comparison이 공정한 비교(세 모델 모두 동일 경고 예산). "
                "default_threshold_reference는 각 모델의 기본 predict()/threshold=0.5를 그대로 쓴 "
                "참고용 수치로, 서로 다른 경고량을 비교하는 것이므로 Recall을 직접 비교하면 안 됨.",
        "matched_budget_comparison": matched_df.to_dict(orient="records"),
        "one_class_default_threshold_reference": default_threshold_df.to_dict(orient="records"),
        "supervised_default_threshold_reference": supervised_default_row,
    }

    with open(ROOT / "results" / "anomaly_vs_supervised.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = matched_df["detector"]
    colors = ["#4C72B0" if "supervised" in n else "#DD8452" for n in x]
    ax.bar(x, matched_df["recall"], color=colors)
    ax.set_ylabel("Recall (defect detection rate)")
    ax.set_title(f"One-class anomaly detection vs supervised — top {FLAG_FRACTION:.0%} flagged (matched budget)")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_anomaly_vs_supervised.png", dpi=120)
    plt.close(fig)

    matched_df.to_csv(ROOT / "results" / "anomaly_detection_comparison.csv", index=False)
    print("\n결과 저장 완료: results/anomaly_detection_comparison.csv, results/anomaly_vs_supervised.json")


if __name__ == "__main__":
    main()
