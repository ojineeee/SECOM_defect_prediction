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
"""
import json
from functools import partial
from pathlib import Path

from plot_style import plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from data import load_raw

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
RANDOM_STATE = 42
CONTAMINATION_DEFAULT = 0.05  # 실전 가정: 정확한 불량률을 모른다고 보고 흔히 쓰이는 기본값 사용


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


def main():
    X, y, _ = load_raw()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
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

    rows = []
    for name, model in detectors.items():
        model.fit(X_train_sel)
        raw_pred = model.predict(X_test_sel)  # 1=정상, -1=이상
        y_pred = (raw_pred == -1).astype(int)  # 1=불량으로 변환

        scores = model.decision_function(X_test_sel) if hasattr(model, "decision_function") else -raw_pred
        pr_auc = average_precision_score(y_test, -scores)

        rows.append(
            {
                "detector": name,
                "recall": round(recall_score(y_test, y_pred), 3),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
                "pr_auc": round(pr_auc, 3),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("recall", ascending=False)
    print("=== 비지도 이상탐지 vs 지도학습(분류) 비교 ===")
    print(result_df.to_string(index=False))

    supervised_path = ROOT / "results" / "final_report.json"
    comparison = {"unsupervised_anomaly_detection": result_df.to_dict(orient="records")}
    if supervised_path.exists():
        with open(supervised_path, encoding="utf-8") as f:
            supervised = json.load(f)
        comparison["supervised_classification"] = supervised["held_out_test_metrics"]
        comparison["supervised_classification"]["model"] = supervised["best_model"]

    with open(ROOT / "results" / "anomaly_vs_supervised.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = result_df["detector"]
    ax.bar(x, result_df["recall"], color="#DD8452", label="Recall")
    if "supervised_classification" in comparison:
        ax.axhline(
            comparison["supervised_classification"]["recall"],
            color="#4C72B0",
            linestyle="--",
            label=f"Supervised ({comparison['supervised_classification']['model']}) recall",
        )
    ax.set_ylabel("Recall (defect detection rate)")
    ax.set_title("Unsupervised anomaly detection vs supervised classification")
    ax.legend()
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_anomaly_vs_supervised.png", dpi=120)
    plt.close(fig)

    result_df.to_csv(ROOT / "results" / "anomaly_detection_comparison.csv", index=False)
    print("\n결과 저장 완료: results/anomaly_detection_comparison.csv, results/anomaly_vs_supervised.json")


if __name__ == "__main__":
    main()
