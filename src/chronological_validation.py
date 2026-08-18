"""시간순 분할(chronological split) 검증.

지금까지의 최종 모델은 랜덤 stratified split으로 평가했다. 그런데 EDA에서
수집 기간(3개월) 동안 불량률이 8.56% -> 4.72%로 뚜렷하게 줄어드는 시간
드리프트를 발견했었다. 랜덤 분할은 과거와 미래 데이터를 뒤섞어서
train/test를 나누기 때문에, 실제 운영처럼 "과거로 학습해서 미래를
예측"하는 상황보다 성능이 낙관적으로 나올 수 있다.

이 스크립트는 같은 전처리·모델(SMOTE + LogisticRegression)을 그대로 쓰되,
분할 방식만 "시간순 앞 80% 학습 / 뒤 20% 평가"로 바꿔서 실제로 성능이
달라지는지 확인한다.
"""
import json
from pathlib import Path

from plot_style import plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from data import load_raw
from train import RANDOM_STATE, add_time_features, build_base_steps

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def main():
    X, y, ts = load_raw()
    X_fe = add_time_features(X, ts)

    # 시간순 정렬 후 앞 80% / 뒤 20%로 분할 (랜덤 아님)
    order = np.argsort(ts.values)
    split_point = int(len(order) * 0.8)
    train_idx, test_idx = order[:split_point], order[split_point:]

    X_train, X_test = X_fe.iloc[train_idx], X_fe.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"train 기간: {ts.iloc[train_idx].min()} ~ {ts.iloc[train_idx].max()}")
    print(f"test 기간:  {ts.iloc[test_idx].min()} ~ {ts.iloc[test_idx].max()}")
    print(f"train 불량률: {y_train.mean():.4f}  ({y_train.sum()}/{len(y_train)})")
    print(f"test 불량률:  {y_test.mean():.4f}  ({y_test.sum()}/{len(y_test)})")

    steps = build_base_steps() + [
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]
    pipe = ImbPipeline(steps)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "recall": round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
    }
    print("\n=== 시간순 분할 held-out 성능 ===")
    print(json.dumps(metrics, indent=2))
    print("\n", classification_report(y_test, y_pred, target_names=["Pass", "Fail"], zero_division=0))

    # train.py에 기록된 랜덤 분할 최종 성능(재실행 없이 문서화된 값 그대로 비교)
    random_split_metrics = {"recall": 0.381, "precision": 0.098, "f1": 0.155, "pr_auc": 0.152}

    comparison = {
        "train_period": [str(ts.iloc[train_idx].min()), str(ts.iloc[train_idx].max())],
        "test_period": [str(ts.iloc[test_idx].min()), str(ts.iloc[test_idx].max())],
        "train_defect_rate": round(float(y_train.mean()), 4),
        "test_defect_rate": round(float(y_test.mean()), 4),
        "test_n_fail": int(y_test.sum()),
        "test_n_total": int(len(y_test)),
        "chronological_split": metrics,
        "random_split_reference": random_split_metrics,
    }
    with open(ROOT / "results" / "chronological_split_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 혼동행렬
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pass", "Fail"]); ax.set_yticklabels(["Pass", "Fail"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — chronological split")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_chronological_confusion_matrix.png", dpi=120)
    plt.close(fig)

    # 랜덤 분할 vs 시간순 분할 비교 막대그래프
    labels = ["recall", "precision", "f1", "pr_auc"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [random_split_metrics[m] for m in labels], 0.4, label="Random split (기존)")
    ax.bar(x + 0.2, [metrics[m] for m in labels], 0.4, label="Chronological split (시간순)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Random split vs Chronological split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_split_comparison.png", dpi=120)
    plt.close(fig)

    print("\n결과 저장 완료: results/chronological_split_comparison.json, figures/14~15")


if __name__ == "__main__":
    main()
