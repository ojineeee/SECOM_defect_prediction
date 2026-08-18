"""운영 관점 지표 추가: Recall@상위 N% 경고, 웨이퍼 1,000장당 오경보, Precision@고정 Recall.

Recall/Precision만으로는 "실제로 이 모델을 켜두면 하루에 몇 번 오경보가
울리는가" 같은, 현장 담당자가 실제로 묻는 질문에 답하지 못한다. 이를
직접 계산해 덧붙인다.
"""
import json
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data import load_raw
from train import RANDOM_STATE, add_time_features, build_base_steps

ROOT = Path(__file__).resolve().parent.parent


def main():
    X, y, ts = load_raw()
    X_train_raw, X_test_raw, y_train, y_test, ts_train, ts_test = train_test_split(
        X, y, ts, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    reference_date = ts_train.min()
    X_train = add_time_features(X_train_raw, ts_train, reference_date=reference_date)
    X_test = add_time_features(X_test_raw, ts_test, reference_date=reference_date)

    steps = build_base_steps() + [
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]
    pipe = ImbPipeline(steps)
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    order = np.argsort(-y_proba)  # 확률 높은 순
    n = len(y_true)
    n_fail = int(y_true.sum())

    results = {}
    for pct in [5, 10, 20, 30]:
        top_n = max(1, int(round(n * pct / 100)))
        flagged_idx = order[:top_n]
        recall_at_pct = float(y_true[flagged_idx].sum()) / n_fail if n_fail > 0 else None
        n_false_alarms = int(top_n - y_true[flagged_idx].sum())
        results[f"top_{pct}pct"] = {
            "n_flagged": int(top_n),
            "recall": round(recall_at_pct, 4) if recall_at_pct is not None else None,
            "false_alarms_per_1000_wafers": round(n_false_alarms / n * 1000, 1),
        }

    # Precision@고정 Recall (threshold_sweep.csv 기반 보간)
    import pandas as pd
    sweep = pd.read_csv(ROOT / "results" / "threshold_sweep.csv")
    fixed_recall_targets = [0.3, 0.5, 0.7]
    precision_at_recall = {}
    for target in fixed_recall_targets:
        candidates = sweep[sweep["recall"] >= target]
        if len(candidates) > 0:
            row = candidates.iloc[candidates["recall"].values.argmin()]  # target에 가장 가까운(살짝 넘는) 지점
            precision_at_recall[f"recall_{target}"] = {
                "threshold": float(row["threshold"]), "actual_recall": float(row["recall"]),
                "precision": float(row["precision"]),
            }
        else:
            precision_at_recall[f"recall_{target}"] = None

    out = {
        "n_test": n, "n_test_fail": n_fail,
        "recall_at_top_pct_flagged": results,
        "precision_at_fixed_recall": precision_at_recall,
    }
    with open(ROOT / "results" / "operational_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
