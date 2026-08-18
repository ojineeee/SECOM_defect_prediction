"""SECOM 탐색적 데이터 분석. results/figures/, results/eda_summary.md 생성."""
import json
from pathlib import Path

from plot_style import plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from data import load_raw

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    X, y, ts = load_raw()
    summary = {}

    # 1. 라벨 분포
    label_counts = y.value_counts()
    summary["n_samples"] = int(len(y))
    summary["n_features_raw"] = int(X.shape[1])
    summary["n_fail"] = int(label_counts.get(1, 0))
    summary["n_pass"] = int(label_counts.get(0, 0))
    summary["fail_ratio_pct"] = round(100 * y.mean(), 2)

    fig, ax = plt.subplots(figsize=(4, 4))
    label_counts.rename({0: "Pass", 1: "Fail"}).plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Label distribution")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_label_distribution.png", dpi=120)
    plt.close(fig)

    # 2. 결측치 비율
    missing_rate = X.isna().mean().sort_values(ascending=False)
    summary["cols_missing_over_55pct"] = int((missing_rate > 0.55).sum())
    summary["cols_any_missing"] = int((missing_rate > 0).sum())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(missing_rate, bins=40, color="#55A868")
    ax.axvline(0.55, color="red", linestyle="--", label="55% cutoff")
    ax.set_title("Per-feature missing rate")
    ax.set_xlabel("missing rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_missing_rate_hist.png", dpi=120)
    plt.close(fig)

    # 2b. 결측 심한 컬럼이 센서 인덱스상 뭉쳐있는지 확인 (같은 물리 서브시스템 의심)
    heavy_missing_idx = sorted(int(c.split("_")[1]) for c in missing_rate[missing_rate > 0.55].index)
    clusters, cur = [], [heavy_missing_idx[0]]
    for i in heavy_missing_idx[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    summary["heavy_missing_sensor_index_clusters"] = clusters

    # 2c. 시간 드리프트: 수집 기간(약 3개월) 동안 불량률이 안정적인지 확인
    order = ts.rank(method="first").astype(int) - 1
    df_time = pd.DataFrame({"label": y.values, "order": order.values}).sort_values("order")
    half = len(df_time) // 2
    first_half_rate = df_time.iloc[:half]["label"].mean()
    second_half_rate = df_time.iloc[half:]["label"].mean()
    rolling_rate = df_time["label"].rolling(50, min_periods=50).mean()
    summary["defect_rate_first_half"] = round(float(first_half_rate), 4)
    summary["defect_rate_second_half"] = round(float(second_half_rate), 4)
    summary["rolling50_defect_rate_min_max"] = [
        round(float(rolling_rate.min()), 4),
        round(float(rolling_rate.max()), 4),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_time["order"], rolling_rate, color="#C44E52")
    ax.axhline(y.mean(), color="gray", linestyle="--", label=f"overall mean ({y.mean():.3f})")
    ax.set_title("Rolling (window=50) defect rate over collection order")
    ax.set_xlabel("sample order (chronological)")
    ax.set_ylabel("defect rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02b_defect_rate_drift.png", dpi=120)
    plt.close(fig)

    # 3. 결측치 처리 + 상수 피처 제거 (이후 단계 공통 전처리)
    keep_cols = missing_rate[missing_rate <= 0.55].index
    X_reduced = X[keep_cols]
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_reduced), columns=keep_cols)

    variances = X_imputed.var()
    near_constant = variances[variances < 1e-6].index
    X_clean = X_imputed.drop(columns=near_constant)
    summary["cols_after_missing_drop"] = int(X_reduced.shape[1])
    summary["cols_near_constant_removed"] = int(len(near_constant))
    summary["cols_after_cleaning"] = int(X_clean.shape[1])

    # 4. PCA 2D 시각화
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X_clean)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    summary["pca_explained_var_ratio"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color, name in [(0, "#4C72B0", "Pass"), (1, "#C44E52", "Fail")]:
        mask = y == label
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=12, alpha=0.6, label=name, color=color)
    ax.set_title("PCA (2D) — Pass vs Fail")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_pca_scatter.png", dpi=120)
    plt.close(fig)

    # 5. 라벨과 상호정보량 상위 피처
    mi = mutual_info_classif(X_clean, y, random_state=42, discrete_features=False)
    mi_series = pd.Series(mi, index=X_clean.columns).sort_values(ascending=False)
    top_features = mi_series.head(10)
    summary["top10_mutual_info_features"] = {k: round(float(v), 5) for k, v in top_features.items()}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, feat in zip(axes.flat, top_features.index[:6]):
        data_pass = X_clean.loc[y == 0, feat]
        data_fail = X_clean.loc[y == 1, feat]
        ax.boxplot([data_pass, data_fail], tick_labels=["Pass", "Fail"])
        ax.set_title(feat, fontsize=9)
    fig.suptitle("Top discriminative sensors (by mutual information)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_top_features_boxplot.png", dpi=120)
    plt.close(fig)

    # 5b. 상위 피처들끼리도 중복(다중공선성)이 있는지 확인
    top10_corr = X_clean[top_features.index].corr()
    high_corr_pairs = []
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            c = top10_corr.iloc[i, j]
            if abs(c) > 0.7:
                high_corr_pairs.append(
                    [top_features.index[i], top_features.index[j], round(float(c), 3)]
                )
    summary["top10_features_high_corr_pairs"] = high_corr_pairs

    # 6. 상관관계 (상위 피처들만)
    top30 = mi_series.head(30).index
    corr = X_clean[top30].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation heatmap — top 30 informative sensors")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_correlation_heatmap_top30.png", dpi=120)
    plt.close(fig)

    # 저장
    X_clean.to_csv(ROOT / "data" / "processed" / "X_clean.csv", index=False)
    y.to_frame("label").to_csv(ROOT / "data" / "processed" / "y.csv", index=False)

    with open(ROOT / "results" / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
