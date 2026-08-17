"""SECOM 원본 데이터 로드."""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_raw():
    X = pd.read_csv(RAW_DIR / "secom.data", sep=r"\s+", header=None)
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

    labels_raw = pd.read_csv(
        RAW_DIR / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label", "timestamp"],
    )
    y = labels_raw["label"].map({-1: 0, 1: 1})  # 0=정상(pass), 1=불량(fail)
    timestamp = pd.to_datetime(labels_raw["timestamp"], dayfirst=True)

    return X, y, timestamp


if __name__ == "__main__":
    X, y, ts = load_raw()
    print("X shape:", X.shape)
    print("라벨 분포:\n", y.value_counts(normalize=True))
