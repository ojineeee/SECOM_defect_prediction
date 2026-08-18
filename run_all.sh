#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data/raw
if [ ! -f data/raw/secom.data ]; then
  curl -sS -o data/raw/secom.data https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data
  curl -sS -o data/raw/secom_labels.data https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data
fi

pip install -q -r requirements.txt

cd src
python3 eda.py
python3 train.py
python3 anomaly_detection.py
python3 shap_analysis.py
python3 chronological_validation.py
python3 walk_forward_validation.py
python3 walk_forward_model_selection.py
python3 operational_metrics.py

echo "Done. See ../results/"
