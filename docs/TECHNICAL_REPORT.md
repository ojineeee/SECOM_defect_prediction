# 기술 상세 리포트

> 이 문서는 개발자/데이터 분석 담당자를 위한 상세 기술 문서입니다.
> 프로젝트 배경과 요약은 [메인 README](../README.md)를 먼저 봐주세요.

# SECOM 반도체 공정 불량 예측 (양불 판정 + 이상 탐지)

반도체 제조 공정에서 수집된 590개 센서 신호로 웨이퍼의 양/불량을 예측하고,
지도학습 분류와 약한 지도 피처 선택을 포함한 One-class 이상탐지를 비교한 프로젝트입니다.

- 데이터: [UCI SECOM Data Set](https://archive.ics.uci.edu/dataset/179/secom) (Kaggle에도 동일 데이터 공개됨)
- 샘플 수: 1,567개 (불량 104개, 6.64% — 심한 클래스 불균형)
- 피처 수: 원본 590개 + 타임스탬프 기반 파생변수 3개

## 왜 이 문제가 어려운가

- 피처가 전부 익명화되어 있어 도메인 지식으로 피처를 고를 수 없고, 순수 통계적 방법에 의존해야 합니다.
- 불량 비율이 6.64%에 불과해 Accuracy는 의미가 없습니다 (전부 "정상"으로 찍어도 93.4%). 그래서 **Recall(불량 검출률)과 PR-AUC**를 핵심 지표로 삼았습니다.
- 이 프로젝트의 목표는 "완벽한 모델"이 아니라 **불균형 데이터를 다루는 방법론을 제대로 적용하고, EDA에서 발견한 가설을 실제로 검증하며, 그 한계까지 정직하게 분석하는 것**입니다.

## 파이프라인

```
data/raw (원본) → 타임스탬프 파생변수 생성 → 결측치 처리 → 상수 피처 제거 → 스케일링 → 상위 피처 선택
   → [시간 파생변수 효과 검증] → [불균형 처리 전략 비교] → [모델 비교] → 최종 모델 held-out 평가
   → [비지도 이상탐지와 비교]
```

데이터 누수를 막기 위해 **train/test 분리를 가장 먼저** 수행하고, imputer·scaler·
feature selection·리샘플링(SMOTE)은 전부 `Pipeline` 안에 넣어 교차검증 fold의
train 부분에만 적용되도록 구성했습니다.

## 실행 방법

```bash
pip install -r requirements.txt
bash run_all.sh
```

또는 단계별로:
```bash
cd src
python3 eda.py                       # EDA + 전처리, results/figures/01~05
python3 train.py                     # 시간피처 검증 + 불균형 처리 비교 + 모델 비교 + 최종 평가, 06~09
python3 anomaly_detection.py         # One-class 이상탐지 비교, 10
python3 shap_analysis.py             # SHAP 설명력 분석, 11~13
python3 chronological_validation.py  # 시간순 단일 분할 재검증, 14~15
python3 walk_forward_validation.py   # walk-forward + 부트스트랩/Clopper-Pearson CI, 18
python3 walk_forward_model_selection.py  # walk-forward 기준 모델 선택 재검증, 20
python3 operational_metrics.py       # Recall@상위 N%, 오경보/1,000장, Precision@고정 Recall
```
모든 결과는 시드(42)를 고정해 **재실행해도 동일한 수치**가 나오도록 만들었습니다(재현성 확인 완료).

## EDA에서 발견한 특이점 (파생변수로 이어진 것들)

### 1. 시간 드리프트 — 불량률이 수집 기간 동안 고정되어 있지 않음
수집 기간(2008-07-19 ~ 2008-10-17, 약 3개월) 동안 불량률을 시간 순으로 보면:

| 구간 | 불량률 |
|---|---|
| 앞 절반 | 8.56% |
| 뒤 절반 | 4.72% |

50건 단위 rolling window로 보면 불량률이 0%~24% 사이를 오가며(`02b_defect_rate_drift.png`),
**시간이라는 변수가 불량 여부와 관련된 신호를 담고 있다**는 것이 뚜렷합니다. 다만 이 데이터만으로는
원인(공정 개선/조업 조건 변화/다른 요인)까지는 특정할 수 없어, "공정이 안정화됐다"는 인과적
해석은 하지 않고 상관관계로만 기록합니다. 이건 원본 피처만 봐서는 알 수 없고,
라벨 파일에 있던 타임스탬프를 뜯어봐야 나오는 발견이었습니다.

**→ 파생변수 생성**: `days_since_start`, `hour_of_day`, `day_of_week`를 만들어 실험한 결과,
`days_since_start`가 실제로 최종 모델(Logistic Regression)에 **채택**되었고 계수는 **-0.50**
(시간이 지날수록 불량 확률이 낮아지는 방향)으로, EDA에서 관찰한 드리프트 방향과 정확히
일치합니다. CV 기준 Recall도 0.374 → 0.387로 소폭 개선되어 최종 파이프라인에 포함했습니다
(`results/time_feature_comparison.csv`).

`days_since_start`의 기준일은 각 train split의 최소 timestamp로 정하고 같은 기준일을
validation/test에도 적용합니다. split별로 기준일을 다시 계산하면 같은 값이 서로 다른 날짜를
뜻하므로, 학습·추론 시간축이 달라지는 문제를 만들 수 있습니다.

> 참고: `hour_of_day`/`day_of_week`는 선택되지 않았습니다 — 요일별 불량률 차이(일요일 10.5%
> vs 화요일 4.3%)는 있었지만 그룹당 샘플이 190~253개뿐이라 노이즈일 가능성이 높다고 판단해,
> 실제로 모델도 이 두 피처는 유의미한 정보로 채택하지 않았습니다.

### 2. 결측치가 무작위가 아니라 "센서 그룹" 단위로 뭉쳐 있음
55% 이상 결측인 24개 컬럼의 센서 인덱스를 나열하면 무작위가 아니라 **연속된 인덱스 클러스터**로
뭉쳐 있습니다: `[109,110,111]`, `[157,158]`, `[244,245,246]`, `[292,293]`, `[382,383,384]`,
`[516,517,518]`, `[578,579,580,581]` 등 (총 11개 클러스터). 이는 개별 센서가 무작위로
고장난 게 아니라, **다채널 센서 모듈 하나가 특정 구간에서 로깅에 실패했을 가능성**을
시사합니다. 실제 fab라면 이 클러스터를 근거로 "어느 계측 서브시스템을 점검해야 하는지"를
좁혀서 제안할 수 있습니다.

### 3. 상위 예측 피처들 사이에도 중복(다중공선성)이 존재
Mutual information 상위 10개 피처 중에서도 `sensor_577`↔`sensor_573` (corr=0.958),
`sensor_541`↔`sensor_407` (corr=0.912)처럼 **거의 같은 정보를 담은 피처 쌍**이 발견됩니다.
즉 "정보량이 높다"고 나온 피처 10개가 실질적으로는 8개에 가까운 독립 정보라는 뜻이며,
이는 앞서 PCA에서 설명 분산이 낮게 나온 것(9.9%)과 함께, 이 데이터가 겉보기보다
**차원이 낮은 구조**를 가지고 있을 가능성을 뒷받침합니다.

## 진행 중 발견한 버그 2건 (투명하게 기록)

1. **타임스탬프 파싱 버그**: 라벨 파일의 날짜 필드가 큰따옴표로 감싸져 있어 처음엔 3개
   컬럼으로 잘못 나눠 읽었고, 그 결과 모든 타임스탬프가 `NaT`(파싱 실패)였습니다.
   이 때문에 위 시간 드리프트 발견 자체가 처음엔 불가능했습니다 — `src/data.py` 수정 후
   정상 파싱되는 것을 확인했습니다.
2. **피처 선택 비결정성**: `SelectKBest(score_func=mutual_info_classif)`에 `random_state`를
   고정하지 않아서, 코드를 재실행할 때마다 선택되는 피처와 최종 성능 지표가 달라지는
   재현성 문제가 있었습니다. `functools.partial`로 `random_state`를 고정해 두 번 연속
   실행 결과가 완전히 동일함을 확인했습니다.

## 불균형 처리 전략 비교 (RandomForest 고정, 5-fold CV)

| 전략 | Recall | Precision | PR-AUC |
|---|---|---|---|
| 리샘플링 없음 (baseline) | 0.000 | 0.000 | 0.194 |
| class_weight='balanced' | 0.024 | 0.150 | 0.173 |
| **SMOTE** | **0.036** | 0.086 | 0.149 |

리샘플링을 전혀 하지 않으면 모델이 불량을 단 한 건도 잡아내지 못합니다. **Recall을
우선시해야 하는 도메인이므로 SMOTE를 선택**했습니다.

## 모델 비교 (SMOTE + 시간 파생변수 포함, 5-fold CV)

| 모델 | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|
| **Logistic Regression** | **0.387** | 0.087 | 0.142 | 0.127 |
| SVM (RBF) | 0.120 | 0.173 | 0.135 | 0.149 |
| XGBoost | 0.072 | 0.100 | 0.083 | 0.153 |
| Random Forest | 0.036 | 0.086 | 0.051 | 0.149 |

가장 단순한 모델인 Logistic Regression이 Recall에서 압도적으로 앞섰습니다. 트리 기반
모델은 Precision/PR-AUC는 낫지만 Recall이 크게 떨어집니다 — 샘플이 적고(1,567개) 피처가
많은 상황에서 복잡한 모델이 과적합되기 쉽다는 정황과 일치합니다.

## 최종 모델 성능 (Logistic Regression + SMOTE + 시간 파생변수, held-out test 20%)

| 지표 | 값 |
|---|---|
| Recall (불량 검출률) | **0.381** |
| Precision | 0.098 |
| F1 | 0.155 |
| PR-AUC | 0.152 |

21건의 실제 불량 중 8건을 검출했습니다(`06_confusion_matrix.png`, `07_pr_curve.png`). Precision이
0.098이라는 것은, 8건을 검출하기 위해 정상 웨이퍼 약 74건을 함께 "불량 의심"으로 잘못 지목했다는
뜻이기도 합니다. **Threshold 조정 실험**(`08_threshold_sweep.png`)에서 확인되듯, threshold를
낮추면 Recall을 더 끌어올릴 수 있지만 Precision이 급격히 낮아지는 트레이드오프가 뚜렷합니다.
실제 운영에서는 이 threshold를 "불량 1건을 놓치는 비용 vs 정상을 재검사하는 비용"에 맞춰
조정해야 하며, 구체적인 Recall/Precision 조합은 아래 [운영 관점 지표](#운영-관점-지표-recall상위-n-오경보-빈도-precision고정-recall) 절에서 다룹니다.

## Feature Importance

로지스틱 회귀 계수 절댓값 기준 상위 피처: `sensor_277`, `sensor_576`, `sensor_415`,
`sensor_574`, `sensor_443` 등 (`09_feature_importance.png`). `days_since_start`도 선택된
피처 목록에 포함되어 있으나 절대 계수 크기(-0.50)로는 상위 15개 안에는 들지 않았습니다 —
즉 개별 센서만큼 강한 신호는 아니지만, 모델이 유의미하다고 판단해 채택한 보조 신호입니다.

## One-class 기반 이상탐지 vs 지도학습 비교

정상(Pass) 샘플만으로 "정상의 분포"를 학습시킨 뒤 불량을 이상치로 탐지하는 방식을
지도학습과 비교했습니다. 완전한 비지도 학습은 아닙니다 — 피처 선택(`SelectKBest` +
`mutual_info_classif`)은 라벨을 쓰므로 약한 지도 신호가 섞여 있습니다.

### 버그 수정 이력

최초 구현에는 두 가지 문제가 있었습니다.
1. **피처 선택을 정상(Pass) 샘플만 걸러낸 뒤 수행**했습니다. 이 상태에서는 `y`가 전부 0인
   상수 라벨이 되어 `mutual_info_classif`가 모든 피처에 대해 0에 가까운 점수를 반환합니다 —
   즉 사실상 무작위로 60개 피처를 고른 것과 다르지 않았습니다.
2. **`contamination`/`nu`에 실제 test 불량률(`y_train.mean()`)을 그대로 넣어** 실전에서는
   알 수 없는 정보를 모델에 흘려보내고 있었습니다.

수정 후에는 (1) 피처 선택은 정상+불량이 섞인 전체 train(`X_train, y_train`)에 대해 한 번만
수행하고, 그 결과로 선택된 피처 공간 위에서 이상탐지 모델 자체만 정상 샘플로 학습하며,
(2) `contamination`/`nu`는 실제 불량률을 모른다고 가정하고 일반적으로 쓰이는 보수적
기본값 5%(`CONTAMINATION_DEFAULT`)를 사용하도록 `src/anomaly_detection.py`를 다시
작성했습니다.

### 결과 (버그 수정 후) — 기본 설정 그대로 비교 (참고용)

| 방법 | Recall | Precision | PR-AUC |
|---|---|---|---|
| Isolation Forest | 0.048 | 0.048 | 0.057 |
| One-Class SVM | 0.048 | 0.024 | 0.055 |
| **지도학습 (Logistic Regression, threshold=0.5)** | **0.381** | 0.098 | 0.152 |

### 한계: 위 비교는 경고 예산이 다르다 (matched budget으로 재비교)

One-class 모델은 contamination=5%로 학습되어 test에서 대략 그 비율만큼 플래깅하는 반면,
지도학습은 기본 threshold(0.5)를 그대로 썼다 — 두 방식이 실제로 몇 장을 "의심 대상"으로
지목하는지 자체가 다르므로, 위 표의 Recall을 직접 비교하는 것은 불공정하다. 세 모델 모두
이상치/불량 점수를 기준으로 **테스트셋(314장) 상위 5%(16장)를 플래깅**했을 때로 다시
계산했다(`flag_top_fraction`, `FLAG_FRACTION=0.05`).

| 방법 | 검출 불량 | 상위 5%(16장) 플래깅 시 Recall | Precision |
|---|---:|---:|---:|
| 지도학습 (Logistic Regression) | **3/21건** | **0.143** | **0.188** |
| One-Class SVM | 1/21건 | 0.048 | 0.062 |
| Isolation Forest | 0/21건 | 0.000 | 0.000 |

동일 예산의 이 split에서는 지도학습이 가장 많은 불량을 검출했다(`10_anomaly_vs_supervised.png`는
이 matched-budget 결과를 시각화한 것). Isolation Forest는 기본 predict() 기준으로는 Recall 0.048이었지만,
점수 기준 상위 5%만 뽑으면 Recall이 0으로 떨어진다 — `contamination` 파라미터가 학습
자체의 하이퍼파라미터로도 작용해, `predict()`가 플래깅하는 정확한 지점과 "점수 상위 5%"가
정확히 일치하지 않기 때문이다. 다만 불량이 21건뿐인 단일 split에서 3건과 1건을 비교한 결과이므로
일반적인 우열로 단정하지 않는다. 이 데이터에서는 라벨을 활용한 분류 모델을 우선 후보로 두고,
One-class 이상탐지는 불량 라벨이 거의 없는 초기 공정의 보조 후보로 해석한다.
(EllipticEnvelope는 60차원 공분산 추정이 불안정해 결과가 신뢰할 수 없어 비교에서 제외했습니다.)

## 모델 설명력 분석 (SHAP, `src/shap_analysis.py`)

### 방법

- 최종 파이프라인(전처리 → SMOTE → LogisticRegression)에서 전처리 단계만 분리해 60차원 선택
  피처 공간으로 변환한 뒤, `shap.LinearExplainer(clf, X_train_transformed, feature_perturbation="correlation_dependent")`로 계산.
  `correlation_dependent` 옵션은 피처 간 독립을 가정하지 않고 실제 공분산을 반영 — 이 데이터셋은
  EDA에서 이미 다중공선성(`sensor_577`↔`sensor_573` corr 0.958 등)이 확인됐으므로 필수적인 선택.
- Background/설명 대상 모두 held-out test(314개)에 대해 계산.

### 전역 중요도: 계수 vs SHAP

| 방법 | 상위 15개 중 겹치는 피처 수 |
|---|---|
| \|coefficient\| vs SHAP mean\|value\| | **5 / 15** |

SHAP 기준 상위 5개: `sensor_132`(0.405), `sensor_103`(0.324), **`days_since_start`(0.305, 3위)**,
`sensor_344`(0.294), `sensor_416`(0.247). `days_since_start`는 계수 기준으로는 상위 15위 밖(계수
절댓값 0.50, 60개 피처 중 하위권)이었으나 SHAP 기준으로는 3위로 크게 상승 — 상관관계를 반영하지
않는 단순 계수 크기가 이 피처의 실제 기여도를 과소평가하고 있었음을 시사.

### 개별 사례 설명

held-out test 중 실제 Fail이고 모델도 Fail로 정확히 예측한 사례(test index 56, 예측 확률
97.56%)를 `shap.plots.waterfall`로 분해 (`13_shap_individual_example.png`).

### 재현성 참고

`shap.LinearExplainer`는 근사 없이 정확한 값을 계산하므로(선형모델이므로), 동일 입력에 대해
결정론적으로 동일한 SHAP 값을 반환합니다.

## 시간순 분할(Chronological Split) 검증 (`src/chronological_validation.py`)

### 동기

지금까지의 평가는 `train_test_split(..., stratify=y, random_state=42)`로 만든 랜덤 20% 홀드아웃
기준이었다. EDA에서 확인한 시간 드리프트(앞 절반 8.56% → 뒤 절반 4.72% 불량률)를 고려하면,
랜덤 분할은 과거·미래 샘플을 뒤섞어 평가하므로 실제 운영(과거로 학습 → 미래를 예측) 대비
성능을 낙관적으로 추정할 위험이 있다. 이를 직접 검증했다.

### 방법

- `np.argsort(ts)`로 전체 데이터를 시간순 정렬 후 앞 80% / 뒤 20%로 분할 (랜덤 아님, stratify 없음).
- 전처리·모델은 `train.py`의 최종 파이프라인과 완전히 동일 (`build_base_steps` + SMOTE + LogisticRegression).
- train/test 기간: train `2008-07-19 ~ 2008-10-02`, test `2008-10-02 ~ 2008-10-17`.
- train 불량률 6.94%(87/1253), test 불량률 5.41%(17/314).

### 결과

| 지표 | 랜덤 분할 (기존) | 시간순 분할 |
|---|---|---|
| Recall | 0.381 | **0.000** |
| Precision | 0.098 | 0.000 |
| F1 | 0.155 | 0.000 |
| PR-AUC | 0.152 | 0.084 |

Held-out 17건의 실제 불량 중 **단 한 건도 검출하지 못함** (`14_chronological_confusion_matrix.png`).
PR-AUC(threshold에 무관한 랭킹 지표)는 0.084로 완전히 0은 아니지만 랜덤 분할(0.152) 대비 크게
낮아, 확률 순위 자체도 저하됨을 시사한다.

Recall이 정확히 0(17건 중 0건)이라 부트스트랩(고정된 예측 벡터를 재표본추출)으로는 CI를 구성할
수 없다. 대신 "실제 불량 17건 중 몇 건을 맞혔는가"를 이항 비율(k=0, n=17)로 보고 Clopper-Pearson
정확 신뢰구간을 계산하면 **[0.0, 0.1951]** 이다. 다만 무작위 분할(불량 21건, 부트스트랩 CI)과
시간순 분할(불량 17건, Clopper-Pearson CI)은 테스트셋도 다르고 CI 계산 방식도 달라 두 구간의
"겹침 여부"를 엄밀한 통계 검정으로 보기는 어렵다 — 정확한 결론은 "무작위 분할보다 시간순
분할에서 현저히 낮은 성능이 관측됐으며, 미래 구간 일반화가 어렵다는 증거"라는 것이다.

### 해석

기본 비율 차이(6.94% → 5.41%)만으로는 이 정도의 성능 붕괴를 설명하기 어렵다. 더 유력한 설명은
**공정 드리프트로 인해 "불량을 만드는 센서 패턴" 자체가 시간에 따라 변했다는 것** — 즉 이 모델이
학습한 t1~t2 구간의 불량 패턴이 t2~t3 구간에는 더 이상 유효하지 않을 가능성이다. 랜덤 분할
평가는 이런 시간적 일반화 실패를 감지할 수 없다는 것이 이번 검증의 핵심 결론이다.

## Walk-forward 검증 및 부트스트랩 신뢰구간 (`src/walk_forward_validation.py`)

### 동기

위 시간순 단일 분할은 "8~10월 초로 학습 → 10월을 예측"이라는 딱 하나의 스냅샷이라, 성능 붕괴가
전체 기간의 보편적 특성인지 특정 구간에 국한된 현상인지 구분할 수 없었다. 또한 지금까지 보고한
모든 성능 수치(랜덤 분할 Recall 0.381 등)는 314개 test set(불량 21건) 위의 단일 실행 결과였고,
신뢰구간 없이는 이후 실험 간의 작은 차이가 실제 개선인지 노이즈인지 판단할 수 없었다.

### 방법

1. **부트스트랩 CI**: 기존 랜덤 분할·시간순 단일 분할의 테스트 예측 결과에 대해, 테스트셋을
   복원추출로 2,000회 리샘플링하여 Recall/Precision/F1/PR-AUC의 95% CI(2.5~97.5 백분위수)를 계산.
   불량이 0건인 리샘플은 recall/precision이 정의되지 않으므로 제외.
2. **Walk-forward**: `TimeSeriesSplit(n_splits=4)`로 시간순 정렬된 데이터에 확장 윈도우 fold를
   생성 (fold i는 처음부터 i번째 구간까지 학습, 그다음 구간을 평가). 전처리·모델은 기존과 동일.

### 결과 — 부트스트랩 CI

| 평가 | Recall (95% CI) | PR-AUC (95% CI) |
|---|---|---|
| 랜덤 분할 | 0.381 (0.167~0.588) | 0.152 (0.068~0.332) |
| 시간순 단일 분할 | 0.000 (Clopper-Pearson 0.000~0.195) | 0.084 (0.045~0.171) |

랜덤 분할의 Recall 신뢰구간이 0.17~0.59로 매우 넓어, 점 추정치(0.381) 자체가 상당한 불확실성을
내포한다. 시간순 분할은 고정된 예측 벡터에 대한 부트스트랩에서는 모든 재표본의 Recall이 0이지만,
이는 새로운 불량 표본에 대한 모수 불확실성을 반영하지 못한다. 따라서 보고에는 17건 중 0건 검출에
대한 Clopper-Pearson 구간 0.000~0.195를 사용한다.

### 결과 — Walk-forward (4 fold)

| Fold | 평가 구간 | n_test_fail | Recall | Recall CP 95% CI | PR-AUC | No-skill 기준선 | PR-AUC < 기준선 |
|---|---|---|---|---|---|---|---|
| 1 | 08-19 ~ 09-01 | 21 | 0.619 | 0.3844~0.8189 | 0.055 | 0.067 | **True** |
| 2 | 09-01 ~ 09-20 | 11 | 0.182 | 0.0228~0.5178 | 0.060 | 0.035 | False |
| 3 | 09-20 ~ 10-02 | 11 | 0.000 | 0.0~0.2849 | 0.036 | 0.035 | False |
| 4 | 10-02 ~ 10-17 | 17 | 0.000 | 0.0~0.1951 | 0.083 | 0.054 | False |

No-skill 기준선은 그 fold의 실제 불량 비율(prevalence)이다. PR-AUC가 이 값보다 낮으면 적어도
점 추정치 기준으로는 무작위 순위보다 낫다는 근거가 없다. **fold 1은 Recall(0.619)만 보면
가장 성능이 좋아 보이지만, PR-AUC(0.055)가 no-skill 기준선(0.067)보다 낮다** — 즉 이 구간에서는
고정 임계값이 우연히 많은 양성을 잡아냈을 뿐, 모델이 웨이퍼 간 위험도를 실제로 구별해내는
능력(랭킹 성능)은 확인되지 않는다. fold 2~4는 PR-AUC 점 추정치가 기준선을 상회하거나 근접하지만,
별도 신뢰구간이나 검정 없이 랭킹 우위를 단정할 수는 없다.

Recall만 보면 0.619 → 0.182 → 0.000 → 0.000으로 감소하는 패턴을 보인다 (`18_walk_forward_recall.png`).
다만 fold 1의 PR-AUC가 애초에 기준선 미달이었다는 점을 감안하면, "초반엔 실제로 잘 맞다가 점차
붕괴한다"기보다는 **이 모델의 랭킹 성능 자체가 이 시기 전반에 걸쳐 약했고, 그 위에서 임계값
기준 Recall이 fold 1에서 우연히 높게 나왔을 가능성**을 함께 고려해야 한다. 두 해석 모두
"재학습 없이 오래 쓰면 신뢰할 수 없다"는 결론으로 수렴하지만, Recall 하나만으로 fold별 성능
추이를 읽으면 실제보다 낙관적으로 해석할 위험이 있다는 것을 보여주는 사례다.

### 구현상 주의사항

fold 1처럼 학습 표본이 작은 초기 구간에서는 `SimpleImputer(strategy='median')`이 특정 결측
컬럼(`sensor_244` 등)에 대해 "전부 결측이라 중앙값을 계산할 수 없다"는 경고를 발생시켰다.
scikit-learn은 이런 컬럼을 0으로 대체해 학습을 계속하므로 파이프라인이 깨지지는 않지만, 가장
작은 fold의 해당 컬럼 값은 신뢰도가 낮다는 점을 감안해야 한다.

## 모델 선택 재검증 (`src/walk_forward_model_selection.py`)

### 동기

`train.py`의 모델 비교(Logistic Regression 선정)는 무작위 분할 5-fold CV로 진행됐다. 위
walk-forward 검증에서 무작위 분할이 시간적 일반화를 대변하지 못한다는 게 확인된 이상, 모델
선택 기준 자체도 walk-forward로 재검증해야 논리적으로 일관된다.

### 방법

`walk_forward_validation.py`와 동일한 `TimeSeriesSplit(n_splits=4)` fold에서, 4개 모델
(Logistic Regression, Random Forest, XGBoost, SVM RBF)을 동일 전처리 + SMOTE 조건으로 재학습.

### 결과

| 모델 | 무작위 분할 CV Recall | Walk-forward 평균 Recall |
|---|---|---|
| Logistic Regression | 0.387 | **0.2002** |
| SVM (RBF) | 0.120 | 0.0119 |
| XGBoost | 0.072 | 0.0357 |
| Random Forest | 0.036 | 0.0000 |

두 평가 방식 모두에서 순위가 Logistic Regression > (SVM/XGBoost 혼재) > Random Forest로
일관됨 (`20_model_selection_walk_forward.png`). 절대 수치는 walk-forward가 전반적으로 낮지만,
**순위 자체는 뒤집히지 않아** 원래의 모델 선택이 평가 방식에 상관없이 견고했음을 확인했다.

## 운영 관점 지표 (Recall@상위 N%, 오경보 빈도, Precision@고정 Recall) (`src/operational_metrics.py`)

### 동기

Recall·Precision·PR-AUC는 모델 간 비교에는 유용하지만, "이 모델을 켜두면 재검사를 얼마나 더
해야 하는가" 같은 운영 의사결정에는 바로 쓰기 어렵다. 무작위 분할 테스트셋(314건, 불량 21건) 위에서
이를 직접 계산했다.

### 방법

1. **Recall@상위 N% 플래깅**: `predict_proba`로 얻은 불량 확률 기준 내림차순 정렬 후, 상위
   N%(5/10/20/30%)를 "재검사 대상"으로 지정했을 때의 Recall과, 1,000 웨이퍼당 오경보(false
   positive) 건수를 계산.
2. **Precision@고정 Recall**: `results/threshold_sweep.csv`에서 목표 Recall(0.3/0.5/0.7) 이상을
   만족하는 threshold 중 가장 낮은 초과분을 찾아, 그 지점의 실제 Recall과 Precision을 보간 없이 조회.

### 결과

| 재검사 대상 | 검사 대상 수 | Recall | 오경보/1,000장 |
|---|---|---|---|
| 상위 5% | 16 | 0.1429 | 41.4 |
| 상위 10% | 31 | 0.1905 | 86.0 |
| 상위 20% | 63 | 0.2857 | 181.5 |
| 상위 30% | 94 | 0.4286 | 270.7 |

| 목표 Recall | 실제 Recall | Precision | Threshold |
|---|---|---|---|
| 0.3 | 0.333 | 0.100 | 0.55 |
| 0.5 | 0.619 | 0.078 | 0.25 |
| 0.7 | 0.714 | 0.081 | 0.20 |

Recall을 0.3대에서 0.7대로 올려도 Precision은 0.08~0.10 범위에서 거의 변하지 않는다 — 이 모델은
특정 임계값 구간에서 트레이드오프가 급격한 게 아니라, **전체 임계값 범위에서 Precision 자체가
구조적으로 낮다**. 결과는 `results/operational_metrics.json`에 저장된다.

### 한계: threshold도 test set에서 탐색했다

②(Precision@고정 Recall)는 `threshold_sweep.csv`, 즉 이 test set 위에서 목표 Recall을 만족하는
threshold를 찾은 결과다. 별도 validation set에서 threshold를 정하고 test에서는 그 threshold
하나만 평가하는 구조가 아니므로, test set이 여기서는 사실상 validation 역할도 겸하고 있다. 위
표는 "검증된 운영 threshold"가 아니라 **이 test set에서 관찰된 트레이드오프를 보여주는 탐색적
결과**로 읽어야 한다.

## 한계점

- 피처가 익명화되어 있어 도메인 기반 피처 엔지니어링에 한계가 있습니다 (시간 파생변수 정도가 유일하게 시도 가능했던 도메인 무관 파생변수).
- 불량 샘플이 104개뿐이라 test set(랜덤 분할 21개 / 시간순 분할 17개)의 결과가 몇 건의 예측에 따라 크게 흔들릴 수 있습니다 — 실제로 랜덤 분할 Recall의 95% CI가 0.17~0.59로 넓게 나옵니다.
- SMOTE는 합성 샘플을 생성하므로 실제 물리적으로 존재하지 않는 센서 조합을 만들어낼 수 있다는 한계가 있습니다.
- Walk-forward 검증 결과 Recall이 시간이 지날수록 0.62→0.18→0.00→0.00으로 단조 감소하는 것을 확인했습니다. 이 모델을 실제로 배포하려면 최소한 주기적 재학습, concept drift 감지, 혹은 시간에 강건한 피처 설계가 추가로 필요합니다.
- walk-forward의 초기 fold는 학습 표본이 작아(약 260~520개) 일부 결측 컬럼의 대체값 신뢰도가 낮습니다.
- 이번 실험들은 고정된 시드/분할 기준이며, 다른 시드로 반복하면 walk-forward의 구체적인 fold별 수치는 달라질 수 있습니다(다만 전반적인 감소 추세는 신뢰구간으로 뒷받침됨).
- 모든 신뢰구간은 테스트셋 재표본추출(부트스트랩) 또는 이항 신뢰구간(Clopper-Pearson)만 반영합니다. 랜덤 시드를 바꿔 모델을 재학습했을 때의 변동이나, 데이터 자체를 다시 수집했을 때의 변동까지는 정량화하지 않았습니다.
- walk-forward fold 1은 Recall(0.619)만 보면 가장 우수하지만 PR-AUC(0.055)가 no-skill 기준선(0.067)보다 낮습니다 — Recall 하나만으로 fold별 성능을 해석하면 랭킹 성능이 실제로는 약하다는 사실을 놓칠 수 있습니다.
- One-class 이상탐지도 완전한 비지도 학습은 아닙니다 — 피처 선택 단계가 라벨을 쓰는 `mutual_info_classif` 기준이라 약한 지도 신호가 섞여 있습니다.

## 프로젝트 구조

```
secom-defect-detection/
├── data/
│   ├── raw/            # 원본 (run_all.sh 실행 시 자동 다운로드)
│   └── processed/       # EDA 단계에서 정제된 X, y
├── src/
│   ├── data.py           # 데이터 로드
│   ├── eda.py            # EDA + 전처리 + 시각화 + 특이점 탐지
│   ├── train.py          # 시간피처 검증 + 불균형 처리 비교 + 모델 비교 + 최종 평가
│   ├── anomaly_detection.py  # 비지도 이상탐지 비교
│   ├── shap_analysis.py  # SHAP 기반 전역/개별 설명력 분석
│   ├── chronological_validation.py  # 시간순 단일 분할 재검증
│   ├── walk_forward_validation.py  # walk-forward 다중 fold + 부트스트랩/Clopper-Pearson 신뢰구간
│   ├── walk_forward_model_selection.py  # walk-forward 기준 모델 선택 재검증
│   └── operational_metrics.py  # Recall@상위 N%, 오경보/1,000장, Precision@고정 Recall
├── results/
│   ├── figures/          # 01~18 시각화 결과
│   └── *.json, *.csv     # 수치 결과
├── requirements.txt
└── run_all.sh
```
