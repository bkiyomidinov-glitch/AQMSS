# AMQSS AI Layer - Professional Market Environment Classifier

## Overview

This is a **professional-grade AI layer** built on top of the AMQSS (Adaptive Market Quality Score System) engine.

**Key Principles:**
- ✅ **No price direction prediction** — Classifies market *environment* quality only
- ✅ **Future-based learning** — Trains on forward returns & drawdowns (no look-ahead bias)
- ✅ **Time-based splits** — Chronological train/test, no shuffling, zero data leakage
- ✅ **Engine frozen** — AMQSS engine unchanged, modular AI layer only
- ✅ **Production-ready** — Proper scaling, model persistence, evaluation metrics

## Architecture

```
main.py (AMQSS Engine - Frozen)
    ↓
results/market_scores.csv
    ↓
[AI Layer]
    ├─ dataset_builder.py → Extracts features + future performance → data/dataset.csv
    ├─ trainer.py → Time-based split, RandomForest → models/model.pkl
    ├─ evaluator.py → ROC-AUC, PR-AUC, feature importance → models/metrics.json
    └─ predictor.py → Load model, classify market quality → AI probability
    ↓
main.py (updated)
    ├─ Calls predictor.py
    └─ Appends ai_high_quality_prob to results
```

## Components

### 1. **dataset_builder.py**
Builds a clean ML dataset from AMQSS output + price data.

**Features:**
- Aligns AMQSS timestamps with price candles
- Computes forward returns over N bars (future window)
- Computes max drawdown (risk metric)
- Generates binary labels: high-quality if `forward_return ≥ threshold` AND `max_drawdown ≥ threshold`
- Normalizes features to [0, 1]
- **Zero data leakage:** Uses only historical (past) data for features

**Output:** `data/dataset.csv` with columns:
- Features: `volatility_regime`, `true_liquidity`, `structure`, `momentum`, `volatility_level`, `current_atr`
- Targets: `forward_return`, `max_drawdown`, `label` (0/1)
- Metadata: `timestamp`, `price_index`

### 2. **trainer.py**
Time-based model training with proper data splits.

**Process:**
1. Loads `data/dataset.csv`
2. **Time-based split:** First 80% for training, last 20% for testing (NO SHUFFLING)
3. Scales features using `StandardScaler` (fit on train data only)
4. Trains `RandomForestClassifier`:
   - 100 trees
   - Max depth: 10
   - Balanced class weights (handles imbalance)
5. Saves to `/models`:
   - `model.pkl` — Trained model
   - `scaler.pkl` — Fitted scaler
   - `metadata.json` — Training info (dates, feature names, etc.)

### 3. **evaluator.py**
Comprehensive evaluation on **test set only**.

**Metrics:**
- **ROC-AUC:** Area under ROC curve
- **PR-AUC:** Area under precision-recall curve
- **Precision, Recall, F1:** Classification metrics
- **Specificity:** True negative rate
- **Confusion Matrix:** TP, TN, FP, FN
- **Feature Importance:** Random Forest feature contributions

**Outputs:**
- `models/metrics.json` — Full metric report
- `models/roc_curve.png` — ROC plot
- `models/pr_curve.png` — Precision-Recall plot
- `models/feature_importance.png` — Feature contributions

### 4. **predictor.py**
Inference engine: Load model and classify new AMQSS readings.

**Class:** `AMQSSPredictor`
```python
from predictor import load_predictor

predictor = load_predictor('models')
prob = predictor.predict({
    'volatility_regime': 5.0,
    'true_liquidity': 7.0,
    'structure': 8.0,
    'momentum': 6.0,
    'volatility_level': 4.0,
    'current_atr': 0.001
})
# Returns float in [0, 1]: probability of high-quality market
```

### 5. **main.py** (Updated)
AMQSS engine now calls the AI predictor.

**Changes:**
- Imports `predictor.py`
- After computing AMQSS scores, calls `predictor.predict()` with current features
- Appends `ai_high_quality_prob` to `market_scores.csv`
- Prints AI score and gracefully handles missing model (during initial training)

**Output:** `results/market_scores.csv` now includes:
- AMQSS metrics (as before)
- `ai_high_quality_prob` — AI probability (None until model trained)

## Setup & Usage

### 1. Install Dependencies
```bash
cd C:\Users\bkiyo\Desktop\Project.X
python -m pip install -r requirements.txt
```

### 2. Generate Initial AMQSS Results
```bash
python main.py
```
This creates `results/market_scores.csv` with raw AMQSS data.

### 3. Build Dataset (First Time)
```bash
python dataset_builder.py
```
Output: `data/dataset.csv` — Ready for ML training.

### 4. Train Model (First Time)
```bash
python trainer.py
```
Output: `models/model.pkl`, `models/scaler.pkl`, `models/metadata.json`

### 5. Evaluate Model
```bash
python evaluator.py
```
Output: `models/metrics.json` + evaluation plots.

### 6. Run Main with AI
```bash
python main.py
```
Now `ai_high_quality_prob` is populated in results!

### **One-Command Pipeline (Optional)**
```bash
python run_pipeline.py
```
Runs all steps: dataset_builder → trainer → evaluator → main.

## Key Design Decisions

### No Data Leakage
- **Features:** Built from *past* AMQSS metrics only (e.g., ATR, structure as of timestamp)
- **Labels:** Built from *future* price data (forward returns 20 bars ahead)
- **Train/Test:** Chronological split — no information from test set leaks to train

### Time-Based Split
- First 80% of data → Training
- Last 20% → Testing
- **No shuffling** — Preserves temporal structure, no look-ahead bias

### Market Environment (Not Price Direction)
- Model outputs: "Is market in good condition for trading?"
- **Not:** "Will price go up or down?"
- Uses stability, volatility, structure, liquidity as predictors
- Labels based on risk-adjusted returns, not direction

### Feature Normalization
- Each AMQSS metric (0-10 scale) → normalized to [0, 1]
- Fitted scaler saved with model for consistency at inference time

### Class Imbalance Handling
- `class_weight='balanced'` in RandomForest
- Precision-Recall curve preferred over ROC when classes are imbalanced

## Monitoring & Iteration

### Check Model Performance
```bash
cat models/metrics.json
```

### View Feature Importance
- Open `models/feature_importance.png` to see which AMQSS metrics matter most

### Retrain Periodically
When enough new data accumulates:
```bash
python dataset_builder.py  # Builds larger dataset
python trainer.py           # Retrains model
python evaluator.py         # Evaluates on test set
```

## Files & Structure

```
Project.X/
├── main.py                      # AMQSS engine (frozen)
├── dashboard.py                 # (Existing)
│
├── [AI Layer]
├── dataset_builder.py           # Feature + label generation
├── trainer.py                   # Model training
├── evaluator.py                 # Model evaluation
├── predictor.py                 # Inference engine
├── run_pipeline.py              # Orchestration
│
├── data/
│   └── dataset.csv              # ML training dataset
├── models/
│   ├── model.pkl                # Trained RandomForest
│   ├── scaler.pkl               # Feature scaler
│   ├── metadata.json            # Training metadata
│   ├── metrics.json             # Performance report
│   ├── roc_curve.png            # Evaluation plot
│   ├── pr_curve.png             # Evaluation plot
│   └── feature_importance.png   # Evaluation plot
├── results/
│   └── market_scores.csv        # AMQSS + AI output
└── requirements.txt             # Dependencies
```

## Constraints & Guarantees

| Constraint | Implementation |
|---|---|
| No price direction | Model only classifies market environment (0/1: bad/good) |
| Forward learning | Features from past, labels from future (20-bar window) |
| No data leakage | Time-based split, no shuffling, proper scaler fit |
| Engine frozen | main.py AMQSS logic unchanged, AI is pure overlay |
| Production-ready | Model persistence, scaling, proper evaluation metrics |

## Troubleshooting

**Q: "Model not found" error when running main.py?**
- A: Model hasn't been trained yet. Run `python trainer.py` first.

**Q: Metrics look too good/bad?**
- A: Check label distribution in `models/metrics.json`. Imbalanced data → misleading metrics.

**Q: Low ROC-AUC?**
- A: AMQSS features may not predict future performance well. Review forward return threshold or collect more data.

**Q: Feature importance shows one feature dominates?**
- A: Possible multicollinearity. Consider PCA or feature engineering in dataset_builder.

---

**Built with:**
- pandas, numpy — Data processing
- scikit-learn — RandomForest, scaling
- matplotlib — Evaluation plots

**Questions?** Check individual module docstrings or run with `--help` flag.
