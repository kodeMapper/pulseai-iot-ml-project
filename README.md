# PulseAI – IoT Health Monitoring (ML)

**Status**: Phase 1 Complete ✅ | **Best Accuracy**: 52.76% | **Target**: 85%

Classify patient health risk as **Low**, **Medium**, or **High** using IoT-derived vitals (Temperature, ECG, Pressure). This project implements a comprehensive machine learning pipeline with data augmentation, feature engineering, traditional ML, deep learning, and advanced ensemble techniques.

## 🎯 Project Status

**Phase 1 Complete** - All planned tasks (1.1-1.5) finished with the following results:

- ✅ **Data Augmentation**: 150 → 663 samples (+10% accuracy improvement)
- ✅ **Feature Engineering**: 3 → 64 features (0% improvement - curse of dimensionality)
- ✅ **Traditional ML**: Logistic Regression 52.26%
- ✅ **Deep Learning**: MLP, 1D-CNN, Deep MLP (underperformed at 50.25%)
- ✅ **Advanced Ensemble**: Extra Trees 52.76% ⭐ **BEST MODEL**
- ✅ **Hyperparameter Optimization**: Optuna (51.76% - slight overfit)

**Total Improvement**: +9.76% over baseline (43% → 52.76%)  
**Gap to Target**: 32.24 percentage points remaining

📄 **Full Report**: See [`PHASE1_FINAL_REPORT.md`](reports/PHASE1_FINAL_REPORT.md)

## 📁 Repository Contents

### Core Files
- `dataset.csv` — Original dataset (150 samples, 3 features)
- `dataset_augmented.csv` — Augmented dataset (663 samples)
- `dataset_engineered.csv` — Augmented + 64 engineered features
- `PROGRESS_SUMMARY_FINAL.md` — Quick progress overview
- `PROJECT_ROADMAP.md` — Original improvement plan

### Models & Training
```
models/
├── ensemble_best_model.pkl          # Extra Trees (52.76%) ⭐
├── ensemble_scaler.pkl              # StandardScaler
├── corrected_best_model.pkl         # Logistic Regression (52.26%)
├── mlp_model.keras                  # Deep learning models
├── *_metadata.json                  # Training results & metrics
```

### Source Code
```
src/
├── corrected_training.py            # Traditional ML training
├── advanced_ensemble.py             # CatBoost, LightGBM, XGBoost, Stacking
├── deep_learning_models.py          # MLP, 1D-CNN, Deep MLP
├── hyperparameter_optimization.py   # Optuna Bayesian optimization
├── data_augmentation.py             # SMOTE + Gaussian noise
├── advanced_features.py             # Feature engineering pipeline
└── predictor.py                     # Inference module
```

### Reports & Analysis
```
reports/
├── PHASE1_FINAL_REPORT.md           # Comprehensive Phase 1 report
├── EXECUTIVE_SUMMARY.md             # High-level overview
├── classification_reports.txt       # Per-model detailed results
├── *.png                            # Visualizations
```

## Dataset

Each row represents a single observation for a patient.

Columns:
- `Sl.No` (int): serial number/index (not used for modeling)
- `Patient ID` (int): anonymized patient identifier
- `Temperature Data` (numeric): temperature reading (unit depends on source, assumed °C)
- `ECG Data` (numeric): ECG-derived feature
- `Pressure Data` (numeric): blood pressure related value (assumed mmHg)
- `Target` (int): encoded label of patient condition

Label encoding (assumed):
- `0` → Low
- `1` → Medium
- `2` → High

Note: If your labeling differs, adjust the mapping in the notebook where predictions are interpreted.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ and pip
- Virtual environment recommended

### Installation

```powershell
# Create and activate virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Using the Best Model (Extra Trees - 52.76%)

```python
import joblib
import numpy as np

# Load best model and scaler
model = joblib.load('models/ensemble_best_model.pkl')
scaler = joblib.load('models/ensemble_scaler.pkl')

# Prepare input (Temperature, ECG, Pressure + 61 engineered features)
# Use dataset_engineered.csv structure
sample_data = np.array([[...]])  # 64 features

# Scale and predict
scaled_data = scaler.transform(sample_data)
prediction = model.predict(scaled_data)

# 0 = Low Risk, 1 = Medium Risk, 2 = High Risk
risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
print(f"Predicted: {risk_labels[prediction[0]]}")
```

### Training from Scratch

```powershell
# Run complete pipeline (Tasks 1.1-1.5)
python src/corrected_training.py          # Traditional ML
python src/deep_learning_models.py        # Neural networks
python src/advanced_ensemble.py           # Best models
python src/hyperparameter_optimization.py # Optimization
```

## 📊 Model Performance

| Model | Test Accuracy | CV Score | Status |
|-------|---------------|----------|--------|
| **Extra Trees** | **52.76%** | 50.83% | ⭐ Best |
| Logistic Regression | 52.26% | 52.56% | ✅ Good |
| Stacking Ensemble | 52.26% | - | ✅ Good |
| LightGBM | 51.26% | 53.20% | ✅ OK |
| Random Forest | 51.26% | 51.91% | ✅ OK |
| MLP (Deep Learning) | 50.25% | 50.00% | ⚠️ Underperformed |

### Per-Class Performance (Extra Trees)

| Class | Precision | Recall | F1-Score | Samples |
|-------|-----------|--------|----------|---------|
| Low Risk | 77% | 39% | 52% | 69 |
| Medium Risk | 43% | 82% | 56% | 68 |
| High Risk | 67% | 35% | 46% | 62 |

**Key Issue**: Model biased toward Medium Risk predictions. Low and High risk classes need improvement.

## 🔬 Technical Approach

### Phase 1 Pipeline
1. **Data Augmentation** (SMOTE + Gaussian Noise)
   - 150 real samples → 663 samples
   - Balanced class distribution
   - +10% accuracy improvement

2. **Feature Engineering** (64 features)
   - Polynomial features (Temp², Temp³)
   - Interaction features (Temp×ECG, Temp×Pressure)
   - Statistical features (Z-scores, percentiles)
   - Domain features (Critical_Vitals, Multiple_Abnormalities)
   - **Result**: 0% improvement (curse of dimensionality)

3. **Model Development**
   - Traditional ML: Logistic Regression, SVM, Decision Tree
   - Ensemble: Random Forest, Extra Trees, Gradient Boosting
   - Deep Learning: MLP, 1D-CNN
   - Advanced: CatBoost, LightGBM, XGBoost
   - Meta-Learning: Stacking, Voting

4. **Optimization**
   - Bayesian optimization with Optuna (50 trials/model)
   - Cross-validation (10-fold stratified)
   - Early stopping and regularization

## 📈 Key Findings

### What Worked ✅
- **Data augmentation** (+10% impact) - Most effective intervention
- **Tree-based ensembles** (Extra Trees, Random Forest) - Consistent 51-53%
- **Proper evaluation** (199 test samples vs original 11)

### What Didn't Work ❌
- **Feature engineering** (0% impact) - Too many features for dataset size
- **Deep learning** (-2% impact) - Requires 1000s of samples
- **Hyperparameter optimization** (-1% impact) - Hit performance ceiling

### Root Cause
Fundamental limitation: **Only 150 real samples**. Synthetic augmentation helps but has limits.

## 💡 Recommendations

### Critical Priority
1. **Collect 500-1000 real samples** (Expected: +15-20% improvement)
2. **Consult medical domain experts** for feature validation

### If continuing with current data
3. Focus on class imbalance handling (SMOTE variants, class weights)
4. Simplify to 5-15 most important features
5. Optimize decision thresholds per class

### Realistic Expectations
- **Current ceiling**: 52-58% (with perfect tuning)
- **With 500 samples**: 70-80% achievable
- **To reach 85%**: Need 1000-2000 samples + domain features

## 📄 Documentation

- **[PHASE1_FINAL_REPORT.md](reports/PHASE1_FINAL_REPORT.md)** - Complete Phase 1 analysis
- **[PROGRESS_SUMMARY_FINAL.md](PROGRESS_SUMMARY_FINAL.md)** - Quick overview
- **[MODEL_IMPROVEMENT_PLAN.md](MODEL_IMPROVEMENT_PLAN.md)** - Original roadmap
- **[EXECUTIVE_SUMMARY.md](reports/EXECUTIVE_SUMMARY.md)** - High-level summary

## 🤝 Contributing

This project is in research phase. Key areas for contribution:
- Data collection (highest priority)
- Domain-specific feature engineering
- Alternative problem formulations (e.g., binary classification)

## 📧 Contact

For questions about Phase 1 results or future collaboration, please open an issue.

---

**Project Status**: Phase 1 Complete ✅ | Phase 2 blocked on data collection

## Project structure

```
.
├─ IotFile.ipynb
├─ dataset.csv
└─ iot_dataset.csv
```

## Tips and troubleshooting (Windows)

- Virtual environment won’t activate: Allow PowerShell scripts for the current user.
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- `jupyter` not found: Install it with `pip install jupyter` in the active virtual environment.
- Plot rendering issues: Re-run the cell that sets plotting backend (if any), or restart the kernel and `Run All`.

## Next steps / roadmap

- Add a training script (`train.py`) and export the best model with `joblib`
- Add unit tests and a simple CLI for inference
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Feature engineering and scaling for algorithms that are sensitive to feature scales

## License

No license has been specified for this repository. If you plan to share or reuse, consider adding a license (e.g., MIT).

## Acknowledgements

- IoT health monitoring concept and classic ML baselines (LogReg, NB, DT, SVM)
- Built with Python, pandas, scikit-learn, seaborn, matplotlib, and Jupyter Notebook
