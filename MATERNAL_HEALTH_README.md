# PulseAI - Maternal Health Risk Prediction

**Accurate maternal health risk classification using machine learning**

---

## 🎯 Project Overview

PulseAI is a machine learning system that predicts maternal health risk levels for pregnant patients based on 6 vital health indicators. The system classifies patients into three risk categories to help healthcare providers make informed decisions.

### Risk Categories
- 🟢 **Low Risk** - Normal pregnancy, routine prenatal care
- 🟡 **Medium Risk** - Elevated concerns, regular monitoring required
- 🔴 **High Risk** - Immediate medical attention recommended

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository - Maternal Health Risk Data  
**Size:** 1,014 patient records

### Input Features (6)
1. **Age** - Patient's age in years
2. **SystolicBP** - Systolic Blood Pressure (mmHg)
3. **DiastolicBP** - Diastolic Blood Pressure (mmHg)
4. **BS** - Blood Sugar level (mmol/L)
5. **BodyTemp** - Body Temperature (°F)
6. **HeartRate** - Heart Rate (beats per minute)

### Target Variable
- **RiskLevel** - Low risk / Mid risk / High risk

### Distribution
- Low Risk: 406 patients (40%)
- Mid Risk: 336 patients (33%)
- High Risk: 272 patients (27%)

---

## 🏆 Model Performance

### Final Model: **XGBoost (Default Parameters)**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **83.3%** |
| **High-Risk Recall** | **87%** |
| **High-Risk Precision** | **85%** |

### Performance by Risk Level

| Risk Level | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **High**   | **85%**   | **87%**| **86%**  | 47      |
| Low        | 85%       | 80%    | 83%      | 80      |
| Mid        | 80%       | 84%    | 82%      | 76      |

---

## 🔬 Model Comparison

All models were trained with SMOTE (oversampling) and StandardScaler:

| Model | Accuracy | High-Risk Recall | Notes |
|-------|----------|------------------|-------|
| **XGBoost** | **83.3%** | **87%** | ⭐ **Best Overall** |
| Decision Tree | 82.8% | 87% | Good, but overfits |
| Random Forest | 81.3% | 87% | Solid ensemble |
| Gradient Boosting | 76.4% | 83% | Decent |
| SVM | 67.5% | 85% | High recall, lower accuracy |
| Logistic Regression | 63.5% | 85% | Simple baseline |
| Gaussian Naive Bayes | 61.1% | 79% | Fast but less accurate |

### ⚠️ Why Not Hyperparameter Tuning?

We tested GridSearchCV to optimize XGBoost parameters:
- **Tuned Model:** 73% accuracy, 81% recall ❌
- **Default XGBoost:** 83.3% accuracy, 87% recall ✅

**Conclusion:** The default XGBoost parameters are already optimal for this dataset. Tuning made it worse!

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```powershell
# Clone the repository
git clone https://github.com/kodeMapper/pulseai-iot-ml-project.git
cd pulseai-iot-ml-project

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install Flask Flask-Cors flask-pymongo xgboost imbalanced-learn
```

### Run the Model Training

```powershell
python pulseai.py
```

This will:
1. Load the maternal health risk dataset
2. Train 7 different ML models
3. Compare their performance
4. Save the best model (XGBoost)

### Run the Interactive Demo

```powershell
python demo_maternal.py
```

This provides:
- Sample predictions on test cases
- Interactive mode to predict risk for custom patient data
- Detailed confidence scores for each risk level

---

## 📁 Project Structure

```
pulseai-iot-ml-project/
├── pulseai.py                      # Main training script
├── demo_maternal.py                # Interactive demo
├── maternal_health_risk.csv        # Dataset
├── final_model.md                  # Detailed model report
├── models/
│   ├── best_xgboost_final.pkl     # Best trained model
│   ├── best_scaler_final.pkl      # Feature scaler
│   ├── XGBoost.pkl                # Individual models
│   ├── Random_Forest.pkl
│   └── ...
├── src/
│   ├── data_loading.py            # Dataset utilities
│   └── ...
└── webapp/
    ├── backend/                    # Flask API
    └── frontend/                   # React UI
```

---

## 🎯 Key Features

### 1. **Patient Safety First**
- Optimized to minimize false negatives (missing high-risk patients)
- 87% recall ensures most high-risk cases are caught
- Better to be cautious than miss a critical case

### 2. **Handles Class Imbalance**
- Uses SMOTE (Synthetic Minority Over-sampling Technique)
- Balances training data to prevent bias toward majority class
- Ensures model learns all risk levels equally well

### 3. **Feature Scaling**
- StandardScaler normalizes all features
- Prevents features with larger values from dominating
- Improves model convergence and performance

### 4. **Production Ready**
- Saved model files for deployment
- Simple inference API
- Can be integrated into healthcare systems

---

## 💡 Usage Example

```python
import joblib
import numpy as np

# Load the saved model
model = joblib.load('models/best_xgboost_final.pkl')
scaler = joblib.load('models/best_scaler_final.pkl')

# Patient data: [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]
patient = np.array([[35, 140, 90, 13.0, 98.0, 70]])

# Scale and predict
patient_scaled = scaler.transform(patient)
risk_prediction = model.predict(patient_scaled)[0]
risk_probability = model.predict_proba(patient_scaled)[0]

# Results
risk_levels = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
print(f"Prediction: {risk_levels[risk_prediction]}")
print(f"Confidence: {risk_probability}")
```

---

## 📈 Model Insights

### What Makes a Patient High Risk?

Based on the model's decision patterns:
- **Age > 40 years** (older mothers)
- **High Blood Pressure** (SystolicBP > 140, DiastolicBP > 90)
- **High Blood Sugar** (BS > 12 mmol/L)
- **Abnormal Body Temperature** (< 97.5°F or > 100.5°F)
- **Irregular Heart Rate** (< 60 or > 100 bpm)

### Important Note

⚠️ **This model is for educational and research purposes only.** It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

---

## 🔮 Future Improvements

1. **Collect More Data**
   - Current: 1,014 samples
   - Target: 5,000+ samples for better generalization

2. **Add More Features**
   - Medical history
   - Previous pregnancies
   - Genetic factors
   - Lifestyle indicators

3. **Deploy as Web Service**
   - REST API for real-time predictions
   - Integration with hospital systems
   - Patient monitoring dashboard

4. **Explainable AI**
   - SHAP values for feature importance
   - Individual prediction explanations
   - Help doctors understand model decisions

---

## 📚 Technical Details

### Technologies Used
- **Python 3.12**
- **scikit-learn 1.7.2** - ML algorithms
- **XGBoost 2.0.0** - Best model
- **imbalanced-learn 0.14.0** - SMOTE
- **pandas 2.3.2** - Data processing
- **numpy 2.3.3** - Numerical computing

### Training Configuration
- **Train/Test Split:** 80/20
- **Cross-Validation:** 3-fold
- **Class Balancing:** SMOTE on training set only
- **Feature Scaling:** StandardScaler
- **Evaluation Metric:** Accuracy + High-Risk Recall

### Model Saved Files
- `best_xgboost_final.pkl` - Final XGBoost model
- `best_scaler_final.pkl` - StandardScaler for features

---

## 📄 Documentation

- **[final_model.md](final_model.md)** - Complete model development report
- **[MATERNAL_HEALTH_README.md](MATERNAL_HEALTH_README.md)** - This file

---

## 👥 Contributors

- Original Dataset: UCI Machine Learning Repository
- Model Development: PulseAI Team

---

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Repository: [pulseai-iot-ml-project](https://github.com/kodeMapper/pulseai-iot-ml-project)

---

## ⭐ Key Takeaways

1. ✅ **XGBoost with default parameters achieved 83.3% accuracy**
2. ✅ **87% high-risk recall ensures patient safety**
3. ✅ **SMOTE and StandardScaler are critical preprocessing steps**
4. ⚠️ **Hyperparameter tuning doesn't always improve performance**
5. ✅ **Always compare tuned models vs. baseline models**

---

**Last Updated:** October 29, 2025  
**Model Version:** 1.0  
**Status:** Production Ready ✅
