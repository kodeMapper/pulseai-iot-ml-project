# PulseAI – IoT Health Monitoring (ML)

Classify patient health risk as Low, Medium, or High using IoT-derived vitals (ECG, Temperature, Pressure) and classic ML models.

This repository contains a Jupyter Notebook that explores the data, trains multiple classifiers (Logistic Regression, Gaussian Naive Bayes, Decision Tree, and SVM), and evaluates them using accuracy, classification reports, confusion matrices, and comparison plots.

## Contents

- `IotFile.ipynb` — main notebook with EDA, model training, and evaluation
- `dataset.csv` — tabular dataset (sample IoT readings)
- `iot_dataset.csv` — duplicate of `dataset.csv` (kept for convenience/compatibility)

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

## Quickstart (Windows / PowerShell)

Prerequisites:
- Python 3.9+ (recommended) and pip

Create and activate a virtual environment, then install dependencies:

```powershell
py -3 -V
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install jupyter pandas numpy scikit-learn seaborn matplotlib
```

Launch Jupyter and open the notebook:

```powershell
jupyter notebook
```

Then open `IotFile.ipynb` and Run All cells.

## What the notebook does

1. Data loading and preprocessing
   - Reads `dataset.csv` (or `iot_dataset.csv`)
   - Drops/ignores `Sl.No`
   - Basic checks for nulls, class balance
2. Exploratory Data Analysis (EDA)
   - Histograms/KDEs, bar charts
   - Correlation heatmap and summary statistics
3. Model training
   - Logistic Regression, Gaussian Naive Bayes, Decision Tree, SVM
   - Train/validation split and per-model accuracy
4. Evaluation and comparison
   - Classification report (precision/recall/F1)
   - Confusion matrix visualizations
   - Side-by-side score comparison
5. Simple inference
   - Example prediction for a user-provided measurement

## Example: Predict from a new measurement (inside the notebook)

Assuming the trained model variable is `best_model` and the feature order is `[Patient ID, Temperature Data, ECG Data, Pressure Data]`:

```python
# Example single sample (Patient ID, Temperature, ECG, Pressure)
sample = [[1, 36.5, 85, 120]]
pred = best_model.predict(sample)[0]
label_map = {0: "Low", 1: "Medium", 2: "High"}
print("Predicted condition:", label_map.get(int(pred), pred))
```

If your notebook names the trained model differently (e.g., `svm_model`, `dt_model`), use that variable instead.

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
