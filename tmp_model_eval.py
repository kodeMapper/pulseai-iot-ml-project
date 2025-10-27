import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def evaluate_models():
    df = pd.read_csv('iot_dataset_expanded.csv')
    X = df[['Temperature Data', 'ECG Data', 'Pressure Data']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    def evaluate(name, estimator, use_scaler=False):
        if use_scaler:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', estimator)
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            bal = balanced_accuracy_score(y_test, preds)
            cv = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
        else:
            estimator.fit(X_train, y_train)
            preds = estimator.predict(X_test)
            acc = accuracy_score(y_test, preds)
            bal = balanced_accuracy_score(y_test, preds)
            cv = cross_val_score(estimator, X, y, cv=5, scoring='accuracy').mean()
        results.append((name, acc, bal, cv))

    evaluate('LogReg_bal', LogisticRegression(max_iter=5000, class_weight='balanced'), use_scaler=True)
    evaluate('SVM_rbf_bal', SVC(kernel='rbf', class_weight='balanced', probability=True), use_scaler=True)
    evaluate('KNN15', KNeighborsClassifier(n_neighbors=15), use_scaler=True)
    evaluate('RandomForest_bal', RandomForestClassifier(n_estimators=500, max_depth=8, class_weight='balanced', random_state=42))
    evaluate('ExtraTrees_bal', ExtraTreesClassifier(n_estimators=500, max_depth=8, class_weight='balanced', random_state=42))
    evaluate('GradientBoosting', GradientBoostingClassifier(random_state=42))
    evaluate('XGBoost_tuned', XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42
    ))
    evaluate('LightGBM_bal', LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight='balanced',
        random_state=42
    ))

    for name, acc, bal, cv in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:15s} | Test Acc: {acc:.4f} | Balanced Acc: {bal:.4f} | CV: {cv:.4f}")


if __name__ == '__main__':
    evaluate_models()
