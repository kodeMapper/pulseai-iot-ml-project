import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier


def main():
    df = pd.read_csv('iot_dataset_expanded.csv')
    df = df.drop_duplicates()

    X = df[['Temperature Data', 'ECG Data', 'Pressure Data']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'LogisticRegression': (
            LogisticRegression(max_iter=2000, multi_class='multinomial'),
            True
        ),
        'RandomForest': (RandomForestClassifier(n_estimators=500, random_state=42), False),
        'GradientBoosting': (GradientBoostingClassifier(random_state=42), False),
        'ExtraTrees': (ExtraTreesClassifier(n_estimators=500, random_state=42), False)
    }

    results = {}
    for name, (model, use_scaled) in models.items():
        if use_scaled:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"{name}: {acc:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name} ({results[best_name]:.4f})")

    if best_name == 'LogisticRegression':
        best_model = models[best_name][0]
        preds = best_model.predict(X_test_scaled)
    else:
        best_model = models[best_name][0]
        preds = best_model.predict(X_test)

    print("\nClassification report for best model:")
    print(classification_report(y_test, preds))


if __name__ == '__main__':
    main()
