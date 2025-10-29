import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/best_gradient_boosting_final.pkl')
scaler = joblib.load('models/best_scaler_final.pkl')

# Risk mapping
risk_map = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}

print("=" * 70)
print("MODEL VERIFICATION - Testing 3 Real Dataset Samples")
print("=" * 70)

# Three test samples from the actual dataset
samples = [
    {
        'name': 'Sample 1: LOW RISK',
        'Age': 21,
        'SystolicBP': 90,
        'DiastolicBP': 65,
        'BS': 7.5,
        'BodyTemp': 98.0,
        'HeartRate': 76,
        'Expected': 'Low Risk'
    },
    {
        'name': 'Sample 2: MEDIUM RISK',
        'Age': 30,
        'SystolicBP': 120,
        'DiastolicBP': 80,
        'BS': 6.9,
        'BodyTemp': 101.0,
        'HeartRate': 76,
        'Expected': 'Medium Risk'
    },
    {
        'name': 'Sample 3: HIGH RISK',
        'Age': 35,
        'SystolicBP': 85,
        'DiastolicBP': 60,
        'BS': 11.0,
        'BodyTemp': 102.0,
        'HeartRate': 86,
        'Expected': 'High Risk'
    }
]

correct = 0
total = len(samples)

for sample in samples:
    print(f"\n{sample['name']}")
    print("-" * 70)
    
    # Prepare input
    input_data = {
        'Age': sample['Age'],
        'SystolicBP': sample['SystolicBP'],
        'DiastolicBP': sample['DiastolicBP'],
        'BS': sample['BS'],
        'BodyTemp': sample['BodyTemp'],
        'HeartRate': sample['HeartRate']
    }
    
    # Print input
    print("Input Features:")
    for key, value in input_data.items():
        print(f"  {key:15s}: {value}")
    
    # Make prediction
    df = pd.DataFrame([input_data])
    scaled_features = scaler.transform(df)
    prediction = model.predict(scaled_features)[0]
    predicted_risk = risk_map[prediction]
    
    # Check result
    is_correct = predicted_risk == sample['Expected']
    if is_correct:
        correct += 1
        status = "✅ CORRECT"
    else:
        status = "❌ INCORRECT"
    
    print(f"\nExpected Risk : {sample['Expected']}")
    print(f"Predicted Risk: {predicted_risk} {status}")

print("\n" + "=" * 70)
print(f"VERIFICATION RESULT: {correct}/{total} predictions correct ({correct/total*100:.1f}%)")
print("=" * 70)
