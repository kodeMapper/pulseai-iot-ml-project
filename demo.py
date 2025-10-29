"""
Demo script to test PulseAI model with sample predictions
"""

import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from advanced_features import AdvancedFeatureEngineer
import joblib

def load_best_model():
    """Load the best performing model"""
    model_path = 'models/corrected_best_model.pkl'
    scaler_path = 'models/corrected_scaler.pkl'
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✓ Model and scaler loaded successfully!")
    
    return model, scaler

def make_prediction(model, scaler, feature_engineer, temperature, ecg, pressure):
    """Make a prediction for given patient vitals"""
    # Create input dataframe with raw features
    input_df = pd.DataFrame({
        'Patient ID': [999],
        'Temperature Data': [temperature],
        'ECG Data': [ecg],
        'Pressure Data': [pressure],
        'Target': [0],  # Placeholder
        'Augmentation_Method': ['Live']
    })
    
    # Apply feature engineering (same as training)
    engineered_df = feature_engineer.engineer_all_features(input_df)
    
    # Drop non-feature columns
    feature_cols = [col for col in engineered_df.columns 
                   if col not in ['Patient ID', 'Target', 'Augmentation_Method']]
    X = engineered_df[feature_cols]
    
    # Scale the data
    scaled_data = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0]
    
    # Map prediction to risk level
    risk_levels = {0: "Low Risk 🟢", 1: "Medium Risk 🟡", 2: "High Risk 🔴"}
    
    return risk_levels[prediction], probability, prediction

def display_sample_data():
    """Display some sample data from the dataset"""
    print("\n📊 Sample Data from Dataset:")
    print("=" * 60)
    
    # Try to load the dataset
    if os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv')
        print(df.head(10).to_string(index=False))
        print(f"\nDataset shape: {df.shape}")
        print(f"Features: {', '.join(df.columns.tolist())}")
    else:
        print("Dataset not found.")

def main():
    print("=" * 60)
    print("🏥 PulseAI - IoT Health Monitoring System Demo")
    print("=" * 60)
    
    # Load model
    model, scaler = load_best_model()
    
    # Display sample data
    display_sample_data()
    
    print("\n" + "=" * 60)
    print("🔮 Making Sample Predictions")
    print("=" * 60)
    
    # Test cases with different vital signs
    test_cases = [
        {"name": "Patient A (Normal)", "temp": 36.5, "ecg": 75, "pressure": 120},
        {"name": "Patient B (Mild Alert)", "temp": 37.8, "ecg": 95, "pressure": 140},
        {"name": "Patient C (High Alert)", "temp": 39.2, "ecg": 120, "pressure": 170},
        {"name": "Patient D (Low Vitals)", "temp": 35.5, "ecg": 55, "pressure": 90},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Temperature: {case['temp']}°C")
        print(f"   ECG: {case['ecg']} bpm")
        print(f"   Pressure: {case['pressure']} mmHg")
        
        risk_level, probabilities, pred_class = make_prediction(
            model, scaler, 
            case['temp'], case['ecg'], case['pressure']
        )
        
        print(f"   → Prediction: {risk_level}")
        print(f"   → Confidence: Low={probabilities[0]:.1%}, Med={probabilities[1]:.1%}, High={probabilities[2]:.1%}")
    
    print("\n" + "=" * 60)
    print("✨ Interactive Prediction Mode")
    print("=" * 60)
    print("Enter patient vitals for real-time prediction (or 'q' to quit)")
    
    while True:
        print("\n")
        temp_input = input("Enter Temperature (°C) [or 'q' to quit]: ")
        if temp_input.lower() == 'q':
            break
            
        try:
            temperature = float(temp_input)
            ecg = float(input("Enter ECG (bpm): "))
            pressure = float(input("Enter Blood Pressure (mmHg): "))
            
            risk_level, probabilities, pred_class = make_prediction(
                model, scaler, temperature, ecg, pressure
            )
            
            print(f"\n🏥 Diagnosis: {risk_level}")
            print(f"📊 Confidence Breakdown:")
            print(f"   - Low Risk:    {probabilities[0]:.1%}")
            print(f"   - Medium Risk: {probabilities[1]:.1%}")
            print(f"   - High Risk:   {probabilities[2]:.1%}")
            
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thank you for using PulseAI!")
    print("=" * 60)

if __name__ == "__main__":
    main()
