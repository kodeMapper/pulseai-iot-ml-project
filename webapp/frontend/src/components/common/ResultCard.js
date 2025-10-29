import React from 'react';
import Button from './Button';
import './ResultCard.css';

const ResultCard = ({ riskLevel, confidence, inputData, onNewAssessment, safetyOverride, safetyReason }) => {
  // Determine risk styling
  const getRiskConfig = (level) => {
    const configs = {
      'low': {
        label: 'Low Risk',
        className: 'risk-low',
        icon: '✓',
        message: 'The vital signs indicate low risk. Continue regular monitoring.'
      },
      'mid': {
        label: 'Medium Risk',
        className: 'risk-medium',
        icon: '⚠',
        message: 'Moderate risk detected. Consider additional monitoring and consultation.'
      },
      'high': {
        label: 'High Risk',
        className: 'risk-high',
        icon: '!',
        message: safetyOverride 
          ? 'CRITICAL: One or more vital signs are dangerously abnormal. Seek immediate emergency medical attention.'
          : 'High risk detected. Immediate medical attention is recommended.'
      }
    };
    return configs[level?.toLowerCase()] || configs['mid'];
  };

  const riskConfig = getRiskConfig(riskLevel);
  const confidencePercent = (confidence * 100).toFixed(1);

  // Format input data for display
  const formatInputData = () => {
    return [
      { label: 'Age', value: `${inputData.age} years`, unit: 'years' },
      { label: 'Systolic BP', value: inputData.systolic_bp, unit: 'mmHg' },
      { label: 'Diastolic BP', value: inputData.diastolic_bp, unit: 'mmHg' },
      { label: 'Blood Sugar', value: inputData.blood_sugar, unit: 'mmol/L' },
      { label: 'Body Temperature', value: inputData.body_temp, unit: '°F' },
      { label: 'Heart Rate', value: inputData.heart_rate, unit: 'bpm' }
    ];
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="result-card">
      <div className="result-header">
        <h2 className="result-title">Risk Assessment Result</h2>
      </div>

      {/* Risk Badge */}
      <div className={`result-risk-badge ${riskConfig.className}`}>
        <span className="result-risk-icon" aria-hidden="true">{riskConfig.icon}</span>
        <span className="result-risk-label">{riskConfig.label}</span>
      </div>

      {/* Risk Message */}
      <p className="result-message">{riskConfig.message}</p>

      {/* Safety Override Warning */}
      {safetyOverride && (
        <div className="result-safety-warning">
          <div className="safety-warning-header">
            <span className="safety-warning-icon" aria-hidden="true">🚨</span>
            <strong>Critical Values Detected</strong>
          </div>
          <p className="safety-warning-text">
            {safetyReason}
          </p>
          <p className="safety-warning-action">
            These values fall outside safe medical ranges. This assessment was automatically 
            flagged as HIGH RISK. Please seek emergency medical care immediately.
          </p>
        </div>
      )}

      {/* Confidence Score */}
      <div className="result-confidence">
        <div className="result-confidence-header">
          <span className="result-confidence-label">Confidence Score</span>
          <span className="result-confidence-value">{confidencePercent}%</span>
        </div>
        <div className="result-confidence-bar-container">
          <div 
            className="result-confidence-bar"
            style={{ width: `${confidencePercent}%` }}
            role="progressbar"
            aria-valuenow={confidencePercent}
            aria-valuemin="0"
            aria-valuemax="100"
            aria-label={`Confidence score: ${confidencePercent}%`}
          />
        </div>
      </div>

      {/* Input Summary */}
      <div className="result-inputs">
        <h3 className="result-inputs-title">Input Summary</h3>
        <div className="result-inputs-grid">
          {formatInputData().map((item, index) => (
            <div key={index} className="result-input-item">
              <span className="result-input-label">{item.label}</span>
              <span className="result-input-value">
                {item.value} <span className="result-input-unit">{item.unit}</span>
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="result-actions">
        <Button variant="primary" onClick={onNewAssessment} fullWidth>
          New Assessment
        </Button>
        <Button variant="secondary" onClick={handlePrint} fullWidth>
          Print Result
        </Button>
      </div>

      {/* Disclaimer */}
      <div className="result-disclaimer">
        <p>
          <strong>Medical Disclaimer:</strong> This is an AI-assisted prediction tool and should not replace professional medical advice. 
          Always consult with healthcare professionals for proper diagnosis and treatment.
        </p>
      </div>
    </div>
  );
};

export default ResultCard;
