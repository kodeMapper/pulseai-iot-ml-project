# Frontend Product Requirements Document (PRD)
## PulseAI Maternal Health Risk Predictor

**Version:** 1.0  
**Last Updated:** October 30, 2025  
**Status:** Active Development

---

## 1. Product Overview

### Purpose
Web-based interface for healthcare providers to predict maternal health risk levels using ML model with 86.7% accuracy and 94.5% high-risk recall.

### Target Users
- Doctors and nurses in prenatal care
- Hospital staff performing patient intake
- Healthcare administrators monitoring risk trends

---

## 2. Core Features

### 2.1 Patient Input Form
**Priority:** P0 (Critical)

**Requirements:**
- 6 input fields for vital signs:
  - Age (18-50 years, integer)
  - Systolic BP (70-180 mmHg, integer)
  - Diastolic BP (40-120 mmHg, integer)
  - Blood Sugar (6-19 mmol/L, decimal)
  - Body Temperature (96-106°F, decimal)
  - Heart Rate (50-120 bpm, integer)
- Real-time validation with error messages
- Clear labels with units
- Submit button (disabled until valid)
- Reset button

**UI/UX:**
- Clean, medical-professional aesthetic
- Large, touch-friendly inputs for tablets
- Auto-focus on first field
- Tab navigation support

---

### 2.2 Risk Prediction Display
**Priority:** P0 (Critical)

**Requirements:**
- Color-coded risk levels:
  - 🟢 **Low Risk:** Green (#4CAF50) - "Routine prenatal care"
  - 🟡 **Medium Risk:** Yellow (#FFC107) - "Enhanced monitoring recommended"
  - 🔴 **High Risk:** Red (#F44336) - "Immediate medical attention required"
- Display confidence level (model probability)
- Show all input values for verification
- Medical action recommendations
- Print-friendly format

**UI/UX:**
- Large, clear result card
- Icon + color + text for accessibility
- Animated transition on result load
- Copy result to clipboard button

---

### 2.3 Model Performance Dashboard
**Priority:** P1 (High)

**Requirements:**
- Display current model metrics:
  - Overall Accuracy: 86.7%
  - High-Risk Recall: 94.5%
  - False Negatives: 3/55
  - Model Type: Gradient Boosting Classifier
- Last model update timestamp
- Link to detailed documentation
- Model version number

**UI/UX:**
- Collapsible info panel
- Tooltip explanations for metrics
- Badge/tag design

---

### 2.4 Patient History (Optional)
**Priority:** P2 (Medium)

**Requirements:**
- Save predictions to MongoDB
- View last 10 predictions (table view)
- Filter by risk level
- Export to CSV
- Basic search by patient ID

---

## 3. Technical Requirements

### 3.1 Framework & Libraries
- **React 19.2.0** - UI framework
- **Axios** - HTTP client for API calls
- **Material-UI or Tailwind CSS** - Component library
- **React Hook Form** - Form validation
- **Chart.js** (optional) - Visualizations

### 3.2 API Integration
- **Endpoint:** `POST http://localhost:5000/predict`
- **Request Format:**
```json
{
  "Age": 25,
  "SystolicBP": 120,
  "DiastolicBP": 80,
  "BS": 7.5,
  "BodyTemp": 98.6,
  "HeartRate": 76
}
```
- **Response Format:**
```json
{
  "predicted_class": "Low Risk",
  "confidence": 0.92,
  "model_info": {
    "accuracy": "86.7%",
    "recall": "94.5%"
  }
}
```

### 3.3 Performance
- Initial load: < 2 seconds
- Prediction response: < 500ms
- Mobile responsive (320px - 1920px)
- Browser support: Chrome, Firefox, Safari, Edge (latest 2 versions)

---

## 4. Non-Functional Requirements

### 4.1 Security
- Input sanitization (prevent XSS)
- HTTPS in production
- CORS policy configured
- No PHI stored in localStorage (use sessionStorage)

### 4.2 Accessibility
- WCAG 2.1 Level AA compliance
- Screen reader support
- Keyboard navigation
- High contrast mode support

### 4.3 Error Handling
- Network error messages
- Invalid input feedback
- API timeout handling (30s)
- Graceful degradation if model unavailable

---

## 5. User Flows

### 5.1 Primary Flow: Risk Assessment
1. User opens application
2. User fills in 6 vital sign fields
3. User clicks "Predict Risk"
4. System validates inputs
5. System calls backend API
6. System displays color-coded result + recommendations
7. User can print/save result or start new assessment

### 5.2 Error Flow: Invalid Input
1. User enters invalid value (e.g., age = 200)
2. Field shows red border + error message
3. Submit button remains disabled
4. User corrects input
5. Error clears, submit button enables

---

## 6. UI Mockup Structure

```
┌─────────────────────────────────────────┐
│  🏥 PulseAI Maternal Health Predictor   │
│  Accuracy: 86.7% | Recall: 94.5%        │
├─────────────────────────────────────────┤
│                                         │
│  Patient Vital Signs Input:             │
│  ┌─────────────┐  ┌──────────────┐     │
│  │ Age (years) │  │ Systolic BP  │     │
│  └─────────────┘  └──────────────┘     │
│  ┌─────────────┐  ┌──────────────┐     │
│  │Diastolic BP │  │ Blood Sugar  │     │
│  └─────────────┘  └──────────────┘     │
│  ┌─────────────┐  ┌──────────────┐     │
│  │ Body Temp   │  │  Heart Rate  │     │
│  └─────────────┘  └──────────────┘     │
│                                         │
│  [Predict Risk]  [Reset]                │
│                                         │
├─────────────────────────────────────────┤
│  📊 Risk Assessment Result:             │
│  ┌───────────────────────────────────┐  │
│  │  🟢 LOW RISK                      │  │
│  │  Confidence: 92%                  │  │
│  │  Recommendation: Routine care     │  │
│  └───────────────────────────────────┘  │
│  [Print Result]  [New Assessment]       │
└─────────────────────────────────────────┘
```

---

## 7. Success Metrics

### 7.1 User Engagement
- Average session duration: > 2 minutes
- Prediction completion rate: > 90%
- Return user rate: > 30% (monthly)

### 7.2 Performance
- API success rate: > 99%
- Average prediction time: < 500ms
- Zero data loss incidents

### 7.3 User Satisfaction
- System Usability Scale (SUS) score: > 75
- Net Promoter Score (NPS): > 50
- Critical bug reports: < 2 per month

---

## 8. Out of Scope (Future Versions)

- Multi-language support
- Patient authentication system
- Real-time IoT device integration
- Advanced analytics dashboard
- Mobile native apps (iOS/Android)
- Telemedicine video integration
- Electronic Health Record (EHR) integration

---

## 9. Development Milestones

| Milestone | Deliverable | Timeline |
|-----------|-------------|----------|
| M1 | Basic input form + validation | Week 1 |
| M2 | API integration + result display | Week 2 |
| M3 | Model info dashboard + styling | Week 3 |
| M4 | Error handling + accessibility | Week 4 |
| M5 | Testing + bug fixes | Week 5 |
| M6 | Production deployment | Week 6 |

---

## 10. Dependencies

### External
- Backend API must be running (`localhost:5000`)
- MongoDB connection (for history feature)
- Gradient Boosting model trained and saved

### Internal
- Design system/style guide
- API documentation
- Test data for validation

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| API downtime | High | Show cached model info, queue requests |
| Invalid predictions | Critical | Add confidence threshold (>70%) |
| Slow network | Medium | Loading states, timeout handling |
| Browser compatibility | Low | Polyfills, feature detection |

---

**Approval:**
- [ ] Product Manager
- [ ] Tech Lead
- [ ] UI/UX Designer
- [ ] Medical Advisor

**Next Steps:** Begin M1 development after approval
