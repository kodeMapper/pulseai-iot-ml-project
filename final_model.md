# Final Model Report: Predicting Maternal Health Risk

This document explains the complete process of building and finalizing a machine learning model to predict maternal health risks. The goal was to create a reliable model that prioritizes patient safety by minimizing the chances of misclassifying a high-risk patient.

---

### **Step 1: Understanding and Loading the Data**

**What we did:**
We started with the "Maternal Health Risk" dataset from the UCI Machine Learning Repository. This dataset contains real-world anonymous data about pregnant patients.

**Facts and Figures:**
*   **Dataset Size:** 1014 patient records.
*   **Features (The Clues):** We used 6 pieces of information to make predictions:
    1.  `Age`
    2.  `SystolicBP` (Systolic Blood Pressure)
    3.  `DiastolicBP` (Diastolic Blood Pressure)
    4.  `BS` (Blood Sugar levels)
    5.  `BodyTemp` (Body Temperature)
    6.  `HeartRate`
*   **Target (What we want to predict):** `RiskLevel` - categorized as 'low risk', 'mid risk', or 'high risk'.

**Why we did this:**
The first step in any machine learning project is to understand the data you're working with. It's like a detective getting to know the case file before starting an investigation.

---

### **Step 2: Preparing the Data for the Model**

Before a machine learning model can learn from data, the data needs to be cleaned and prepared.

**Action 1: Feature Scaling**
*   **What we did:** We applied `StandardScaler`. Imagine you have features in different units (like age in years and heart rate in beats per minute). Some models can get confused by the different scales.
*   **Why we did this (in simple terms):** Feature scaling puts all features on a level playing field. It's like converting different currencies to a single currency before comparing them. This helps the model learn more effectively without getting biased by features with larger numbers.

**Action 2: Handling Imbalanced Data with SMOTE**
*   **What we did:** We noticed that the dataset had an unequal number of patients in each risk category (e.g., more low-risk patients than high-risk). This is called "class imbalance." We used a technique called **SMOTE** (Synthetic Minority Over-sampling Technique).
*   **Why we did this (in simple terms):** If a model sees too many examples of one class, it can become lazy and just predict that class most of the time. SMOTE helps by creating new, artificial examples of the under-represented classes (like 'high risk'). This gives the model a more balanced diet of data to learn from, making it smarter at identifying all classes, not just the common ones.

---

### **Step 3: Choosing the Right Metric - Why Accuracy Isn't Enough**

**The Problem with "Accuracy":**
Accuracy tells you how many predictions the model got right overall. While that sounds good, it can be dangerously misleading in medical cases.

**A Simple Example:** Imagine a model that is 99% accurate at predicting a rare disease. But what if it achieves that by simply predicting "no disease" for everyone? It would be right 99% of the time but would miss every single person who actually has the disease.

**The Importance of "Recall" (Minimizing False Negatives):**
*   **False Negative:** This is when the model predicts a patient is "low risk" when they are actually "high risk". This is the worst possible mistake in our case, as it could lead to a lack of necessary medical care.
*   **Recall:** This metric answers the crucial question: **"Of all the patients who were truly high risk, how many did our model correctly identify?"**

**Our Goal:** We shifted our focus from just getting a high accuracy to getting the **highest possible recall for the 'high risk' class.** It's better to be overly cautious (and have some false alarms) than to miss a single patient in danger.

---

### **Step 4: Experimenting with Different Models**

We trained several different types of machine learning models to see which one performed best, especially on our key metric: **Recall**.

| Model                  | Overall Accuracy | High-Risk Recall |
| ---------------------- | ---------------- | ---------------- |
| Logistic Regression    | 63.5%            | 85%              |
| Decision Tree          | 82.8%            | 87%              |
| Random Forest          | 80.8%            | 87%              |
| Gradient Boosting      | 76.4%            | 83%              |
| Support Vector Machine | 67.5%            | 85%              |
| Gaussian Naive Bayes   | 61.1%            | 79%              |
| **XGBoost**            | **83.3%**        | **87%**          |

**Observation:**
The **XGBoost** model gave us the best combination of overall accuracy and high-risk recall.

---

### **Step 5: Finalizing the Model - Optimizing for Safety**

Even though XGBoost was already doing well, we wanted to push it further to be as safe as possible.

**What we did:**
We used a technique called `GridSearchCV` to fine-tune the XGBoost model. We specifically instructed it to **adjust its internal settings to maximize the recall of the 'high risk' class.**

**Final Model Performance:**
After tuning, our final XGBoost model produced the following results:

| Risk Level | Precision | **Recall** | F1-Score |
| ---------- | --------- | ---------- | -------- |
| **high**   | **85%**   | **87%**    | **86%**  |
| low        | 85%       | 80%        | 83%      |
| mid        | 80%       | 84%        | 82%      |

---

### **Conclusion**

Our final recommended model is an **XGBoost classifier, specifically tuned to maximize recall for high-risk patients.**

This model correctly identifies **87%** of all patients who are genuinely at high risk. While no model is perfect, this focus on recall ensures that we have built a system that prioritizes patient safety by significantly minimizing the chance of dangerous false negatives. The `pulseai.py` script now contains the code for this final, optimized model.
