# Student-Dropout-Risk-Early-Warning-System

## Project Overview

Student dropout remains a significant challenge for educational institutions. Students experiencing academic disengagement, burnout, mental health challenges, financial stress, and low motivation are more likely to leave their studies before completing their programs.

This project develops a machine learning-based early warning system that predicts student dropout risk using academic, mental health, financial, lifestyle, and support-related indicators. The objective is to help educational institutions identify at-risk students earlier and prioritize intervention efforts such as counseling, academic support, mentoring, and financial assistance. 【1-b7dbf1】

---

## Business Problem

Educational institutions often identify struggling students only after academic performance has already declined. An early warning system can help institutions:

- Identify students at risk sooner
- Improve intervention effectiveness
- Increase student retention
- Enhance student well-being support
- Reduce dropout rates

【1-b7dbf1】

---

## Dataset

Source: Kaggle (Harpartap Singh)

The dataset contains:

- 800 student records
- Academic indicators
- Mental health indicators
- Lifestyle factors
- Financial stress indicators
- Student support indicators

Target Variable:

```text
Dropout_Risk
```

Classes:

```text
Yes = At Risk of Dropping Out
No  = Not At Risk of Dropping Out
```

Dataset distribution:

- High Risk: 437 students (54.6%)
- Low Risk: 363 students (45.4%)

【1-b7dbf1】

---

## Repository Structure

```text
Student-Dropout-Risk-Early-Warning-System/

├── src/
│   └── Project scripts and reusable code
│
├── notebooks/
│   ├── Carlito_Cabalsi_Capstone.ipynb
│   └── Carlito_Cabalsi_Technical_Presentation.ipynb
│
├── data/
│   ├── Raw/
│   │   └── student_burnout_dropout_dataset_2.csv
│   │
│   └── Processed/
│       ├── data_dictionary.csv
│       ├── student_dropout_clean.csv
│       └── student_dropout_model_ready.csv
│
├── models/
│   └── Model artifacts
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Methodology

### Data Preparation

- Removed Student_ID
- Target encoding
- Missing value strategy to prevent data leakage
- Categorical encoding
- Train-test split
- Feature engineering

### Feature Engineering

Two engineered features were created:

#### Academic Engagement Index

Combines:

- Attendance Percentage
- Study Hours Per Day
- Previous GPA

#### Mental Health Index

Combines:

- Stress Level
- Anxiety Score
- Financial Stress Score

These engineered features became some of the strongest predictors of dropout risk. 【1-b7dbf1】

---

## Key Findings

### Strongest Risk Factors

The analysis found the following factors to be the strongest predictors of student dropout risk:

1. Burnout Level
2. Academic Engagement Index
3. Attendance Percentage
4. Mental Health Index
5. Motivation Score

【1-b7dbf1】

### Burnout Impact

High-risk classification by burnout level:

| Burnout Level | High Risk |
|--------------|-----------|
| Low | 23.8% |
| Medium | 63.2% |
| High | 92.0% |

Burnout was identified as the strongest individual predictor of dropout risk. 【1-b7dbf1】

---

## Models Evaluated

The following classification models were compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

【1-b7dbf1】

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| Logistic Regression | 70.00% | 72.41% | 72.41% | 72.41% |
| Decision Tree | 59.38% | 61.96% | 65.52% | 63.69% |
| Random Forest | 68.13% | 70.45% | 71.26% | 70.86% |
| XGBoost | 68.13% | 70.00% | 72.41% | 71.19% |

【1-b7dbf1】

### Best Overall Model

**Logistic Regression**

- Accuracy: 70.00%
- Precision: 72.41%
- Recall: 72.41%
- F1 Score: 72.41%

The model achieved the strongest overall performance and was selected as the primary model. 【1-b7dbf1】

### Best ROC-AUC

**Random Forest**

- ROC-AUC: 0.778

Random Forest demonstrated the strongest overall ability to distinguish between high-risk and low-risk students across varying decision thresholds. 【1-b7dbf1】

---

## Explainability and Fairness

### Explainability

Random Forest feature importance and SHAP analysis identified:

- Burnout Level
- Attendance Percentage
- Motivation Score
- Academic Engagement
- Mental Health factors

as the most influential contributors to predictions. 【1-b7dbf1】

### Fairness Review

The model was reviewed across:

- Gender
- Age Group
- Family Income Bracket
- Counseling Access

Findings showed relatively small differences across gender and age groups, while larger differences were observed across income levels and counseling access categories. 【1-b7dbf1】

---

## Responsible Use

This model should be used as a decision-support tool to assist educators and student-support teams.

The model should **not** be used as the sole basis for decisions affecting individual students. Human review and intervention remain essential. 【1-b7dbf1】

---

## Future Improvements

Potential enhancements include:

- Larger multi-institution datasets
- Longitudinal student tracking
- Hyperparameter



## Versioning

Version 1.0
- Logistic Regression baseline deployment

Rollback Plan:
- Revert to previous GitHub commit if deployment issues occur.


## Monitoring

Monitor:

- Prediction volume
- Error rate
- Model drift
- High-risk prediction percentage
