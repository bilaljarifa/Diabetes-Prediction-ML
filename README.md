# Diabetes Prediction using Machine Learning

## Project Overview
This project aims to predict whether a patient has diabetes based on medical diagnostic measurements.

The problem is treated as a Binary Classification task:
- 0 → No Diabetes
- 1 → Diabetes

## Dataset
- Source: Pima Indians Diabetes Dataset (Kaggle)
- 768 patient records
- 8 medical features
- 1 target variable (Outcome)

## Data Preprocessing
- Replaced medically impossible zeros with missing values
- Applied Median Imputation
- Performed Train/Test split (80/20)
- Applied StandardScaler for feature scaling

## Model Used
- Logistic Regression

## Results
- Accuracy: 70.8%
- ROC-AUC: 0.81
- Recall (Diabetes class): 50%

## Key Insight
While the model performs well overall, recall for diabetic patients can be improved since false negatives are critical in healthcare applications.

## How to Run
```bash
python Diabetes_Prediction.py

---

# 🎯 Why This Matters

A project without README = looks beginner  
A project with structured README = looks professional  

---

Now tell me:

Do you want to keep it simple like this  
or  
Make it stronger with model comparison tomorrow?