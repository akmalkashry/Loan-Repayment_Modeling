#Loan Default Prediction using Machine Learning

## 📌 Project Overview

This project builds a machine learning system to predict whether a borrower will fully repay a loan or default.

The goal is to support financial institutions in improving credit risk assessment at the point of loan application, enabling better underwriting decisions and reducing potential loan losses.

---

## 📊 Dataset

The dataset consists of three main sources:

- Loan application data (`loan.csv`)
- Payment history data (`payment.csv`)
- Underwriting variables (`clarity_underwriting_variables.csv`)

### Final dataset:
- 46,006 originated loans used for modeling
- 133 engineered features after preprocessing and encoding

---

## 🎯 Problem Statement

Financial institutions need to accurately assess borrower risk before issuing loans.

This project aims to build a classification model that predicts whether a loan will be fully repaid using historical lending behavior and underwriting signals.

---

## ⚙️ Data Processing & Feature Engineering

Key steps performed:

- Merged loan, payment, and underwriting datasets
- Aggregated payment-level data into loan-level features
- Created key business features:
  - repayment_ratio
  - payment_success_rate
  - days_to_fund
- Removed target leakage variables (e.g. repayment-related fields)
- Handled missing values and outliers
- Encoded categorical variables for machine learning models

---

## 🧹 Data Validation & Cleaning

To ensure data quality and model reliability:

- Checked for invalid values (negative loans, extreme APR, etc.)
- Detected and reviewed outliers using statistical thresholds
- Analyzed missing data patterns (process-driven vs random missingness)
- Applied appropriate imputation strategies:
  - Numeric → filled with 0
  - Categorical → filled with "Unknown"

---

## 🤖 Modeling Approach

Two models were developed:

### 1. Logistic Regression (Baseline)
- Interpretable and fast
- Suitable for regulatory environments
- Provides benchmark performance

### 2. XGBoost Classifier (Advanced Model)
- Gradient boosting algorithm
- Captures non-linear relationships
- Handles feature interactions effectively

---

## 📈 Model Evaluation

Models were evaluated using:

- AUC-ROC (ranking ability)
- Precision (correct positive predictions)
- Recall (ability to detect defaults)
- F1-score (balance between precision and recall)

---

## 🏆 Results

### Logistic Regression
- AUC: **0.9767**
- Precision: **0.9925**
- Recall: **0.9756**
- F1-score: **0.9840**

### XGBoost
- AUC: **0.9791**
- Precision: **0.9919**
- Recall: **0.9763**
- F1-score: **0.9840**

---

## 📊 Key Insights

- Both models achieved excellent predictive performance (AUC ~0.97)
- Logistic Regression performs nearly as well as XGBoost
- The dataset contains strong predictive signals for loan repayment behavior
- Important drivers include:
  - Fraud-related indicators
  - Lead type (customer acquisition channel)
  - APR (interest rate)

---

## 💡 Business Impact

This model can help financial institutions:

- Identify high-risk borrowers early
- Reduce loan default exposure
- Improve underwriting decision-making
- Enable data-driven risk-based pricing strategies

---

## 🧠 Key Learnings

- Careful feature engineering significantly improves model performance
- Leakage detection is critical in financial modeling
- Simple models can perform nearly as well as complex models in structured financial datasets
- Business context is as important as model accuracy

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
