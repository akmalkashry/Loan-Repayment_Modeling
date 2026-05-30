!pip install xgboost

import os
#location folder
os.chdir(r'C:\Users\data')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb

//Styles
sns.set(style="whitegrid", palette="muted", font_scale=1)

//Data Preparation
# Parsing dates immediately ensures can do time-based calculations later without manual conversion issues.
loan = pd.read_csv('loan.csv', parse_dates=['applicationDate', 'originatedDate'])
payment = pd.read_csv('payment.csv', parse_dates=['paymentDate'])
clarity = pd.read_csv('clarity_underwriting_variables.csv', low_memory=False)

// Standardize column names for consistency
# Standardize column names by:
#     - Stripping extra spaces
#     - Lowercasing
#     - Replacing spaces and dots with underscores
#     This makes future referencing cleaner and avoids typos.
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '_')
    return df
loan = clean_column_names(loan)
payment = clean_column_names(payment)
clarity = clean_column_names(clarity)

// Remove duplicate rows to ensure data integrity
# Duplicates can skew summaries (like total payments) and lead to inaccurate model features.
loan.drop_duplicates(subset='loanid', inplace=True)
payment.drop_duplicates(subset=['loanid', 'installmentindex', 'paymentdate'], inplace=True)

// Summarize payment data at loan level
# Convert payment records into loan-level features so that can merge with loan.csv.
payment_summary = payment.groupby('loanid').agg(
    total_paid=('paymentamount', 'sum'),
    total_principal_paid=('principal', 'sum'),
    total_fees_paid=('fees', 'sum'),
    num_payments=('paymentamount', 'count'),
    num_success_payments=('paymentstatus', lambda x: (x == 'Checked').sum()),
    num_failed_payments=('paymentstatus', lambda x: (x == 'Rejected').sum())
).reset_index()

// Merge data
# -------------------------------------------------
# Step 1: Merge loan data with payment summaries
# -------------------------------------------------
# Using left join to keep all loans, even those without payment history (e.g., defaults or cancellations).
loan_pay = loan.merge(payment_summary, on='loanid', how='left')
# -------------------------------------------------
# Step 2: Merge clarity underwriting data
# -------------------------------------------------
# Using left join, some loans may not have Clarity data due to underwriting flow differences.
loan_full = loan_pay.merge(clarity, left_on='clarityfraudid', right_on='underwritingid', how='left')

// Business Metric Generation
# -------------------------------------------------
# Step 1: Ensure date fields are correctly formatted
# -------------------------------------------------
# Technical Note:
# - Converting to datetime ensures accurate date calculations.
# - errors="coerce" converts invalid date strings to NaT instead of breaking the process.
# Business Value:
# - Dates are critical for calculating metrics like funding speed and repayment timelines.
# - Proper date handling avoids misleading time-based KPIs.
loan_full['applicationdate'] = pd.to_datetime(loan_full['applicationdate'], errors='coerce')
loan_full['originateddate'] = pd.to_datetime(loan_full['originateddate'], errors='coerce')
# -------------------------------------------------
# Step 2: Funding Speed (days_to_fund)
# -------------------------------------------------
# Business Value:
# - Measures operational efficiency in disbursing funds.
# - Longer funding times may indicate additional verification or higher perceived risk.
# - Can be used to identify process bottlenecks in the loan approval pipeline.
loan_full['days_to_fund'] = (loan_full['originateddate'] - loan_full['applicationdate']).dt.days
# -------------------------------------------------
# Step 3: Repayment Ratio
# -------------------------------------------------
# Business Value:
# - A core loan performance metric.
# - Repayment ratio >= 1 → fully repaid or overpaid.
# - Repayment ratio < 1 → partially repaid or defaulted.
# - Useful for EDA, but should be excluded from predictive modeling to avoid leakage.
loan_full['repayment_ratio'] = loan_full['total_paid'] / loan_full['loanamount']
# -------------------------------------------------
# Step 4: Full Repayment Flag
# -------------------------------------------------
# Business Value:
# - Provides a clean binary target variable for repayment risk prediction models.
# - Aligns with underwriting objectives: separate high vs low risk customers.
loan_full['is_fully_paid'] = (loan_full['repayment_ratio'] >= 1).astype(int)
# -------------------------------------------------
# Step 5: Payment Success Rate
# -------------------------------------------------
# Business Value:
# - Measures repayment consistency over the loan term.
# - Customers with higher success rates are statistically less likely to default.
# - Can be monitored post-origination to trigger early intervention for at-risk accounts.
loan_full['payment_success_rate'] = loan_full['num_success_payments'] / loan_full['num_payments']
# -------------------------------------------------
# Step 6: Customer Funnel Segmentation
# -------------------------------------------------
# Business Value:
# - Originated loans → primary dataset for repayment risk modeling.
# - Non-originated loans → supports marketing funnel analysis, conversion optimization, 
#   and pre-approval targeting strategies.
loan_originated = loan_full[loan_full['originated'] == True].copy()
loan_not_originated = loan_full[loan_full['originated'] == False].copy()
print(f"✅ Originated loans: {len(loan_originated)}")
print(f"❌ Non-originated loans: {len(loan_not_originated)}")
# ✅ Originated loans: 46006
# ❌ Non-originated loans: 531421

// Validation & Data Quality Checks
# -------------------------------------------------
# Step 1: Impossible Value Checks
# -------------------------------------------------
# Business Value:
# - Ensures data aligns with realistic lending scenarios.
# - Catches potential data entry mistakes or unusual cases worth manual review.
# - Helps avoid misleading signals in modeling (e.g., an APR of 500% skewing predictions).
def check_invalid_values(df):
    issues = {}
    issues['negative_loan_amount'] = (df['loanamount'] < 0).sum()
    issues['zero_or_missing_loan_amount'] = (df['loanamount'] <= 0).sum()
    issues['negative_apr'] = (df['apr'] < 0).sum()
    issues['apr_over_100'] = (df['apr'] > 100).sum()  # Over 100% APR is usually a data or business rule anomaly
    issues['negative_repayment_ratio'] = (df['repayment_ratio'] < 0).sum()
    issues['extreme_repayment_ratio'] = (df['repayment_ratio'] > 5).sum()  # Unlikely someone pays 5x the loan
    return issues
invalid_counts = check_invalid_values(loan_originated)
print("\n🚨 Invalid Value Summary:")
for k, v in invalid_counts.items():
    print(f"{k}: {v}")
# -------------------------------------------------
# Step 2: Outlier Detection
# -------------------------------------------------
# Business Value: 
# - Outliers can disproportionately affect model training.
# - Example: Extremely high APR values could bias the model to over-predict defaults.
# - Identifying outliers allows for business-driven handling (e.g., cap, remove, or keep as 'exception' cases).
def detect_outliers(df, cols, threshold=3):
    outlier_summary = {}
    for col in cols:
        if col in df.select_dtypes(include=np.number).columns:
            mean, std = df[col].mean(), df[col].std()
            outliers = ((df[col] - mean).abs() > threshold * std).sum()
            outlier_summary[col] = outliers
    return outlier_summary
outliers_found = detect_outliers(
    loan_originated, 
    ['loanamount', 'apr', 'repayment_ratio', 'payment_success_rate']
)
print("\n🔍 Outlier Summary (>3 std from mean):")
for col, count in outliers_found.items():
    print(f"{col}: {count} outliers")
# -------------------------------------------------
# Step 3: Missing Value Analysis
# -------------------------------------------------
# Business Value:
# - Identifies features with high missingness for imputation or removal.
# - Helps distinguish between random missingness (can impute) vs process-driven missingness 
#   (may be predictive in itself, e.g., missing fraud check for returning customers).
missing_summary = loan_originated.isnull().mean().sort_values(ascending=False) * 100
print("\n📊 Missing Data in Originated Loans (Top 10 features):")
print(missing_summary.head(10))
# -------------------------------------------------
# Step 4: Basic Missing Value Handling
# -------------------------------------------------
# Business Value:
# - Keeps categorical distributions intact without dropping rows.
# - Avoids bias from removing missing rows entirely, which could exclude important cases.
# Fill numeric columns with 0 (safe for metrics like 'total_fees_paid' where 0 means no fees paid)
numeric_cols = loan_originated.select_dtypes(include=np.number).columns
loan_originated[numeric_cols] = loan_originated[numeric_cols].fillna(0)
# Fill categorical columns with 'Unknown' to preserve category integrity
categorical_cols = loan_originated.select_dtypes(exclude=np.number).columns
loan_originated[categorical_cols] = loan_originated[categorical_cols].fillna('Unknown')
# -------------------------------------------------
# Step 5: Final Sanity Checks
# -------------------------------------------------
print("\n✅ Final Sanity Check After Cleaning:")
print("Negative loan amounts:", (loan_originated['loanamount'] < 0).sum())
print("Negative APR:", (loan_originated['apr'] < 0).sum())
print("Negative repayment ratio:", (loan_originated['repayment_ratio'] < 0).sum())
# -------------------------------------------------
# Step 6: Save Clean Dataset for Modeling
# -------------------------------------------------
loan_originated.to_csv('loan_originated_clean.csv', index=False)
print("\n🎯 Data Cleaning & Validation Complete — Ready for Exploration & Visualization.")
#🚨 Invalid Value Summary:
#negative_loan_amount: 0
#zero_or_missing_loan_amount: 0
#negative_apr: 0
#apr_over_100: 45964
#negative_repayment_ratio: 0
#extreme_repayment_ratio: 1087

#🔍 Outlier Summary (>3 std from mean):
#loanamount: 1104 outliers
#apr: 60 outliers
#repayment_ratio: 374 outliers
#payment_success_rate: 0 outliers

# 📊 Missing Data in Originated Loans (Top 10 features):
# _underwritingdataclarity_clearfraud_clearfraudidentityverification_phonetype                           97.806808
# _underwritingdataclarity_clearfraud_clearfraudidentityverification_ssnnamereasoncode                   96.159197
# _underwritingdataclarity_clearfraud_clearfraudidentityverification_ssnnamereasoncodedescription        96.159197
# _underwritingdataclarity_clearfraud_clearfraudidentityverification_nameaddressreasoncode               92.074947
# _underwritingdataclarity_clearfraud_clearfraudidentityverification_nameaddressreasoncodedescription    92.074947
# _underwritingdataclarity_clearfraud_clearfraudidentityverification_ssndobreasoncode                    87.242968
# _underwritingdataclarity_clearfraud_clearfraudindicator_driverlicenseinconsistentwithonfile            86.049646
# _underwritingdataclarity_clearfraud_clearfraudindicator_workphonepreviouslylistedascellphone           67.558579
# _underwritingdataclarity_clearfraud_clearfraudindicator_workphonepreviouslylistedashomephone           67.558579
# _underwritingdataclarity_clearfraud_clearfraudindicator_driverlicenseformatinvalid                     36.869104
# dtype: float64

# ✅ Final Sanity Check After Cleaning:
# Negative loan amounts: 0
# Negative APR: 0
# Negative repayment ratio: 0

# 🎯 Data Cleaning & Validation Complete — Ready for Exploration & Visualization.

// Data Exploration & Visualization
// Loan Status Distribution
# Business Takeaway:
# - Observation:
#     - The distribution of loan statuses gives a snapshot of overall portfolio health.
#     - A high proportion of "Defaulted" or "Charged-Off" loans signals elevated credit risk.
# - Business Impact:
#     - If default rates are high, underwriting criteria may need tightening (e.g., higher minimum credit score, stricter fraud checks).
#     - If a large number of loans are in "Late" status, collection strategies might need improvement.
# - Modeling Relevance:
#     - 'loanstatus' itself is a post-outcome variable and should not be used in predictive modeling 
#       to avoid target leakage.
#     - However, analyzing its distribution helps validate model objectives and focus areas.
plt.figure(figsize=(15, 6))
sns.countplot(data=loan_originated, 
              x='loanstatus', 
              order=loan_originated['loanstatus'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title('Loan Status Distribution - Originated Loans')
plt.tight_layout()
plt.show()

// Repayment Ratio Distribution
# Business Takeaway:
# - Observation:
#     - Most fully paid loans cluster around a repayment ratio of 1.0.
#     - Ratios below 1 indicate partial or no repayment.
#     - Ratios far above 1 may indicate overpayment, early settlement, or extra interest payments.
# - Business Impact:
#     - Borrowers below 1.0 are potential defaults and require collection strategies.
#     - Overpayment patterns could signal opportunities for cross-selling or loyalty programs.
# - Modeling Relevance:
#     - Repayment ratio itself is leakage for modeling repayment prediction,
#       but its distribution informs data understanding and helps in feature engineering for derived metrics.
sns.histplot(loan_originated['repayment_ratio'], bins=50, kde=True)
plt.axvline(1, color='red', linestyle='--', label='Fully Paid Threshold')
plt.legend()
plt.title('Repayment Ratio Distribution - Originated Loans')
plt.xlabel('Repayment Ratio')
plt.show()

// Payment Success Rate by Lead Type
# Business Takeaway: 
# - Observation: Certain acquisition channels, such as 'organic' and 'rc_returning',
#   consistently produce customers with higher payment success rates.
# - Marketing Insight:
#     - Channels with higher repayment consistency deliver better long-term value.
#     - Lower-risk channels can support higher approval rates and potentially more flexible terms.
# - Business Impact:
#     - Allocate more marketing budget toward high-quality lead sources.
#     - Re-evaluate or tighten criteria for channels with lower repayment performance.
# - Modeling Relevance:
#     - Lead type is a strong predictor of repayment behavior and should be included in the model.
#     - For new applicants from high-risk lead types, consider stricter fraud checks or adjusted APR.
plt.figure(figsize=(10,6))
sns.boxplot(data=loan_originated, x='leadtype', y='payment_success_rate', showfliers=False)
plt.xticks(rotation=45)
plt.title('Payment Success Rate by Lead Type')
plt.show()

// APR vs Repayment Ratio
# Business Takeaway: 
# - Observation: Higher APR loans tend to have slightly lower repayment ratios.
# - Possible Interpretation:
#     1. Price Sensitivity: Borrowers facing higher interest rates may struggle more 
#        to repay the loan in full, suggesting affordability issues.
#     2. Risk-based Pricing Feedback Loop: Higher APRs may be given to riskier borrowers,
#        so lower repayment ratios could reflect underlying credit risk.
# - Business Impact:
#     - Adjust APR strategies for borderline applicants to improve affordability 
#       and reduce default rates.
#     - Consider pairing high APR loans with stricter underwriting checks.
# - Modeling Relevance:
#     - APR should remain as a key predictive feature, but be monitored closely
#       for potential policy interventions.

plt.figure(figsize=(8,5))
sns.scatterplot(data=loan_originated, x='apr', y='repayment_ratio', 
                hue='is_fully_paid', alpha=0.6)
plt.title('APR vs Repayment Ratio')
plt.show()

// Correlation Heatmap for Numeric Features
# Business Takeaway: 
# - Correlation helps identify redundant features that may contain overlapping information.
# - Example: 'total_paid' and 'repayment_ratio' are strongly correlated 
#   because both describe repayment outcomes.
# - In predictive modeling:
#     - Highly correlated variables (correlation > 0.9) can cause multicollinearity,
#       inflating coefficient variance in linear models.
#     - In tree-based models, redundancy can dilute feature importance rankings.
# - Business Context in Lending:
#     - Dropping redundant features improves interpretability and model stability.
#     - Avoids over-weighting certain aspects of borrower behavior.
#     - Keeps the model lean, faster to train, and easier to deploy.
numeric_corr = loan_originated.select_dtypes(include=np.number).corr()
plt.figure(figsize=(12,8))
sns.heatmap(numeric_corr, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap - Numeric Features (Originated Loans)')
plt.show()

// Missing Data Visualization
# - High missingness can signal process-related differences in data collection.
# - Example: Certain underwriting variables may only be captured for specific lead types 
#   or during manual reviews, leading to structured missingness rather than random gaps.
# - Understanding this pattern is critical:
#     - If missingness is process-driven, we might keep the feature and treat 'missing' as a category.
#     - If missingness is random, imputation (e.g., median/mode) might be appropriate.
# - In lending, process-driven missingness can itself be predictive:
#     - E.g., if a fraud check is skipped only for VIP repeat customers, "missing" could imply lower risk.
missing_pct = loan_originated.isnull().mean().sort_values(ascending=False) * 100
plt.figure(figsize=(8,6))
sns.barplot(x=missing_pct.head(10), y=missing_pct.head(10).index)
plt.title('Top 10 Features with Missing Data - Originated Loans')
plt.xlabel('Missing %')
plt.ylabel('Feature')
plt.show()

// Sensible Modeling
# -------------------------------------------------
# Step 1: Define target variable
# -------------------------------------------------
# 'is_fully_paid' = 1 if loan is fully repaid, 0 otherwise.
# This will be our prediction target for the model.
target = 'is_fully_paid'
# -------------------------------------------------
# Step 2: Remove features that cause target leakage
# -------------------------------------------------
# These variables are only known AFTER loan origination or repayment starts.
# Keeping them would allow the model to "cheat" and produce unrealistically high accuracy.
# Business takeaway: A real lender would not have these values at application time.
leakage_features = [
    'total_paid',
    'total_principal_paid',
    'total_fees_paid',
    'num_payments',
    'num_success_payments',
    'num_failed_payments',
    'repayment_ratio',
    'payment_success_rate'
]
# -------------------------------------------------
# Step 3: Remove identifiers and non-predictive fields
# -------------------------------------------------
# IDs are unique per loan/customer and carry no predictive value for repayment.
# Business takeaway: Keeping IDs would risk memorization of individual records rather than learning patterns.
id_features = [
    'loanid', 
    'anon_ssn', 
    'clarityfraudid', 
    'underwritingid'
]
# -------------------------------------------------
# Step 4: Remove high-missingness features (>80% missing)
# -------------------------------------------------
# Features with excessive missing values contribute little value and can harm model stability.
# Business takeaway: Variables that are rarely available at application time are unreliable for decision-making.
missing_pct = loan_originated.isnull().mean()
high_missing_features = missing_pct[missing_pct > 0.8].index.tolist()
# -------------------------------------------------
# Step 5: Build initial feature list
# -------------------------------------------------
# Keep all features except:
# - The target variable
# - Leakage features
# - ID fields
# - High-missingness fields
features = loan_originated.columns.difference(
    [target] + leakage_features + id_features + high_missing_features
)
print(f"Initial number of candidate features: {len(features)}")
# -------------------------------------------------
# Step 6: Remove highly correlated features (correlation > 0.9)
# -------------------------------------------------
# Highly correlated features can cause multicollinearity, which:
# - Inflates the variance of coefficients (in linear models)
# - Can overweight redundant variables in tree models
# Business takeaway: Removing redundant predictors improves interpretability and stability.
numeric_features = loan_originated[features].select_dtypes(include=np.number).columns
corr_matrix = loan_originated[numeric_features].corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.9)]
final_features = [f for f in features if f not in to_drop_corr]
print(f"Features after correlation filter: {len(final_features)}")
# -------------------------------------------------
# Step 7: Extra strict leakage removal (application-time only)
# -------------------------------------------------
# This step catches hidden leakage columns by keyword matching.
# For example: 'loanstatus', 'approved', 'isfunded', etc.
# Business takeaway: Ensures we only train with information available at loan application time.
extra_leakage_keywords = [
    'status',       # loanstatus, fpstatus
    'funded',       # isfunded
    'approved',     # approval status
    'paid',         # total_paid, num_paid
    'collection',   # collection events
    'void',         # voided loans
    'ratio',        # repayment ratio
    'success_rate', # payment success rate
    'default',      # default indicators
    'days_to_fund', # time to fund
]
safe_features = [
    col for col in final_features
    if not any(keyword in col.lower() for keyword in extra_leakage_keywords)
]
print(f"Features after extra leakage filtering: {len(safe_features)}")
# -------------------------------------------------
# Step 8: Prepare modeling dataset
# -------------------------------------------------
# We only one-hot encode low-cardinality categorical features (<=20 unique values).
# High-cardinality categoricals are dropped for Logistic Regression to avoid memory blow-up.
# Business takeaway: This keeps the model lightweight, interpretable, and deployable.
low_cardinality_cols = [
    col for col in loan_originated[safe_features].select_dtypes(exclude=np.number).columns
    if loan_originated[col].nunique() <= 20
]
high_cardinality_cols = [
    col for col in loan_originated[safe_features].select_dtypes(exclude=np.number).columns
    if loan_originated[col].nunique() > 20
]
print(f"Low-cardinality categorical: {len(low_cardinality_cols)}")
print(f"High-cardinality categorical dropped for LR: {len(high_cardinality_cols)}")
# One-hot encode low-cardinality categoricals
X = pd.get_dummies(
    loan_originated[safe_features],
    columns=low_cardinality_cols,
    drop_first=True
)
# Drop high-cardinality categoricals for Logistic Regression
X = X.drop(columns=high_cardinality_cols)
# Target variable
y = loan_originated[target]
print(f"Final modeling dataset shape after encoding: {X.shape}")
# Initial number of candidate features: 70
# Features after correlation filter: 67
# Features after extra leakage filtering: 61
# Low-cardinality categorical: 43
# High-cardinality categorical dropped for LR: 3
# Final modeling dataset shape after encoding: (46006, 133)

// Modeling & Evaluation
# -------------------------------------------------
# Step 1: Train-Test Split
# -------------------------------------------------
# Split the dataset into training (80%) and testing (20%) sets.
# Stratify ensures the proportion of fully paid vs not fully paid loans
# remains consistent in both sets — critical for balanced evaluation in imbalanced datasets.
# Business takeaway: In lending, keeping class balance ensures evaluation metrics are reliable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
# -------------------------------------------------
# Step 2: Baseline Model - Logistic Regression
# -------------------------------------------------
# Logistic Regression is interpretable and fast to train.
# This acts as a benchmark model to compare against more complex algorithms.
# Business takeaway: Regulatory compliance in lending often requires interpretable models like Logistic Regression.
log_reg = LogisticRegression(max_iter=500, solver='liblinear')
log_reg.fit(X_train, y_train)
# Predictions
y_pred_lr = log_reg.predict(X_test)
y_pred_prob_lr = log_reg.predict_proba(X_test)[:, 1]
# Evaluation Metrics
auc_lr = roc_auc_score(y_test, y_pred_prob_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
print("\n📊 Logistic Regression Performance:")
print(f"AUC: {auc_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1-score: {f1_lr:.4f}")
# -------------------------------------------------
# Step 3: Advanced Model - XGBoost Classifier
# -------------------------------------------------
# XGBoost is a gradient boosting algorithm that typically delivers strong predictive performance.
# It can model non-linear relationships and handle feature interactions automatically.
# Business takeaway: While less interpretable, it can uncover deeper patterns for underwriting decisions.
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    random_state=42
)
xgb_model.fit(X_train, y_train)
# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
# Evaluation Metrics
auc_xgb = roc_auc_score(y_test, y_pred_prob_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
print("\n📊 XGBoost Performance:")
print(f"AUC: {auc_xgb:.4f}")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall: {recall_xgb:.4f}")
print(f"F1-score: {f1_xgb:.4f}")
# -------------------------------------------------
# Step 4: ROC Curve Comparison
# -------------------------------------------------
# The ROC curve visualizes the trade-off between sensitivity (TPR) and specificity (1-FPR).
# AUC measures how well the model can distinguish between fully paid vs not fully paid loans.
# Business takeaway: A higher AUC means better ranking of good vs risky loans.
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob_xgb)
plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.3f})")
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()
# -------------------------------------------------
# Step 5: Feature Importance (XGBoost)
# -------------------------------------------------
# Feature importance shows which variables are most influential in predictions.
# Business takeaway: Highlights actionable factors underwriting can monitor or adjust.
xgb_importances = pd.Series(
    xgb_model.feature_importances_, 
    index=X.columns
).sort_values(ascending=False).head(15)
plt.figure(figsize=(8,6))
sns.barplot(x=xgb_importances, y=xgb_importances.index)
plt.title('Top 15 Feature Importances (XGBoost)')
plt.show()
# -------------------------------------------------
# Step 6: Business Takeaways
# -------------------------------------------------
# 1. Both models achieved high AUC (~0.97), indicating strong predictive power.
# 2. Logistic Regression provides interpretability — useful for explaining loan decisions.
# 3. XGBoost delivers similar accuracy but may detect non-linear risk patterns.
# 4. Top predictive features (e.g., fraud flags, lead type, APR) can directly inform underwriting rules.
# 5. High recall ensures the model catches most risky loans, reducing default exposure.
# Training set: (36804, 133), Test set: (9202, 133)

# 📊 Logistic Regression Performance:
# AUC: 0.9767
# Precision: 0.9925
# Recall: 0.9756
# F1-score: 0.9840

# 📊 XGBoost Performance:
# AUC: 0.9791
# Precision: 0.9919
# Recall: 0.9763
# F1-score: 0.9840

