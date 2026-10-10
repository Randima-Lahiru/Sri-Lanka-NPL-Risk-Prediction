# Sri Lanka Non-Performing Loan Risk Prediction

## Overview

This project develops a machine learning-based system for predicting Non-Performing Loan (NPL) risk using Sri Lankan banking data.

The project compares three machine learning classification models:

- Logistic Regression
- Random Forest
- XGBoost

The models are evaluated using multiple classification metrics, and the best-performing model is used in a Streamlit dashboard for loan risk inference.

---

## Problem

The objective of this project is to classify a loan as:

- `0` = Performing Loan
- `1` = Non-Performing Loan

This is a binary classification problem.

The aim is to identify potentially risky loans while considering the effect of false negatives and false positives.

---

## Dataset

The project uses a Sri Lankan banking loan dataset.

The dataset contains 1,806 records and includes customer and loan-related information.

Main input features:

- Product
- Loan Balance
- Age
- Income
- Customer Category
- Gender
- Education
- CRIB Category

The following identifier columns were excluded from model training:

- ProductID
- Loan ID

---

## Target Variable

The target variable is `Status`.

It was converted into numerical form:

- `Performing` → `0`
- `Non Performing` → `1`

The dataset is imbalanced, with fewer Non-Performing Loan records than Performing Loan records.

Because of this, evaluation does not rely only on Accuracy.

---

## Data Preprocessing

The preprocessing pipeline includes:

- Removal of identifier columns
- Conversion of the target variable into numerical form
- Standardization of numerical features
- One-Hot Encoding of categorical features
- Stratified Train / Validation / Test splitting

The data was divided into:

- 70% Training
- 15% Validation
- 15% Testing

Stratification was used to maintain a similar class distribution in each dataset.

---

## Models

Three classification models were developed.

### Logistic Regression

Logistic Regression was used as an interpretable baseline model.

Because the dataset is imbalanced, balanced class weights were used.

### Random Forest

Random Forest was used as a bagging-based ensemble model.

Important parameters such as:

- Number of estimators
- Maximum depth
- Minimum samples split
- Minimum samples leaf

were configured and evaluated.

### XGBoost

XGBoost was used as a boosting-based ensemble model.

Important parameters included:

- Number of estimators
- Maximum depth
- Learning rate
- Subsample
- Column sample
- Positive class weight

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Final test results:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7786 | 0.3214 | 0.9000 | 0.4737 | 0.9065 |
| Random Forest | 0.8266 | 0.3651 | 0.7667 | 0.4946 | 0.9112 |
| XGBoost | 0.8782 | 0.4516 | 0.4667 | 0.4590 | 0.8700 |

Random Forest was selected as the final model because it achieved the highest F1 Score and ROC-AUC while maintaining relatively strong Recall.

---

## Feature Importance

The Random Forest feature importance analysis showed that CRIB-related features were the strongest predictors.

Other important variables included:

- Loan Balance
- Income
- Age

An interesting observation was that income did not show a simple inverse relationship with NPL risk.

In the dataset, higher-income groups did not always have lower NPL rates.

This demonstrates that loan risk depends on combinations of multiple variables rather than income alone.

---

## Inference

The selected Random Forest model can predict a new loan record.

The dashboard displays:

- Predicted Loan Class
- Performing Loan Probability Estimate
- NPL Probability Estimate

The probability values are model-generated estimates and should be interpreted as relative risk indicators rather than guaranteed real-world default probabilities.

---

## Streamlit Dashboard

The Streamlit dashboard contains three tabs.

### Prediction

Allows users to enter:

- Loan Product
- Loan Balance
- Age
- Income
- Customer Category
- Gender
- Education
- CRIB Category

The model then produces the predicted loan class and probability estimates.

### Model Comparison

Displays the performance comparison between:

- Logistic Regression
- Random Forest
- XGBoost

### Feature Importance

Displays the most influential features used by the selected Random Forest model.

---

## Project Structure

```text
Sri_Lanka_NPL_Risk_Prediction/
│
├── notebook/
│   └── Sri_Lanka_NPL_Risk_Prediction.ipynb
│
├── data/
│   └── BankDataset NPL.xlsx
│
├── models/
│   ├── npl_risk_model.pkl
│   └── model_features.pkl
│
├── dashboard/
│   ├── app.py
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Randima-Lahiru/Sri-Lanka-NPL-Risk-Prediction.git

```

Move into the project folder:

```bash
cd Sri-Lanka-NPL-Risk-Prediction
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

For macOS / Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## Important Package Versions

The saved machine learning model was created using the following main package versions:

```text
scikit-learn==1.6.1
pandas==2.2.3
joblib==1.6.0
xgboost==3.4.1
```

Using a different scikit-learn version may cause compatibility issues when loading the saved model.

---

## Run the Dashboard

Make sure the virtual environment is activated.

From the project root folder, run:

```bash
python -m streamlit run dashboard/app.py
```

Streamlit will display a local URL similar to:

```text
http://localhost:8501
```

Open that URL in your browser.

---

## Dashboard Output

The dashboard provides:

- Performing / Non-Performing Loan prediction
- Performing Loan Probability Estimate
- NPL Probability Estimate
- Model comparison results
- Random Forest feature importance

The probability values should be treated as model-based risk estimates rather than guaranteed real-world default probabilities.

---

## Model Selection

The three models showed different strengths:

- Logistic Regression achieved the highest Recall.
- XGBoost achieved the highest Accuracy and Precision.
- Random Forest achieved the highest F1 Score and ROC-AUC.

Random Forest was selected because it provided the best overall balance for this NPL risk prediction problem.

---

## Conclusion

This project demonstrates a complete machine learning workflow for Non-Performing Loan risk prediction using Sri Lankan banking data.

Three different machine learning approaches were compared:

- Logistic Regression as a linear baseline model
- Random Forest as a bagging-based ensemble model
- XGBoost as a boosting-based ensemble model

Random Forest was selected as the final model based on the final evaluation results.

The selected model was saved and integrated into a Streamlit dashboard that allows users to enter loan and customer information and obtain a model-based NPL risk prediction.

The project demonstrates:

- Data exploration
- Data preprocessing
- Class imbalance handling
- Model training
- Hyperparameter tuning
- Model comparison
- Model evaluation
- Feature importance analysis
- New record inference
- Model persistence
- Simple Streamlit deployment
