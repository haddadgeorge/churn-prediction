# Customer Churn Prediction

A machine learning project that predicts customer churn using structured telecom data. The goal is to identify customers likely to leave and optimize detection using threshold tuning.

---

## Problem Overview

Customer churn is a binary classification problem where the goal is to predict whether a customer will leave the service based on behavioral and contract-related features.

---

## Methodology

### Data Preprocessing

* Removed irrelevant and leakage-prone columns (e.g. churn reason, location metadata)
* Converted target variable into binary format
* Handled missing values in numeric fields
* Applied one-hot encoding to categorical variables

### Models Used

* Logistic Regression (baseline, interpretable model)
* Random Forest (non-linear model for comparison)

### Training Strategy

* Train / validation / test split
* Hyperparameter tuning on validation set only
* Threshold tuning for optimal classification performance

---

## Pipeline Overview

```
Raw Data
   ↓
Data Cleaning
   ↓
Feature Encoding (One-Hot)
   ↓
Train / Validation / Test Split
   ↓
Model Training (LR / RF)
   ↓
Threshold Tuning (Validation)
   ↓
Final Evaluation (Test Set)
```

---

## Evaluation Strategy

Models were evaluated using:

* Accuracy
* Precision / Recall
* F1-score
* Confusion matrix

Special focus was placed on **recall for churn class**, due to business importance of identifying at-risk customers.

---

## Final Results

### Logistic Regression

* Accuracy: ~0.77
* Churn F1-score: ~0.62
* Recall (Churn): ~0.70

### Random Forest

* Accuracy: ~0.74
* Churn F1-score: ~0.62
* Recall (Churn): ~0.79

---

## Key Insights

* Churn prediction performance is highly sensitive to decision threshold
* Lower thresholds improve recall at the cost of precision
* Important drivers of churn:

  * Contract type
  * Tenure
  * Monthly charges
  * Support-related services

---

## Tech Stack

* Python
* pandas, numpy
* scikit-learn

---

## Project Structure

src/ → reusable ML pipeline code
train.py → main training script
data/ → dataset (excluded from GitHub)
README.md → project documentation

---

## How to Run

pip install -r requirements.txt
python train.py
