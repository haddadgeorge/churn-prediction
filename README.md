# Churn Prediction Project

## Goal

Predict customer churn using machine learning models and compare performance.

---

## Dataset

Telco Customer Churn dataset (~7000 rows)

Target:

* 1 = churn
* 0 = no churn

---

## Pipeline

### Preprocessing

* Removed irrelevant columns (IDs, geography, etc.)
* Converted target to binary
* Handled missing values in Total Charges
* One-hot encoded categorical variables (fit on training set only)
* Train/validation/test split (no leakage)

---

## Models

### Logistic Regression

* L2 regularization (C tuned)
* Requires feature scaling
* Final parameters:

  * C = 0.001
  * threshold = 0.47 (validation-tuned)

---

### Random Forest

* n_estimators = 200
* max_depth = 10
* min_samples_leaf = 5
* threshold = 0.29 (validation-tuned)

---

## Evaluation Strategy

* Metrics: precision, recall, F1-score, accuracy
* Threshold tuning performed on validation set only
* Final evaluation done once on test set

---

## Results

### Logistic Regression (Test)

* Accuracy: 0.77
* Precision (Churn): 0.55
* Recall (Churn): 0.70
* F1-score (Churn): 0.62

---

### Random Forest (Test)

* Accuracy: 0.76
* Precision (Churn): 0.53
* Recall (Churn): 0.70
* F1-score (Churn): 0.61

---

## Key Insights

* Both models perform similarly after tuning
* Logistic Regression slightly outperforms Random Forest in F1
* Strongest predictors:

  * tenure
  * contract type
  * monthly/total charges
  * support services
  * payment method

---

## Conclusion

Churn behavior is largely driven by structured linear relationships in the data. Model performance depends more on feature quality and preprocessing than model complexity.
