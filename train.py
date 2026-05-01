import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from src.preprocessing import load_data, clean_data, split_data
from src.models import train_random_forest, train_logistic_regression
from src.evaluation import evaluate, tune_threshold


# --------------------
# LOAD + CLEAN
# --------------------
df = load_data("data/Telco_customer_churn.xlsx")
df = clean_data(df)

X = df.drop(columns=["Churn Label"])
y = df["Churn Label"]


# --------------------
# SPLIT
# --------------------
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# --------------------
# ENCODING
# --------------------
from src.preprocessing import encode_features

X_train_final, X_val_final, X_test_final, encoder = encode_features(
    X_train, X_val, X_test
)

# --------------------
# RANDOM FOREST
# --------------------
from src.models import train_random_forest
from src.evaluation import evaluate, tune_threshold

rf = train_random_forest(X_train_final, y_train)

best_t = 0.29  # tuned on validation set

final_preds = (rf.predict_proba(X_test_final)[:, 1] >= best_t).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_test, final_preds))

# --------------------
# LOGISTIC REGRESSION
# --------------------

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_train_s = scaler.fit_transform(X_train_final)
# X_val_s = scaler.transform(X_val_final)
# X_test_s = scaler.transform(X_test_final)

# from src.models import train_logistic_regression

# best_C = 0.001
# lr_best = train_logistic_regression(X_train_s, y_train, C=best_C)

# best_t = 0.47

# from sklearn.metrics import classification_report

# preds = (lr_best.predict_proba(X_test_s)[:, 1] >= best_t).astype(int)

# print("\nFinal LR Test Results:")
# print(classification_report(y_test, preds))
