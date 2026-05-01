import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_excel(path)


def clean_data(df):
    df = df.copy()

    drop_cols = [
        "customerID", "Churn Value", "Churn Score", "CLTV",
        "Churn Reason", "Count", "Country", "State", "City",
        "Zip Code", "Lat Long", "Latitude", "Longitude"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    df["Churn Label"] = df["Churn Label"].map({"Yes": 1, "No": 0})

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.dropna(subset=["Total Charges"])
    print(f"Final dataset shape after cleaning: {df.shape}")

    return df


def split_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        stratify=y_train_val,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_features(X_train, X_val, X_test):
    """
    One-hot encode categorical features using training set only.
    Returns: encoded train, val, test + encoder
    """

    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns
    num_cols = X_train.select_dtypes(exclude=["object", "string"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_val_cat = encoder.transform(X_val[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    X_train_num = X_train[num_cols].to_numpy()
    X_val_num = X_val[num_cols].to_numpy()
    X_test_num = X_test[num_cols].to_numpy()

    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_val_final = np.hstack([X_val_num, X_val_cat])
    X_test_final = np.hstack([X_test_num, X_test_cat])

    return X_train_final, X_val_final, X_test_final, encoder