import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def predict_with_threshold(model, X, threshold=0.5):
    probs = model.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int)


def evaluate(model, X, y, threshold=0.5):
    preds = predict_with_threshold(model, X, threshold)

    return {
        "report": classification_report(y, preds, zero_division=0),
        "f1": f1_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds)
    }


def tune_threshold(model, X_val, y_val, start=0.1, end=0.6, step=0.01):
    results = []

    probs = model.predict_proba(X_val)[:, 1]

    for t in np.arange(start, end, step):
        preds = (probs >= t).astype(int)

        results.append({
            "threshold": round(t, 2),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds),
            "f1": f1_score(y_val, preds)
        })

    return pd.DataFrame(results)

def get_best_threshold(df, metric="f1"):
    return df.sort_values(metric, ascending=False).iloc[0]