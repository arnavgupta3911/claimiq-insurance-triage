import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "claim_amount",
    "claim_type",
    "policy_age_months",
    "documents_submitted",
    "documents_required",
    "days_since_incident",
    "previous_claims",
]
CATEGORICAL = ["claim_type"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
TARGET = "requires_review"
SEED = 42

MODEL_PATH = Path("models/claim_model.joblib")
METRICS_PATH = Path("models/metrics.json")


def build_pipeline():
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def scores(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def main():
    df = pd.read_csv("data/synthetic_claims.csv")
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_prob = baseline.predict_proba(X_test)[:, 1]

    model_scores = scores(y_test, pred, prob)
    base_scores = scores(y_test, base_pred, base_prob)

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    print(f"Test review rate: {y_test.mean():.3f}\n")
    print("Logistic regression (threshold 0.5):")
    print(classification_report(y_test, pred, digits=3))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(cm)

    print("\nModel vs majority-class baseline:")
    comparison = pd.DataFrame({"model": model_scores, "baseline": base_scores}).round(3)
    print(comparison)

    thresholds = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p = (prob >= t).astype(int)
        thresholds.append({
            "threshold": t,
            "precision": precision_score(y_test, p, zero_division=0),
            "recall": recall_score(y_test, p, zero_division=0),
            "f1": f1_score(y_test, p, zero_division=0),
            "flagged_share": p.mean(),
        })
    print("\nThreshold comparison:")
    print(pd.DataFrame(thresholds).round(3).to_string(index=False))

    names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["classifier"].coef_[0]
    importance = (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .assign(feature=lambda d: d["feature"].str.replace(r"^(num|cat)__", "", regex=True))
        .sort_values("coefficient", key=np.abs, ascending=False)
    )
    print("\nStandardised coefficients:")
    print(importance.round(3).to_string(index=False))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_review_rate": float(y_test.mean()),
        "model": model_scores,
        "baseline": base_scores,
        "confusion_matrix": cm.tolist(),
        "thresholds": thresholds,
        "coefficients": importance.to_dict(orient="records"),
    }, indent=2, default=float))
    print(f"\nSaved model to {MODEL_PATH} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
