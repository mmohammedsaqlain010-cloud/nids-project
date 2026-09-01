"""
train_model.py
---------------
Phases 2-6 of the project methodology:
  Phase 2: Data Preprocessing (encoding, scaling)
  Phase 3: Train/Test split (80/20)
  Phase 4: Random Forest Model Construction
  Phase 5: Prediction
  Phase 6: Performance Evaluation (accuracy, precision, recall, F1, confusion
           matrix, ROC data) -> saved to model/metrics.json for the dashboard

Run:
    python train_model.py
Outputs (in model/):
    rf_model.pkl        - trained RandomForestClassifier
    scaler.pkl           - fitted StandardScaler
    label_encoder.pkl    - fitted LabelEncoder for the target
    encoders.pkl         - dict of LabelEncoders for categorical features
    feature_columns.json - ordered list of feature column names
    metrics.json          - evaluation metrics + confusion matrix for the app
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from data.generate_data import generate_synthetic_dataset

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
LABEL_COL = "label"


def preprocess(df):
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[LABEL_COL])
    X = df.drop(columns=[LABEL_COL])

    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, encoders, label_encoder, feature_columns


def train_and_save():
    """Runs the full training pipeline and saves artifacts to model/.
    Reusable both from the CLI (`python train_model.py`) and as an
    automatic fallback from app.py if artifacts are missing."""
    print("Phase 1: Loading dataset (synthetic NIDS-style traffic)...")
    df = generate_synthetic_dataset()

    print("Phase 2: Preprocessing (encoding + scaling)...")
    X, y, scaler, encoders, label_encoder, feature_columns = preprocess(df)

    print("Phase 3: Splitting train (80%) / test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Phase 4: Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    print("Phase 5: Predicting on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("Phase 6: Evaluating performance...")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    class_names = label_encoder.classes_.tolist()
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Multiclass ROC (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    roc_data = {}
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_data[cname] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc(fpr, tpr)),
        }

    feature_importance = dict(
        sorted(
            zip(feature_columns, model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    metrics = {
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "confusion_matrix": cm,
        "class_names": class_names,
        "classification_report": report,
        "roc_data": roc_data,
        "feature_importance": feature_importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nSaving artifacts to model/ ...")
    joblib.dump(model, MODEL_DIR / "rf_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done. Model artifacts saved in model/")
    return metrics


if __name__ == "__main__":
    train_and_save()
