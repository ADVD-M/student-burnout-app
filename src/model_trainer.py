"""
model_trainer.py
----------------
Trains a Random Forest classification model within a full sklearn Pipeline
and evaluates it using multiple metrics.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import build_preprocessing_pipeline, ALL_FEATURES


def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple:
    """
    Build a full Pipeline (preprocessor + classifier), split data,
    train the model, and return (fitted_pipeline, X_test, y_test).

    Args:
        X: Feature DataFrame.
        y: Target Series (Low / Medium / High).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (fitted_pipeline, X_test, y_test, label_encoder)
    """
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Low=1, Medium=2, High=0 (alphabetical)

    # Train/test split — stratified to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
    )

    # Build full pipeline
    preprocessor = build_preprocessing_pipeline()
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # Fit on training data
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test, le


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """
    Evaluate a fitted pipeline on test data and return a comprehensive
    metrics dictionary.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test feature DataFrame.
        y_test: Test labels (encoded integers).
        label_encoder: LabelEncoder to decode class names.

    Returns:
        Dictionary containing all evaluation metrics and artefacts.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    class_names = label_encoder.classes_.tolist()  # ['High', 'Low', 'Medium']

    # Core metrics (macro-averaged for multi-class)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "class_names": class_names,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        ),
    }

    # ROC AUC (one-vs-rest for multi-class)
    try:
        metrics["roc_auc"] = round(
            roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"), 4
        )
    except Exception:
        metrics["roc_auc"] = None

    # Feature importances
    try:
        rf: RandomForestClassifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        feat_names = preprocessor.get_feature_names_out().tolist()
        importances = rf.feature_importances_.tolist()
        metrics["feature_importances"] = dict(zip(feat_names, importances))
    except Exception:
        metrics["feature_importances"] = {}

    return metrics


def save_model(pipeline: Pipeline, label_encoder: LabelEncoder, path: str) -> None:
    """
    Persist the fitted pipeline and label encoder together as a single dict.

    Args:
        pipeline: Fitted sklearn Pipeline.
        label_encoder: Fitted LabelEncoder.
        path: File path to save to (should end in .pkl).
    """
    joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder}, path)
    print(f"✅ Model saved to: {path}")
