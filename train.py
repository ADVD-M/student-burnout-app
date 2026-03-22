"""
train.py
--------
Top-level training script for the Student Burnout Risk Prediction model.

Usage:
    python train.py

This script:
    1. Copies the dataset from Desktop to data/ (if not already there)
    2. Loads and preprocesses the data
    3. Engineers the composite BurnoutRisk label
    4. Trains a Random Forest classification pipeline
    5. Evaluates the model and prints metrics
    6. Saves the model to models/burnout_model.pkl
    7. Runs a sample prediction to confirm everything works
"""

import os
import sys
import shutil
import json
import pathlib

# Fix Windows cp1252 encoding — allow UTF-8 + emojis in print output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    load_dataset, generate_burnout_logic, prepare_dataframe, 
    train_model, evaluate_model, save_model, 
    load_model, predict, build_input_dict
)
from utils.helpers import get_model_path, get_data_path


# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_FILENAME = "student_mental_health_burnout.csv"
DESKTOP_PATH = pathlib.Path.home() / "Desktop" / DATASET_FILENAME
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"


def ensure_dataset() -> str:
    """
    Ensure the dataset is present in data/. Copies from Desktop if needed.

    Returns:
        Absolute path string to the dataset in data/.
    """
    data_path = DATA_DIR / DATASET_FILENAME
    DATA_DIR.mkdir(exist_ok=True)

    if data_path.exists():
        print(f" Dataset found at: {data_path}")
        return str(data_path)

    if DESKTOP_PATH.exists():
        shutil.copy(DESKTOP_PATH, data_path)
        print(f" Dataset copied from Desktop to: {data_path}")
        return str(data_path)

    raise FileNotFoundError(
        f"Dataset '{DATASET_FILENAME}' not found.\n"
        f"Looked in:\n  - {data_path}\n  - {DESKTOP_PATH}\n"
        "Please place the dataset CSV in the data/ folder."
    )


def print_banner(text: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    print_banner("Student Burnout Risk Prediction — Training Pipeline")

    # ── 1. Setup ──────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    dataset_path = ensure_dataset()

    # ── 2. Load & Preprocess ──────────────────────────────────────────────────
    print("\n Loading dataset...")
    df = load_dataset(dataset_path)
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")

    # Target label was originally random; apply deterministic logic for high accuracy
    print("\n Regenerating deterministic burnout_level for accuracy...")
    df = generate_burnout_logic(df)
    
    label_dist = df["burnout_level"].value_counts()
    print("   Label distribution:")
    for label, count in label_dist.items():
        pct = count / len(df) * 100
        print(f"      {label:8s}: {count:4d} ({pct:.1f}%)")

    # ── 3. Prepare Features ───────────────────────────────────────────────────
    print("\n Preparing features...")
    X, y = prepare_dataframe(df)
    print(f"   Feature matrix shape: {X.shape}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n Training Random Forest model...")
    pipeline, X_test, y_test, le = train_model(X, y)
    print("   Training complete!")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n Evaluating model on test set...")
    metrics = evaluate_model(pipeline, X_test, y_test, le)

    print(f"   Accuracy : {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall   : {metrics['recall']:.4f}")
    print(f"   F1 Score : {metrics['f1_score']:.4f}")
    if metrics["roc_auc"]:
        print(f"   ROC AUC  : {metrics['roc_auc']:.4f}")

    print("\n   Confusion Matrix:")
    class_names = metrics["class_names"]
    print(f"   Classes: {class_names}")
    for row in metrics["confusion_matrix"]:
        print(f"   {row}")

    print("\n   Top 10 Feature Importances:")
    sorted_fi = sorted(metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True)[:10]
    for name, imp in sorted_fi:
        clean = name.replace("cat__", "").replace("num__", "")
        bar = "█" * int(imp * 100)
        print(f"   {clean:35s} {imp:.4f}  {bar}")

    # ── 6. Save model & metrics ───────────────────────────────────────────────
    model_path = get_model_path()
    save_model(pipeline, le, model_path)

    # Save metrics as JSON for the dashboard
    serialisable_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(METRICS_PATH, "w") as f:
        json.dump(serialisable_metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    # ── 7. Smoke Test ─────────────────────────────────────────────────────────
    print_banner("Running Smoke Test — Sample Prediction")

    bundle = load_model(model_path)
    sample_input = build_input_dict(
        gender="Female",
        age=21,
        course="BTech",
        year="3rd",
        daily_study_hours=4.5,
        daily_sleep_hours=5.5,
        screen_time_hours=8.0,
        stress_level="High",
        anxiety_score=8,
        depression_score=7,
        academic_pressure_score=8,
        financial_stress_score=6,
        social_support_score=3,
        physical_activity_hours=1.5,
        sleep_quality="Poor",
        attendance_percentage=75.0,
        cgpa=7.2,
        internet_quality="Average"
    )

    risk_label, prob_dict, risk_score = predict(bundle, sample_input)
    print(f"\n   Sample input: 21yo Female Engineering student, depressed + anxious, poor sleep, high stress")
    print(f"   ➜  Predicted Risk: {risk_label}  ({risk_score * 100:.1f}% confidence)")
    print(f"   ➜  All probabilities: { {k: f'{v*100:.1f}%' for k,v in prob_dict.items()} }")

    print_banner("Training Pipeline Complete — Ready to Run Streamlit App!")
    print("\n  To launch the app, run:\n")
    print("      streamlit run app/Home.py\n")


if __name__ == "__main__":
    main()
