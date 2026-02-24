"""
inference.py
------------
Loads a persisted model and runs predictions on new user-supplied input.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from src.feature_engineering import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def load_model(model_path: str) -> dict:
    """
    Load the saved model bundle (pipeline + label_encoder).

    Args:
        model_path: Path to the .pkl file produced by model_trainer.save_model().

    Returns:
        Dict with 'pipeline' and 'label_encoder' keys.
    """
    bundle = joblib.load(model_path)
    return bundle


def predict(
    bundle: dict, input_dict: dict
) -> Tuple[str, Dict[str, float], float]:
    """
    Run inference on a single student's data.

    Args:
        bundle: Dict with 'pipeline' and 'label_encoder' from load_model().
        input_dict: Dict of feature name → value, matching ALL_FEATURES layout.

    Returns:
        Tuple of:
            - risk_label (str): 'Low', 'Medium', or 'High'
            - probability_dict (dict): {class_name: probability}
            - risk_score (float): probability of the predicted class (0–1)
    """
    pipeline = bundle["pipeline"]
    le = bundle["label_encoder"]

    # Build a single-row DataFrame in correct column order
    input_df = pd.DataFrame([input_dict])[ALL_FEATURES]

    # Predict class and probabilities
    pred_encoded = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]

    risk_label = le.inverse_transform([pred_encoded])[0]
    class_names = le.classes_.tolist()
    probability_dict = {name: round(float(p), 4) for name, p in zip(class_names, probabilities)}

    risk_score = probability_dict[risk_label]

    return risk_label, probability_dict, risk_score


def build_input_dict(
    gender: str,
    age: int,
    course: str,
    year_of_study: str,
    cgpa: float,
    depression: int,
    anxiety: int,
    panic_attack: int,
    specialist_treatment: int,
    symptom_frequency: int,
    has_mental_health_support: int,
    sleep_quality: int,
    study_stress_level: int,
    study_hours_per_week: int,
    academic_engagement: int,
) -> dict:
    """
    Convenience constructor to build the input_dict from individual values.
    All parameters correspond to dataset columns.

    Returns:
        Dict matching ALL_FEATURES schema.
    """
    return {
        "Gender": gender,
        "Age": age,
        "Course": course,
        "YearOfStudy": year_of_study,
        "CGPA": cgpa,
        "Depression": depression,
        "Anxiety": anxiety,
        "PanicAttack": panic_attack,
        "SpecialistTreatment": specialist_treatment,
        "SymptomFrequency_Last7Days": symptom_frequency,
        "HasMentalHealthSupport": has_mental_health_support,
        "SleepQuality": sleep_quality,
        "StudyStressLevel": study_stress_level,
        "StudyHoursPerWeek": study_hours_per_week,
        "AcademicEngagement": academic_engagement,
    }
