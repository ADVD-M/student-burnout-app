"""
feature_engineering.py
-----------------------
Defines the feature columns and builds the scikit-learn preprocessing pipeline
used for both training and inference.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ── Feature Column Definitions ───────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    "Gender",
    "Course",
    "YearOfStudy",
]

NUMERICAL_FEATURES = [
    "Age",
    "CGPA",
    "Depression",
    "Anxiety",
    "PanicAttack",
    "SpecialistTreatment",
    "SymptomFrequency_Last7Days",
    "HasMentalHealthSupport",
    "SleepQuality",
    "StudyStressLevel",
    "StudyHoursPerWeek",
    "AcademicEngagement",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "BurnoutRisk"


def get_feature_columns() -> dict:
    """
    Return a dict with categorical and numerical feature column lists.

    Returns:
        dict with keys 'categorical', 'numerical', 'all'
    """
    return {
        "categorical": CATEGORICAL_FEATURES,
        "numerical": NUMERICAL_FEATURES,
        "all": ALL_FEATURES,
    }


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that:
        - Imputes + OneHotEncodes categorical columns
        - Imputes + Scales numerical columns

    Returns:
        Configured ColumnTransformer (not yet fitted).
    """
    # Categorical sub-pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Numerical sub-pipeline
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def prepare_dataframe(df: pd.DataFrame) -> tuple:
    """
    Extract feature matrix X and target vector y from a labelled DataFrame.

    Args:
        df: DataFrame with ALL_FEATURES + TARGET_COLUMN columns.

    Returns:
        (X, y) as (pd.DataFrame, pd.Series)
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
