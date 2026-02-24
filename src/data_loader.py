"""
data_loader.py
--------------
Handles loading and initial cleaning of the student mental health dataset,
and engineers the composite BurnoutRisk target label.
"""

import pandas as pd
import numpy as np


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the student mental health CSV dataset from the given path.

    Args:
        path: Absolute or relative path to the CSV file.

    Returns:
        Cleaned pandas DataFrame ready for feature engineering.
    """
    df = pd.read_csv(path)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Normalize YearOfStudy casing (e.g. "year 1" -> "Year 1")
    df["YearOfStudy"] = df["YearOfStudy"].str.title()

    # Normalize Course casing
    df["Course"] = df["Course"].str.title()

    # Drop Timestamp if present (not a feature)
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Drop any fully duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def generate_burnout_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a composite BurnoutRisk label (Low / Medium / High) from
    the following symptom columns:
        - Depression (0/1)
        - Anxiety (0/1)
        - PanicAttack (0/1)
        - SleepQuality (1–5, lower = worse)
        - StudyStressLevel (1–5, higher = worse)
        - SymptomFrequency_Last7Days (0–7, higher = worse)

    Scoring logic:
        - Each binary symptom present  → +2 points
        - Poor sleep (SleepQuality ≤ 2) → +2 points
        - High stress (StudyStressLevel ≥ 4) → +2 points
        - High symptom frequency (≥ 5 days) → +2 points

    Risk categories (max 12 points):
        - 0–3  → Low
        - 4–7  → Medium
        - 8–12 → High

    Args:
        df: DataFrame with the above columns present.

    Returns:
        DataFrame with a new 'BurnoutRisk' column added.
    """
    df = df.copy()

    score = pd.Series(np.zeros(len(df)), index=df.index)

    # Binary symptom contributions
    for col in ["Depression", "Anxiety", "PanicAttack"]:
        if col in df.columns:
            score += df[col].astype(int) * 2

    # Sleep quality (1–5 scale; ≤ 2 is poor)
    if "SleepQuality" in df.columns:
        score += (df["SleepQuality"] <= 2).astype(int) * 2

    # Study stress (1–5 scale; ≥ 4 is high)
    if "StudyStressLevel" in df.columns:
        score += (df["StudyStressLevel"] >= 4).astype(int) * 2

    # Symptom frequency (0–7 days; ≥ 5 is high)
    if "SymptomFrequency_Last7Days" in df.columns:
        score += (df["SymptomFrequency_Last7Days"] >= 5).astype(int) * 2

    df["BurnoutScore"] = score.astype(int)

    def categorize(s):
        if s <= 3:
            return "Low"
        elif s <= 7:
            return "Medium"
        else:
            return "High"

    df["BurnoutRisk"] = df["BurnoutScore"].apply(categorize)

    return df
