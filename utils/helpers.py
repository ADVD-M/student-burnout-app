"""
helpers.py
----------
Shared utility functions for path resolution, formatting, and display helpers.
Designed to work correctly both locally and on Streamlit Cloud.
"""

import os
import pathlib
from typing import Optional


def get_project_root() -> pathlib.Path:
    """
    Return the absolute path to the project root directory.
    Works regardless of where the script is called from.

    Returns:
        pathlib.Path pointing at the project root.
    """
    # This file lives at: <project_root>/utils/helpers.py
    return pathlib.Path(__file__).parent.parent.resolve()


def get_model_path() -> str:
    """Return the absolute path to the saved model file."""
    return str(get_project_root() / "models" / "burnout_model.joblib")


def get_data_path(filename: str = "mentalhealth_dataset.csv") -> str:
    """
    Return the absolute path to a file in the data/ directory.

    Args:
        filename: Name of the CSV file (default: mentalhealth_dataset.csv).

    Returns:
        Absolute path string.
    """
    return str(get_project_root() / "data" / filename)


def model_exists() -> bool:
    """Check whether the trained model file exists."""
    return os.path.exists(get_model_path())


def data_exists(filename: str = "mentalhealth_dataset.csv") -> bool:
    """Check whether the dataset file exists in data/."""
    return os.path.exists(get_data_path(filename))


def format_probability(prob: float) -> str:
    """
    Format a probability float as a percentage string.

    Args:
        prob: Float between 0 and 1.

    Returns:
        String like '72.4%'
    """
    return f"{prob * 100:.1f}%"


def risk_color(label: str) -> str:
    """
    Return a hex color string corresponding to a risk label.

    Args:
        label: 'Low', 'Medium', or 'High'.

    Returns:
        Hex color string.
    """
    colors = {
        "Low": "#2ecc71",      # Green
        "Medium": "#f39c12",   # Amber
        "High": "#e74c3c",     # Red
    }
    return colors.get(label, "#95a5a6")  # default grey


def risk_emoji(label: str) -> str:
    """Return an emoji corresponding to a risk label."""
    emojis = {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🔴",
    }
    return emojis.get(label, "⚪")


def sort_feature_importances(importances: dict, top_n: int = 15) -> tuple:
    """
    Sort feature importances and return top N as two parallel lists.

    Args:
        importances: Dict of {feature_name: importance_value}.
        top_n: Number of top features to return.

    Returns:
        Tuple of (feature_names, importance_values) both as lists, sorted descending.
    """
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [item[0].replace("cat__", "").replace("num__", "") for item in sorted_items]
    values = [item[1] for item in sorted_items]
    return names, values


def clean_feature_name(raw_name: str) -> str:
    """
    Clean a sklearn pipeline feature name for display (removes prefixes).

    Args:
        raw_name: e.g. 'cat__Gender_Female' or 'num__Age'

    Returns:
        Cleaned string e.g. 'Gender_Female' or 'Age'
    """
    return raw_name.replace("cat__", "").replace("num__", "")
