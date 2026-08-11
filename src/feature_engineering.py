# src/feature_engineering.py

import pandas as pd

from src.feature_extraction import (
    extract_features,
    FEATURE_NAMES
)


# ============================================================
# FIVE CLASS LABELS
# ============================================================

LABEL_MAPPING = {
    "Very week": "Very Weak",
    "Very weak": "Very Weak",
    "Week": "Weak",
    "Weak": "Weak",
    "Average": "Medium",
    "Medium": "Medium",
    "Strong": "Strong",
    "Very strong": "Very Strong",
    "Very Strong": "Very Strong"
}


CLASS_ORDER = [
    "Very Weak",
    "Weak",
    "Medium",
    "Strong",
    "Very Strong"
]


def create_feature_dataset(
    dataframe,
    common_passwords
):

    rows = []

    for password in dataframe["password"]:

        features = extract_features(
            password,
            common_passwords
        )

        rows.append(features)

    return pd.DataFrame(
        rows,
        columns=FEATURE_NAMES
    )


def prepare_training_dataset(
    dataframe,
    common_passwords
):

    dataframe = dataframe.copy()

    # --------------------------------------------------------
    # CLEAN PASSWORDS
    # --------------------------------------------------------

    dataframe = dataframe.dropna(
        subset=["password"]
    )

    dataframe["password"] = (
        dataframe["password"]
        .astype(str)
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE PASSWORDS
    # --------------------------------------------------------

    dataframe = dataframe.drop_duplicates(
        subset=["password"]
    )

    # --------------------------------------------------------
    # MAP ORIGINAL DATASET LABELS
    # --------------------------------------------------------

    dataframe["target"] = (
        dataframe["class_strength"]
        .map(LABEL_MAPPING)
    )

    # Remove rows with invalid labels
    dataframe = dataframe.dropna(
        subset=["target"]
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = create_feature_dataset(
        dataframe,
        common_passwords
    )

    y = dataframe[
        "target"
    ].reset_index(drop=True)

    return X, y