# src/evaluation.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from src.feature_engineering import (
    prepare_training_dataset
)
from src.preprocessing import (
    load_common_passwords
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


def evaluate_random_forest():

    dataset_path = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "dataset.csv"
    )

    common_path = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "common_passwords.csv"
    )

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "random_forest.pkl"
    )

    output_path = os.path.join(
        BASE_DIR,
        "outputs",
        "confusion_matrix.png"
    )

    df = pd.read_csv(
        dataset_path
    )

    common_passwords = load_common_passwords(
        common_path
    )

    X, y = prepare_training_dataset(
        df,
        common_passwords
    )

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = joblib.load(
        model_path
    )

    predictions = model.predict(
        X_test
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "Weak",
            "Medium",
            "Strong"
        ]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Weak",
            "Medium",
            "Strong"
        ]
    )

    display.plot()

    plt.title(
        "Password Strength Classification"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":
    evaluate_random_forest()