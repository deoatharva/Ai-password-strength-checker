# src/predict.py

import os
import joblib
import pandas as pd

from src.feature_extraction import (
    extract_features,
    FEATURE_NAMES
)

from src.preprocessing import (
    load_common_passwords
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

COMMON_PASSWORD_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "common_passwords.csv"
)

TEN_MILLION_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "10_million_password_list_top_10000.txt"
)

GOOGLE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "google-10000-english.txt"
)


# ============================================================
# LOAD COMMON PASSWORDS
# ============================================================

def get_common_passwords():

    return load_common_passwords(
        COMMON_PASSWORD_PATH,
        [
            TEN_MILLION_PATH,
            GOOGLE_PATH
        ]
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    models = {}

    models["Random Forest"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "random_forest.pkl"
        )
    )

    models["XGBoost"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "xgboost.pkl"
        )
    )

    models["SVM"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "svm.pkl"
        )
    )

    models["Neural Network"] = joblib.load(
        os.path.join(
            MODEL_DIR,
            "neural_network.pkl"
        )
    )

    scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "scaler.pkl"
        )
    )

    metadata = joblib.load(
        os.path.join(
            MODEL_DIR,
            "model_metadata.pkl"
        )
    )

    return (
        models,
        scaler,
        metadata
    )


# ============================================================
# PREDICT
# ============================================================

def predict_password(
    password,
    model_name="XGBoost"
):

    common_passwords = (
        get_common_passwords()
    )

    features = extract_features(
        password,
        common_passwords
    )

    X = pd.DataFrame(
        [[
            features[name]
            for name in FEATURE_NAMES
        ]],
        columns=FEATURE_NAMES
    )

    models, scaler, metadata = (
        load_models()
    )

    model = models[
        model_name
    ]

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    if model_name == "XGBoost":

        numeric_prediction = int(
            model.predict(X)[0]
        )

        prediction = (
            metadata[
                "number_to_label"
            ][numeric_prediction]
        )

        probabilities = (
            model.predict_proba(X)[0]
        )

        probability_dict = {}

        for index, probability in enumerate(
            probabilities
        ):

            label = metadata[
                "number_to_label"
            ][index]

            probability_dict[
                label
            ] = float(probability)

    # --------------------------------------------------------
    # TREE MODEL
    # --------------------------------------------------------

    elif model_name == "Random Forest":

        prediction = str(
            model.predict(X)[0]
        )

        probabilities = (
            model.predict_proba(X)[0]
        )

        probability_dict = {
            str(cls): float(prob)
            for cls, prob in zip(
                model.classes_,
                probabilities
            )
        }

    # --------------------------------------------------------
    # SVM / NEURAL NETWORK
    # --------------------------------------------------------

    else:

        X_scaled = scaler.transform(X)

        prediction = str(
            model.predict(X_scaled)[0]
        )

        probabilities = (
            model.predict_proba(
                X_scaled
            )[0]
        )

        probability_dict = {
            str(cls): float(prob)
            for cls, prob in zip(
                model.classes_,
                probabilities
            )
        }

    return {
        "prediction": prediction,
        "probabilities": probability_dict,
        "features": features,
        "feature_vector": X
    }