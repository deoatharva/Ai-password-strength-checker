# src/explainability.py

import numpy as np
import shap

from src.feature_extraction import (
    FEATURE_NAMES,
    DISPLAY_NAMES
)


# ============================================================
# FRIENDLY FEATURE DESCRIPTIONS
# ============================================================

FEATURE_EXPLANATIONS = {

    "length": {

        "positive":
            "The password is long enough to increase "
            "the number of possible combinations.",

        "negative":
            "The password is short, which makes the "
            "number of possible guesses much smaller."
    },

    "uppercase_count": {

        "positive":
            "Using uppercase letters increases the "
            "variety of characters an attacker has to consider.",

        "negative":
            "The password has little or no uppercase "
            "letter variety."
    },

    "lowercase_count": {

        "positive":
            "Lowercase letters contribute to the "
            "overall character variety.",

        "negative":
            "The password relies heavily on a limited "
            "character set."
    },

    "digit_count": {

        "positive":
            "Numbers add another type of character "
            "that increases the possible combinations.",

        "negative":
            "The numbers do not provide much additional "
            "complexity in this password."
    },

    "special_count": {

        "positive":
            "Special characters add another character "
            "category and make guessing more difficult.",

        "negative":
            "The password has few or no special characters."
    },

    "entropy": {

        "positive":
            "The password has a larger estimated search "
            "space according to the model's entropy feature.",

        "negative":
            "The estimated search space is relatively small."
    },

    "pattern_count": {

        "positive":
            "The model found fewer predictable patterns, "
            "which helps the password.",

        "negative":
            "The password contains predictable patterns "
            "that can make guessing easier."
    },

    "common_password": {

        "positive":
            "The password was not found in the common-password "
            "database.",

        "negative":
            "The password appears in a common-password database, "
            "which is a major security weakness."
    }
}


# ============================================================
# XGBOOST SHAP
# ============================================================

def calculate_xgboost_shap(
    model,
    X
):

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    prediction = int(
        model.predict(X)[0]
    )

    values = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Different SHAP versions return different formats.
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        values = np.asarray(
            shap_values[prediction][0]
        )

    elif values.ndim == 3:

        values = values[
            0,
            :,
            prediction
        ]

    elif values.ndim == 2:

        values = values[0]

    return values


# ============================================================
# RANDOM FOREST SHAP
# ============================================================

def calculate_tree_shap(
    model,
    X
):

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    prediction = model.predict(
        X
    )[0]

    classes = list(
        model.classes_
    )

    prediction_index = classes.index(
        prediction
    )

    values = np.asarray(
        shap_values
    )

    if isinstance(
        shap_values,
        list
    ):

        values = np.asarray(
            shap_values[
                prediction_index
            ][0]
        )

    elif values.ndim == 3:

        values = values[
            0,
            :,
            prediction_index
        ]

    elif values.ndim == 2:

        values = values[0]

    return values


# ============================================================
# SHAP SELECTOR
# ============================================================

def calculate_shap_values(
    model,
    model_name,
    X
):

    if model_name == "XGBoost":

        return calculate_xgboost_shap(
            model,
            X
        )

    if model_name == "Random Forest":

        return calculate_tree_shap(
            model,
            X
        )

    return None


# ============================================================
# FEATURE CONTRIBUTIONS
# ============================================================

def get_feature_contributions(
    shap_values
):

    if shap_values is None:

        return []

    contributions = []

    for feature, value in zip(
        FEATURE_NAMES,
        shap_values
    ):

        value = float(value)

        if value > 0:

            direction = "positive"

        else:

            direction = "negative"

        explanation = (
            FEATURE_EXPLANATIONS
            .get(
                feature,
                {}
            )
            .get(
                direction,
                ""
            )
        )

        contributions.append({

            "feature": feature,

            "display_name":
                DISPLAY_NAMES.get(
                    feature,
                    feature
                ),

            "shap_value": value,

            "absolute_value":
                abs(value),

            "direction":
                direction,

            "explanation":
                explanation
        })

    contributions.sort(
        key=lambda x:
        x["absolute_value"],
        reverse=True
    )

    return contributions


# ============================================================
# USER-FRIENDLY EXPLANATION
# ============================================================

def generate_security_explanation(
    features,
    contributions,
    prediction
):

    weaknesses = []
    strengths = []

    # ========================================================
    # COMMON PASSWORD
    # ========================================================

    if features[
        "common_password"
    ] == 1:

        weaknesses.append(
            "This password appears in a list of "
            "commonly used passwords. Attackers "
            "often try these passwords first."
        )

    # ========================================================
    # LENGTH
    # ========================================================

    length = features[
        "length"
    ]

    if length <= 5:

        weaknesses.append(
            f"The password is only {length} "
            "characters long. This gives an attacker "
            "a much smaller number of possibilities "
            "to try."
        )

    elif length < 12:

        weaknesses.append(
            f"The password is {length} characters "
            "long. Increasing the length would make "
            "brute-force guessing harder."
        )

    else:

        strengths.append(
            f"The password has {length} characters, "
            "which gives it a larger search space."
        )

    # ========================================================
    # CHARACTER TYPES
    # ========================================================

    types = 0

    if features["uppercase_count"] > 0:
        types += 1

    if features["lowercase_count"] > 0:
        types += 1

    if features["digit_count"] > 0:
        types += 1

    if features["special_count"] > 0:
        types += 1

    if types == 1:

        weaknesses.append(
            "The password uses only one type of "
            "character. Adding different character "
            "types would make guessing harder."
        )

    elif types == 2:

        weaknesses.append(
            "The password uses only two character "
            "types. More variety would improve security."
        )

    elif types >= 4:

        strengths.append(
            "The password combines uppercase letters, "
            "lowercase letters, numbers and special "
            "characters."
        )

    # ========================================================
    # PATTERNS
    # ========================================================

    if features[
        "pattern_count"
    ] >= 2:

        weaknesses.append(
            "The password contains multiple "
            "predictable patterns. Patterns can make "
            "passwords easier to guess than they appear."
        )

    elif features[
        "pattern_count"
    ] == 1:

        weaknesses.append(
            "A predictable pattern was detected. "
            "Avoid sequences, repeated characters "
            "and common keyboard patterns."
        )

    else:

        strengths.append(
            "No major predictable pattern was detected."
        )

    # ========================================================
    # SHAP CONTRIBUTIONS
    # ========================================================

    shap_strengths = []
    shap_weaknesses = []

    for item in contributions:

        if item[
            "shap_value"
        ] > 0:

            shap_strengths.append(
                item
            )

        elif item[
            "shap_value"
        ] < 0:

            shap_weaknesses.append(
                item
            )

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "shap_strengths": shap_strengths,
        "shap_weaknesses": shap_weaknesses
    }