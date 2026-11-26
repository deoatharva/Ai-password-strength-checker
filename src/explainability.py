"""SHAP explanations for the password-strength models.

The app supplies the original, unscaled eight-feature DataFrame. Scaling is
performed here for SVM and Neural Network so that SHAP evaluates the same
input representation used by the app's prediction path.

KernelExplainer is model-agnostic and supports the four sklearn-style
estimators used by this project, provided each exposes predict_proba().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


FEATURE_EXPLANATIONS = {
    "length": ("Password length", "The number of characters in the password."),
    "uppercase_count": ("Uppercase letters", "The number of uppercase letters."),
    "lowercase_count": ("Lowercase letters", "The number of lowercase letters."),
    "digit_count": ("Numbers", "The number of numeric characters."),
    "special_count": ("Special characters", "The number of symbols such as !, @ or #."),
    "entropy": ("Entropy", "The estimated unpredictability of the password."),
    "pattern_count": ("Predictable patterns", "Detected sequences or repeated patterns."),
    "common_password": ("Common-password match", "Whether the password appears in the common-password database."),
}

FEATURE_NAMES = list(FEATURE_EXPLANATIONS.keys())


def _prepare_input(X):
    """Return a one-row DataFrame with the expected feature order."""
    if isinstance(X, pd.DataFrame):
        missing = [name for name in FEATURE_NAMES if name not in X.columns]
        if missing:
            raise ValueError(f"Missing SHAP feature columns: {missing}")
        return X.loc[:, FEATURE_NAMES].astype(float)

    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected input with {len(FEATURE_NAMES)} features in this order: {FEATURE_NAMES}"
        )
    return pd.DataFrame(arr, columns=FEATURE_NAMES)


def _model_input(model_name, X, scaler):
    """Match app preprocessing: only SVM and Neural Network are scaled."""
    if model_name in ("SVM", "Neural Network"):
        if scaler is None:
            raise ValueError(
                f"A fitted scaler is required to explain {model_name}."
            )
        transformed = scaler.transform(X)
        return np.asarray(transformed, dtype=float)
    return X


def _class_index(model, model_input):
    """Find the predicted class's probability column."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "This SHAP implementation requires a model with predict_proba(). "
            "For SVM, train SVC with probability=True."
        )

    predicted = np.asarray(model.predict(model_input)).reshape(-1)[0]
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise AttributeError("Model must expose classes_ to select the predicted-class probability.")

    matches = np.flatnonzero(np.asarray(classes) == predicted)
    if len(matches) == 0:
        # Handle equivalent string/numeric labels without guessing a class order.
        matches = np.flatnonzero(np.asarray([str(c) for c in classes]) == str(predicted))
    if len(matches) == 0:
        raise ValueError(f"Predicted class {predicted!r} was not found in model.classes_.")
    return int(matches[0]), predicted


def _make_background(X, size=20, random_state=42):
    """Create a small local reference set if training data was not supplied.

    This fallback is only an approximation. For research-grade explanations,
    pass representative rows from the model's actual training feature set.
    """
    row = X.iloc[0].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    rows = np.repeat(row.reshape(1, -1), size, axis=0)

    # Small perturbations around the observed password features. Keep binary
    # common_password and nonnegative count-like features within valid ranges.
    scales = np.maximum(np.abs(row) * 0.15, np.array([1, 1, 1, 1, 1, 0.5, 1, 1]))
    noise = rng.normal(0.0, scales, size=(size, len(row)))
    rows = rows + noise
    for idx in (0, 1, 2, 3, 4, 6):
        rows[:, idx] = np.maximum(0, np.round(rows[:, idx]))
    rows[:, 7] = np.clip(np.round(rows[:, 7]), 0, 1)
    rows[:, 5] = np.maximum(0, rows[:, 5])
    rows[0] = row
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def calculate_shap_values(
    model,
    model_name,
    X,
    scaler=None,
    background_data=None,
    nsamples=100,
):
    """Calculate SHAP values for the class predicted by the selected model.

    Parameters
    ----------
    model : fitted estimator
        Must implement predict() and predict_proba().
    model_name : str
        One of XGBoost, Random Forest, SVM, Neural Network.
    X : DataFrame or array
        One row of the original, unscaled eight extracted features.
    scaler : fitted scaler, optional
        Required for SVM and Neural Network.
    background_data : DataFrame or array, optional
        Representative ORIGINAL/unscaled training feature rows. If omitted,
        a local synthetic reference set is used (approximate only).
    nsamples : int
        Kernel SHAP sampling budget; larger values are slower but often more stable.

    Returns
    -------
    numpy.ndarray
        One SHAP value per feature, for the model's predicted class probability.
    """
    allowed = {"XGBoost", "Random Forest", "SVM", "Neural Network"}
    if model_name not in allowed:
        raise ValueError(f"Unsupported model_name {model_name!r}. Choose from {sorted(allowed)}")

    X_raw = _prepare_input(X)
    X_model = _model_input(model_name, X_raw, scaler)
    class_idx, _ = _class_index(model, X_model)

    if background_data is None:
        background_raw = _make_background(X_raw)
    else:
        background_raw = _prepare_input(background_data)

    background_model = _model_input(model_name, background_raw, scaler)

    def predicted_class_probability(raw_array):
        raw_df = pd.DataFrame(np.asarray(raw_array, dtype=float), columns=FEATURE_NAMES)
        model_array = _model_input(model_name, raw_df, scaler)
        probabilities = np.asarray(model.predict_proba(model_array), dtype=float)
        if probabilities.ndim != 2 or class_idx >= probabilities.shape[1]:
            raise ValueError("predict_proba returned an unexpected probability shape.")
        return probabilities[:, class_idx]

    explainer = shap.KernelExplainer(
        predicted_class_probability,
        np.asarray(background_raw, dtype=float),
        link="identity",
    )
    values = explainer.shap_values(
        np.asarray(X_raw, dtype=float),
        nsamples=nsamples,
        silent=True,
    )

    # KernelExplainer output shape varies across SHAP versions.
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values, dtype=float)
    values = np.squeeze(values)
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim > 1:
        values = values.reshape(-1, values.shape[-1])[0]
    if values.shape[0] != len(FEATURE_NAMES):
        raise ValueError(
            f"SHAP returned {values.shape[0]} values, expected {len(FEATURE_NAMES)}. "
            f"Returned shape: {np.asarray(values).shape}"
        )
    return values


def get_feature_contributions(shap_values, prediction=None):
    """Convert a SHAP vector into feature records sorted by absolute impact.

    The sign is relative to the predicted class probability: positive values
    support the predicted class; negative values oppose it.
    """
    values = np.asarray(shap_values, dtype=float).reshape(-1)
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} SHAP values, got {len(values)}.")

    contributions = []
    for name, value in zip(FEATURE_NAMES, values):
        display_name, description = FEATURE_EXPLANATIONS[name]
        if value > 0:
            direction = "supports"
            explanation = f"{description} This feature pushed the model toward its predicted class."
        elif value < 0:
            direction = "opposes"
            explanation = f"{description} This feature pushed the model away from its predicted class."
        else:
            direction = "neutral"
            explanation = f"{description} This feature had almost no measured effect on this prediction."

        contributions.append({
            "feature": name,
            "display_name": display_name,
            "shap_value": float(value),
            "direction": direction,
            "explanation": explanation,
        })

    return sorted(contributions, key=lambda item: abs(item["shap_value"]), reverse=True)


def generate_security_explanation(features, contributions, prediction):
    """Create readable strengths/weaknesses plus SHAP-specific summaries."""
    strong_classes = {"Strong", "Very Strong"}
    weak_classes = {"Weak", "Very Weak"}
    prediction_text = str(prediction)

    strengths = []
    weaknesses = []
    shap_strengths = []
    shap_weaknesses = []

    # Basic feature observations are kept separate from model attribution.
    if features.get("length", 0) < 12:
        weaknesses.append("The password is shorter than the recommended 12-character minimum.")
    elif features.get("length", 0) >= 14:
        strengths.append("The password has a good length, which generally makes guessing harder.")

    if features.get("pattern_count", 0) > 0:
        weaknesses.append("Predictable sequences or repeated patterns were detected.")
    if features.get("common_password", 0):
        weaknesses.append("This password appears in a common-password database; avoid using it.")
    if features.get("entropy", 0) >= 50:
        strengths.append("The estimated entropy is relatively high.")

    for item in contributions:
        value = item["shap_value"]
        if prediction_text in strong_classes:
            # Supporting a strong class is a positive security signal.
            if value > 0:
                shap_strengths.append(item)
            elif value < 0:
                shap_weaknesses.append(item)
        elif prediction_text in weak_classes:
            # Supporting a weak class is a negative security signal.
            if value > 0:
                shap_weaknesses.append(item)
            elif value < 0:
                shap_strengths.append(item)
        else:
            # For Medium/unknown labels, don't misrepresent class direction
            # as an absolute security judgement.
            if value > 0:
                shap_strengths.append({**item, "explanation": "This feature supports the Medium prediction. " + item["explanation"]})
            elif value < 0:
                shap_weaknesses.append({**item, "explanation": "This feature opposes the Medium prediction. " + item["explanation"]})

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "shap_strengths": sorted(shap_strengths, key=lambda x: abs(x["shap_value"]), reverse=True),
        "shap_weaknesses": sorted(shap_weaknesses, key=lambda x: abs(x["shap_value"]), reverse=True),
    }
