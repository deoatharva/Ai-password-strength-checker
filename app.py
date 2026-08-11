# app.py

import os
import sys

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(
    BASE_DIR
)


from src.feature_extraction import (
    extract_features,
    FEATURE_NAMES
)

from src.preprocessing import (
    load_common_passwords
)

from src.explainability import (
    calculate_shap_values,
    get_feature_contributions,
    generate_security_explanation
)

from src.crack_time import (
    calculate_crack_time,
    get_crack_time_message
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Password Strength Checker",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_ORDER = [
    "Very Weak",
    "Weak",
    "Medium",
    "Strong",
    "Very Strong"
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #161b22;
        border: 1px solid #30363d;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .explanation-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #161b22;
        border: 1px solid #30363d;
        margin-top: 10px;
    }

    .big-time {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_all_models():

    models = {}

    models[
        "Random Forest"
    ] = joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "random_forest.pkl"
        )
    )

    models[
        "XGBoost"
    ] = joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "xgboost.pkl"
        )
    )

    models[
        "SVM"
    ] = joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "svm.pkl"
        )
    )

    models[
        "Neural Network"
    ] = joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "neural_network.pkl"
        )
    )

    scaler = joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "scaler.pkl"
        )
    )

    metadata = joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "model_metadata.pkl"
        )
    )

    return (
        models,
        scaler,
        metadata
    )


# ============================================================
# COMMON PASSWORDS
# ============================================================

@st.cache_resource
def load_common_password_database():

    return load_common_passwords(
        os.path.join(
            BASE_DIR,
            "data",
            "raw",
            "common_passwords.csv"
        ),
        [
            os.path.join(
                BASE_DIR,
                "data",
                "raw",
                "10_million_password_list_top_10000.txt"
            ),
            os.path.join(
                BASE_DIR,
                "data",
                "raw",
                "google-10000-english.txt"
            )
        ]
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="title">
        🔐 AI-Based Password Strength Checker
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Machine Learning + SHAP Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT AREA
# ============================================================

st.subheader(
    "Enter your password"
)

# ------------------------------------------------------------
# MODEL + PASSWORD
# ------------------------------------------------------------

model_col, password_col = st.columns(
    [1, 2],
    gap="large"
)

with model_col:

    model_name = st.selectbox(
        "Select AI Model",
        [
            "XGBoost",
            "Random Forest",
            "SVM",
            "Neural Network"
        ]
    )

with password_col:

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password..."
    )


# ------------------------------------------------------------
# BUTTON
# ------------------------------------------------------------

analyze = st.button(
    "🔍 Analyze Password",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not password:

        st.warning(
            "Please enter a password."
        )

        st.stop()

    # --------------------------------------------------------
    # LOAD EVERYTHING
    # --------------------------------------------------------

    models, scaler, metadata = (
        load_all_models()
    )

    common_passwords = (
        load_common_password_database()
    )

    model = models[
        model_name
    ]

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

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

    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

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
            ] = float(
                probability
            )

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

    else:

        X_scaled = scaler.transform(
            X
        )

        prediction = str(
            model.predict(
                X_scaled
            )[0]
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

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = (
        max(
            probability_dict.values()
        ) * 100
    )

    # ========================================================
    # STRENGTH RESULT
    # ========================================================

    st.subheader(
        "Password Strength"
    )

    if prediction == "Very Weak":

        st.error(
            "🔴 Password Strength: VERY WEAK"
        )

    elif prediction == "Weak":

        st.error(
            "🟠 Password Strength: WEAK"
        )

    elif prediction == "Medium":

        st.warning(
            "🟡 Password Strength: MEDIUM"
        )

    elif prediction == "Strong":

        st.success(
            "🟢 Password Strength: STRONG"
        )

    elif prediction == "Very Strong":

        st.success(
            "🔵 Password Strength: VERY STRONG"
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )

    with col2:

        st.metric(
            "Selected Model",
            model_name
        )

    # ========================================================
    # PROBABILITY
    # ========================================================

    st.subheader(
        "Prediction Probabilities"
    )

    probability_data = []

    for strength in CLASS_ORDER:

        probability_data.append({

            "Strength":
                strength,

            "Probability":
                probability_dict.get(
                    strength,
                    0
                ) * 100
        })

    probability_df = pd.DataFrame(
        probability_data
    )

    st.bar_chart(
        probability_df.set_index(
            "Strength"
        )
    )

    # ========================================================
    # CRACK TIME
    # ========================================================

    st.subheader(
        "⏱️ Estimated Password Crack Time"
    )

    crack_data = (
        calculate_crack_time(
            password
        )
    )

    crack_message = (
        get_crack_time_message(
            crack_data,
            features[
                "common_password"
            ] == 1
        )
    )

    st.info(
        "⚠️ Crack-time values are theoretical "
        "estimates. Real-world cracking speed "
        "depends on the hashing algorithm, "
        "hardware and attack method."
    )

    time_col1, time_col2 = st.columns(
        2
    )

    with time_col1:

        st.markdown(
            "### 🌐 Online Attack"
        )

        st.markdown(
            f'<div class="big-time">'
            f'{crack_data["online_time"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Assuming approximately 10 guesses "
            "per second."
        )

    with time_col2:

        st.markdown(
            "### ⚡ Offline Brute Force"
        )

        st.markdown(
            f'<div class="big-time">'
            f'{crack_data["offline_time"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Theoretical estimate using a "
            "high-speed offline attack."
        )

    st.write(
        crack_message
    )

    # ========================================================
    # COMMON PASSWORD
    # ========================================================

    if features[
        "common_password"
    ]:

        st.error(
            "🚨 Important: This password appears "
            "in a common-password database. "
            "A dictionary attack may find it much "
            "faster than the brute-force estimate."
        )

    # ========================================================
    # PASSWORD FEATURES
    # ========================================================

    st.subheader(
        "🔎 Password Analysis"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Length",
        features[
            "length"
        ]
    )

    col2.metric(
        "Uppercase",
        features[
            "uppercase_count"
        ]
    )

    col3.metric(
        "Lowercase",
        features[
            "lowercase_count"
        ]
    )

    col4.metric(
        "Numbers",
        features[
            "digit_count"
        ]
    )

    col5, col6, col7, col8 = (
        st.columns(4)
    )

    col5.metric(
        "Special Characters",
        features[
            "special_count"
        ]
    )

    col6.metric(
        "Entropy",
        f'{features["entropy"]:.2f}'
    )

    col7.metric(
        "Patterns",
        features[
            "pattern_count"
        ]
    )

    col8.metric(
        "Common Password",
        "YES"
        if features[
            "common_password"
        ]
        else "NO"
    )

    # ========================================================
    # SHAP EXPLAINABILITY
    # ========================================================

    st.subheader(
        "🧠 Why did the AI give this result?"
    )

    if model_name in [
        "XGBoost",
        "Random Forest"
    ]:

        try:

            shap_values = (
                calculate_shap_values(
                    model,
                    model_name,
                    X
                )
            )

            contributions = (
                get_feature_contributions(
                    shap_values
                )
            )

            explanation = (
                generate_security_explanation(
                    features,
                    contributions,
                    prediction
                )
            )

            # =================================================
            # SIMPLE HUMAN EXPLANATION
            # =================================================

            st.markdown(
                "### 🧑‍💻 Simple Explanation"
            )

            st.write(
                f"The AI classified this password as "
                f"**{prediction}** by looking at its "
                f"length, character variety, patterns, "
                f"entropy and whether it is commonly used."
            )

            # -------------------------------------------------
            # WEAKNESSES
            # -------------------------------------------------

            if explanation[
                "weaknesses"
            ]:

                st.markdown(
                    "### 🔴 What makes it weaker?"
                )

                for reason in explanation[
                    "weaknesses"
                ]:

                    st.error(
                        reason
                    )

            # -------------------------------------------------
            # STRENGTHS
            # -------------------------------------------------

            if explanation[
                "strengths"
            ]:

                st.markdown(
                    "### 🟢 What makes it stronger?"
                )

                for reason in explanation[
                    "strengths"
                ]:

                    st.success(
                        reason
                    )

            # =================================================
            # SHAP IN PLAIN ENGLISH
            # =================================================

            st.markdown(
                "### 📊 What is SHAP telling us?"
            )

            st.write(
                "SHAP shows which password features "
                "pushed the AI toward its final decision."
            )

            # -------------------------------------------------
            # NEGATIVE FACTORS
            # -------------------------------------------------

            if explanation[
                "shap_weaknesses"
            ]:

                st.markdown(
                    "#### 🔴 Factors hurting the password"
                )

                for item in explanation[
                    "shap_weaknesses"
                ][:5]:

                    st.error(
                        f"**{item['display_name']}**\n\n"
                        f"{item['explanation']}"
                    )

            # -------------------------------------------------
            # POSITIVE FACTORS
            # -------------------------------------------------

            if explanation[
                "shap_strengths"
            ]:

                st.markdown(
                    "#### 🟢 Factors helping the password"
                )

                for item in explanation[
                    "shap_strengths"
                ][:5]:

                    st.success(
                        f"**{item['display_name']}**\n\n"
                        f"{item['explanation']}"
                    )

            # =================================================
            # TECHNICAL SHAP
            # =================================================

            with st.expander(
                "🔬 Technical SHAP Details"
            ):

                st.write(
                    "These are the raw SHAP values "
                    "used by the explainability component. "
                    "They are mainly useful for technical "
                    "analysis and project evaluation."
                )

                shap_table = pd.DataFrame(
                    [
                        {

                            "Feature":
                                item[
                                    "display_name"
                                ],

                            "SHAP Value":
                                round(
                                    item[
                                        "shap_value"
                                    ],
                                    5
                                ),

                            "Impact":
                                (
                                    "Positive"
                                    if item[
                                        "shap_value"
                                    ] > 0
                                    else
                                    "Negative"
                                )
                        }

                        for item in contributions
                    ]
                )

                st.dataframe(
                    shap_table,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:

            st.warning(
                "SHAP explanation could not "
                "be generated for this prediction."
            )

            st.caption(
                f"Technical detail: {error}"
            )

    else:

        st.info(
            "Detailed SHAP explanation is currently "
            "available for XGBoost and Random Forest."
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "💡 How can you improve this password?"
    )

    recommendations = []

    if features[
        "length"
    ] < 12:

        recommendations.append(
            "Make the password longer. Aim for "
            "at least 12–14 characters."
        )

    if features[
        "uppercase_count"
    ] == 0:

        recommendations.append(
            "Add uppercase letters."
        )

    if features[
        "lowercase_count"
    ] == 0:

        recommendations.append(
            "Add lowercase letters."
        )

    if features[
        "digit_count"
    ] == 0:

        recommendations.append(
            "Add numbers."
        )

    if features[
        "special_count"
    ] == 0:

        recommendations.append(
            "Add special characters such as "
            "! @ # $ %."
        )

    if features[
        "pattern_count"
    ] > 0:

        recommendations.append(
            "Avoid predictable sequences, "
            "repeated characters and keyboard patterns."
        )

    if features[
        "common_password"
    ]:

        recommendations.append(
            "Do not use a password that appears "
            "in common-password lists."
        )

    if recommendations:

        for recommendation in recommendations:

            st.write(
                "• " + recommendation
            )

    else:

        st.success(
            "Your password satisfies the main "
            "security recommendations."
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    st.subheader(
        "📄 Final Security Report"
    )

    report = f"""
### AI Password Security Report

**Strength:** {prediction}

**AI Model:** {model_name}

**AI Confidence:** {confidence:.2f}%

**Estimated Online Crack Time:** {crack_data["online_time"]}

**Estimated Offline Crack Time:** {crack_data["offline_time"]}

### Password Characteristics

- Length: {features["length"]}
- Uppercase letters: {features["uppercase_count"]}
- Lowercase letters: {features["lowercase_count"]}
- Numbers: {features["digit_count"]}
- Special characters: {features["special_count"]}
- Entropy: {features["entropy"]:.2f}
- Predictable patterns: {features["pattern_count"]}
- Common password: {
    "Yes"
    if features["common_password"]
    else "No"
}

### AI Explanation

The AI predicted **{prediction}** based on the
password's learned feature patterns.

The SHAP analysis explains which features
helped or hurt this prediction.
"""

    st.markdown(
        report
    )