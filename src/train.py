# src/train.py

import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)


from src.feature_engineering import (
    prepare_training_dataset,
    CLASS_ORDER
)

from src.preprocessing import (
    load_common_passwords,
    save_processed_dataset
)


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "dataset.csv"
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AI PASSWORD STRENGTH CHECKER")
    print("FIVE CLASS MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\n[1/8] Loading dataset...")

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    print(
        "Original rows:",
        len(dataframe)
    )

    # --------------------------------------------------------
    # LOAD COMMON PASSWORDS
    # --------------------------------------------------------

    print(
        "\n[2/8] Loading common passwords..."
    )

    common_passwords = load_common_passwords(
        COMMON_PASSWORD_PATH,
        [
            TEN_MILLION_PATH,
            GOOGLE_PATH
        ]
    )

    print(
        "Common password entries:",
        len(common_passwords)
    )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    print(
        "\n[3/8] Extracting features..."
    )

    X, y = prepare_training_dataset(
        dataframe,
        common_passwords
    )

    print(
        "Training rows:",
        len(X)
    )

    print(
        "\nClass distribution:"
    )

    print(
        y.value_counts()
    )

    # --------------------------------------------------------
    # SAVE PROCESSED DATASET
    # --------------------------------------------------------

    print(
        "\n[4/8] Saving processed features..."
    )

    processed_path = os.path.join(
        PROCESSED_DIR,
        "password_features.csv"
    )

    save_processed_dataset(
        X,
        y,
        processed_path
    )

    print(
        "Saved:",
        processed_path
    )

    # --------------------------------------------------------
    # ENCODE FIVE CLASSES
    # --------------------------------------------------------

    label_to_number = {
        "Very Weak": 0,
        "Weak": 1,
        "Medium": 2,
        "Strong": 3,
        "Very Strong": 4
    }

    number_to_label = {
        0: "Very Weak",
        1: "Weak",
        2: "Medium",
        3: "Strong",
        4: "Very Strong"
    }

    y_encoded = y.map(
        label_to_number
    )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # Numeric labels for XGBoost
    y_train_encoded = y_train.map(
        label_to_number
    )

    y_test_encoded = y_test.map(
        label_to_number
    )

    # --------------------------------------------------------
    # SCALER
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    scaler_path = os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print(
        "\n[5/8] Training Random Forest..."
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_prediction = random_forest.predict(
        X_test
    )

    rf_accuracy = accuracy_score(
        y_test,
        rf_prediction
    )

    rf_precision, rf_recall, rf_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            rf_prediction,
            average="weighted",
            zero_division=0
        )
    )

    joblib.dump(
        random_forest,
        os.path.join(
            MODEL_DIR,
            "random_forest.pkl"
        )
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    print(
        "\n[6/8] Training XGBoost..."
    )

    xgboost_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    xgboost_model.fit(
        X_train,
        y_train_encoded
    )

    xgb_prediction_numeric = (
        xgboost_model.predict(
            X_test
        )
    )

    xgb_prediction = pd.Series(
        xgb_prediction_numeric
    ).map(
        number_to_label
    )

    xgb_accuracy = accuracy_score(
        y_test,
        xgb_prediction
    )

    xgb_precision, xgb_recall, xgb_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            xgb_prediction,
            average="weighted",
            zero_division=0
        )
    )

    joblib.dump(
        xgboost_model,
        os.path.join(
            MODEL_DIR,
            "xgboost.pkl"
        )
    )

    # ========================================================
    # SVM
    # ========================================================

    print(
        "\n[7/8] Training SVM..."
    )

    svm_model = SVC(
        kernel="rbf",
        C=10,
        probability=True,
        class_weight="balanced",
        random_state=42
    )

    svm_model.fit(
        X_train_scaled,
        y_train
    )

    svm_prediction = svm_model.predict(
        X_test_scaled
    )

    svm_accuracy = accuracy_score(
        y_test,
        svm_prediction
    )

    svm_precision, svm_recall, svm_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            svm_prediction,
            average="weighted",
            zero_division=0
        )
    )

    joblib.dump(
        svm_model,
        os.path.join(
            MODEL_DIR,
            "svm.pkl"
        )
    )

    # ========================================================
    # NEURAL NETWORK
    # ========================================================

    print(
        "\n[8/8] Training Neural Network..."
    )

    neural_network = MLPClassifier(
        hidden_layer_sizes=(
            128,
            64,
            32
        ),
        activation="relu",
        solver="adam",
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=42
    )

    neural_network.fit(
        X_train_scaled,
        y_train
    )

    nn_prediction = neural_network.predict(
        X_test_scaled
    )

    nn_accuracy = accuracy_score(
        y_test,
        nn_prediction
    )

    nn_precision, nn_recall, nn_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            nn_prediction,
            average="weighted",
            zero_division=0
        )
    )

    joblib.dump(
        neural_network,
        os.path.join(
            MODEL_DIR,
            "neural_network.pkl"
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = pd.DataFrame([
        {
            "model": "Random Forest",
            "accuracy": rf_accuracy,
            "precision": rf_precision,
            "recall": rf_recall,
            "f1_score": rf_f1
        },
        {
            "model": "XGBoost",
            "accuracy": xgb_accuracy,
            "precision": xgb_precision,
            "recall": xgb_recall,
            "f1_score": xgb_f1
        },
        {
            "model": "SVM",
            "accuracy": svm_accuracy,
            "precision": svm_precision,
            "recall": svm_recall,
            "f1_score": svm_f1
        },
        {
            "model": "Neural Network",
            "accuracy": nn_accuracy,
            "precision": nn_precision,
            "recall": nn_recall,
            "f1_score": nn_f1
        }
    ])

    results = results.sort_values(
        "f1_score",
        ascending=False
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # XGBOOST CLASSIFICATION REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("XGBOOST CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            xgb_prediction,
            labels=CLASS_ORDER,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {

        "feature_names": X.columns.tolist(),

        "classes": CLASS_ORDER,

        "label_to_number": label_to_number,

        "number_to_label": number_to_label,

        "best_model": results.iloc[0]["model"]
    }

    joblib.dump(
        metadata,
        os.path.join(
            MODEL_DIR,
            "model_metadata.pkl"
        )
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nModels saved in:",
        MODEL_DIR
    )


if __name__ == "__main__":
    main()