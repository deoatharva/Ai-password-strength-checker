# src/preprocessing.py

import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler


def load_dataset(path):

    return pd.read_csv(path)


def load_common_passwords(
    csv_path,
    text_paths=None
):

    passwords = set()

    # --------------------------------------------------------
    # COMMON PASSWORD CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            csv_path
        )

        if "password" in df.columns:

            passwords.update(
                df["password"]
                .dropna()
                .astype(str)
                .str.lower()
                .tolist()
            )

    except Exception as error:

        print(
            "Warning loading common_passwords.csv:",
            error
        )

    # --------------------------------------------------------
    # ADD TEXT PASSWORD LISTS
    # --------------------------------------------------------

    if text_paths:

        for path in text_paths:

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as file:

                    for line in file:

                        password = (
                            line.strip()
                            .lower()
                        )

                        if password:
                            passwords.add(
                                password
                            )

            except FileNotFoundError:

                print(
                    f"Warning: {path} not found"
                )

    return passwords


def scale_features(
    X_train,
    X_test
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


def save_scaler(
    scaler,
    path
):

    joblib.dump(
        scaler,
        path
    )


def save_processed_dataset(
    X,
    y,
    path
):

    dataframe = X.copy()

    dataframe["target"] = y.values

    dataframe.to_csv(
        path,
        index=False
    )