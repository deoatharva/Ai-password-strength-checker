# src/feature_extraction.py

import math
import re


FEATURE_NAMES = [
    "length",
    "uppercase_count",
    "lowercase_count",
    "digit_count",
    "special_count",
    "entropy",
    "pattern_count",
    "common_password"
]


DISPLAY_NAMES = {
    "length": "Password Length",
    "uppercase_count": "Uppercase Letters",
    "lowercase_count": "Lowercase Letters",
    "digit_count": "Numbers",
    "special_count": "Special Characters",
    "entropy": "Entropy",
    "pattern_count": "Predictable Patterns",
    "common_password": "Common Password"
}


def calculate_entropy(password):
    """
    Calculate entropy using the same formula
    used by the supplied dataset.

    entropy = length * log2(length)
    """

    length = len(password)

    if length <= 1:
        return 0.0

    return length * math.log2(length)


def detect_sequential_pattern(password):
    """
    Detect numeric/alphabetic sequences.
    """

    p = password.lower()

    sequences = [
        "0123456789",
        "1234567890",
        "9876543210",
        "abcdefghijklmnopqrstuvwxyz",
        "zyxwvutsrqponmlkjihgfedcba"
    ]

    for sequence in sequences:

        for i in range(len(sequence) - 2):

            part = sequence[i:i + 3]

            if part in p:
                return True

    return False


def detect_keyboard_pattern(password):
    """
    Detect common keyboard patterns.
    """

    p = password.lower()

    keyboard_patterns = [
        "qwerty",
        "qwertyui",
        "asdf",
        "asdfgh",
        "asdfghjkl",
        "zxcv",
        "zxcvbn",
        "qaz",
        "wsx",
        "edc",
        "rfv",
        "1qaz",
        "2wsx",
        "3edc",
        "qwe",
        "asd"
    ]

    return any(
        pattern in p
        for pattern in keyboard_patterns
    )


def detect_repeated_pattern(password):
    """
    Detect repeated characters.
    """

    if re.search(r"(.)\1{2,}", password):
        return True

    return False


def detect_year_pattern(password):
    """
    Detect four digit year patterns.
    """

    return bool(
        re.search(
            r"(19|20)\d{2}",
            password
        )
    )


def count_patterns(password):
    """
    Count predictable patterns.
    """

    count = 0

    if detect_sequential_pattern(password):
        count += 1

    if detect_keyboard_pattern(password):
        count += 1

    if detect_repeated_pattern(password):
        count += 1

    if detect_year_pattern(password):
        count += 1

    return count


def extract_features(
    password,
    common_passwords=None
):
    """
    Extract the complete ML feature vector.
    """

    if password is None:
        password = ""

    password = str(password)

    length = len(password)

    uppercase_count = sum(
        1 for char in password
        if char.isupper()
    )

    lowercase_count = sum(
        1 for char in password
        if char.islower()
    )

    digit_count = sum(
        1 for char in password
        if char.isdigit()
    )

    special_count = sum(
        1 for char in password
        if not char.isalnum()
    )

    entropy = calculate_entropy(
        password
    )

    pattern_count = count_patterns(
        password
    )

    common_password = 0

    if common_passwords is not None:

        common_password = int(
            password.lower()
            in common_passwords
        )

    return {
        "length": length,
        "uppercase_count": uppercase_count,
        "lowercase_count": lowercase_count,
        "digit_count": digit_count,
        "special_count": special_count,
        "entropy": entropy,
        "pattern_count": pattern_count,
        "common_password": common_password
    }


def features_to_dataframe(
    password,
    common_passwords=None
):

    import pandas as pd

    features = extract_features(
        password,
        common_passwords
    )

    return pd.DataFrame(
        [[
            features[name]
            for name in FEATURE_NAMES
        ]],
        columns=FEATURE_NAMES
    )