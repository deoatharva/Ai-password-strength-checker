# src/crack_time.py

import math
import re


# ============================================================
# ATTACK SPEED ASSUMPTIONS
# ============================================================
#
# These are estimates, not guarantees.
#
# Online attack:
# Very limited guessing because login systems normally
# implement rate limiting / lockouts.
#
# Offline attack:
# Much faster because an attacker may have access to
# password hashes and can attempt guesses locally.
#
# The actual speed depends heavily on:
# - Hashing algorithm
# - Hardware
# - Password hashing configuration
# - Number of attackers/devices
#

ONLINE_GUESSES_PER_SECOND = 10

OFFLINE_GUESSES_PER_SECOND = 10_000_000_000


# ============================================================
# CHARACTER SET
# ============================================================

def get_character_set_size(password):

    size = 0

    has_lowercase = bool(
        re.search(r"[a-z]", password)
    )

    has_uppercase = bool(
        re.search(r"[A-Z]", password)
    )

    has_numbers = bool(
        re.search(r"[0-9]", password)
    )

    has_special = bool(
        re.search(r"[^a-zA-Z0-9]", password)
    )

    if has_lowercase:
        size += 26

    if has_uppercase:
        size += 26

    if has_numbers:
        size += 10

    if has_special:
        # Common printable special characters
        size += 32

    return size


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(seconds):

    if seconds < 1:
        return "Less than a second"

    if seconds < 60:
        return f"{seconds:.1f} seconds"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.1f} minutes"

    hours = minutes / 60

    if hours < 24:
        return f"{hours:.1f} hours"

    days = hours / 24

    if days < 365:
        return f"{days:.1f} days"

    years = days / 365

    if years < 1000:
        return f"{years:.1f} years"

    if years < 1_000_000:
        return f"{years / 1000:.1f} thousand years"

    if years < 1_000_000_000:
        return f"{years / 1_000_000:.1f} million years"

    return f"{years / 1_000_000_000:.1f} billion years"


# ============================================================
# CRACK TIME
# ============================================================

def calculate_crack_time(password):

    if not password:

        return {
            "character_set_size": 0,
            "possible_combinations": 0,
            "online_seconds": 0,
            "offline_seconds": 0,
            "online_time": "Unknown",
            "offline_time": "Unknown"
        }

    length = len(password)

    character_set_size = (
        get_character_set_size(password)
    )

    if character_set_size == 0:

        return {
            "character_set_size": 0,
            "possible_combinations": 0,
            "online_seconds": 0,
            "offline_seconds": 0,
            "online_time": "Unknown",
            "offline_time": "Unknown"
        }

    # --------------------------------------------------------
    # TOTAL POSSIBLE COMBINATIONS
    # --------------------------------------------------------

    possible_combinations = (
        character_set_size ** length
    )

    # --------------------------------------------------------
    # AVERAGE CRACK TIME
    #
    # On average, half of the search space
    # needs to be searched.
    # --------------------------------------------------------

    average_guesses = (
        possible_combinations / 2
    )

    online_seconds = (
        average_guesses
        / ONLINE_GUESSES_PER_SECOND
    )

    offline_seconds = (
        average_guesses
        / OFFLINE_GUESSES_PER_SECOND
    )

    return {

        "character_set_size":
            character_set_size,

        "possible_combinations":
            possible_combinations,

        "online_seconds":
            online_seconds,

        "offline_seconds":
            offline_seconds,

        "online_time":
            format_time(
                online_seconds
            ),

        "offline_time":
            format_time(
                offline_seconds
            )
    }


# ============================================================
# SECURITY INTERPRETATION
# ============================================================

def get_crack_time_message(
    crack_data,
    common_password=False
):

    if common_password:

        return (
            "⚠️ This password appears in a "
            "common-password list. A dictionary "
            "attack could find it much faster than "
            "the theoretical brute-force estimate."
        )

    offline_seconds = (
        crack_data["offline_seconds"]
    )

    if offline_seconds < 1:

        return (
            "🔴 This password could potentially "
            "be guessed very quickly by a fast "
            "offline attack."
        )

    if offline_seconds < 60:

        return (
            "🔴 The estimated offline crack time "
            "is very short."
        )

    if offline_seconds < 3600:

        return (
            "🟠 The password may not provide "
            "enough protection against fast "
            "offline guessing."
        )

    if offline_seconds < 86400:

        return (
            "🟡 The password has some resistance, "
            "but could still be improved."
        )

    if offline_seconds < 365 * 86400:

        return (
            "🟢 The password has reasonable "
            "resistance to brute-force guessing."
        )

    return (
        "🔵 The theoretical brute-force search "
        "space is very large."
    )