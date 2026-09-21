from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Sequence

_ARABIC_MARKS = re.compile(r"[\u0640\u064b-\u0652\u0670]")
_ARABIC_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_EXTRA_PUNCTUATION = "،؛؟«»…"
_PUNCTUATION = str.maketrans("", "", string.punctuation + _EXTRA_PUNCTUATION)


def normalize(text: str) -> str:
    normalized = _ARABIC_MARKS.sub("", text.lower()).translate(_ARABIC_FORMS)
    normalized = normalized.translate(_PUNCTUATION)
    return " ".join(normalized.split())


def keyword_recall(transcript: str, keywords: Sequence[Sequence[str]]) -> float | None:
    if not keywords:
        return None
    normalized_transcript = normalize(transcript)
    matches = sum(
        any(normalize(variant) in normalized_transcript for variant in group if normalize(variant))
        for group in keywords
    )
    return matches / len(keywords)


def cer(reference: str, hypothesis: str) -> float:
    expected = normalize(reference).replace(" ", "")
    actual = normalize(hypothesis).replace(" ", "")
    if not expected:
        return 0.0 if not actual else 1.0
    if not actual:
        return 1.0
    previous = list(range(len(actual) + 1))
    for row, expected_char in enumerate(expected, start=1):
        current = [row]
        for column, actual_char in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_char != actual_char),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def arabic_ratio(text: str) -> float:
    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    if not letters:
        return 0.0
    arabic = sum("\u0600" <= character <= "\u06ff" for character in letters)
    return arabic / len(letters)


def text_language(text: str) -> str:
    if not any(unicodedata.category(character).startswith("L") for character in text):
        return "none"
    ratio = arabic_ratio(text)
    if ratio >= 0.7:
        return "ar"
    if ratio <= 0.1:
        return "en"
    return "mixed"
