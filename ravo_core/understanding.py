"""Deterministic checks for input that has no actionable request."""

import unicodedata

from ravo_core.prompts import message

EMPTY = "empty"
FILLER_ONLY = "filler_only"
PASS = "pass"

_FILLERS = {
    "um",
    "umm",
    "uh",
    "uhh",
    "uhm",
    "er",
    "erm",
    "hmm",
    "hm",
    "mm",
    "mmm",
    "ah",
    "eh",
    "well",
    "so",
    "like",
    "يعني",
    "امم",
    "اممم",
    "ممم",
    "اه",
    "آه",
    "اهه",
    "طيب",
    "طب",
    "بص",
    "بقى",
}


def _normalized_words(text: str) -> list[str]:
    characters = []
    for character in unicodedata.normalize("NFKC", text.lower()):
        category = unicodedata.category(character)
        if character == "ـ" or category == "Mn":
            continue
        characters.append(" " if category.startswith(("P", "S")) else character)
    return "".join(characters).split()


def classify(text: str) -> str:
    """Classify input as empty, filler-only, or suitable for model handling."""
    words = _normalized_words(text)
    if not words:
        return EMPTY
    if all(word in _FILLERS for word in words):
        return FILLER_ONLY
    return PASS


def _is_arabic_letter(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    )


def script_language(text: str) -> str:
    """Return Arabic when at least 30 percent of letters use Arabic script."""
    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    if letters and sum(_is_arabic_letter(character) for character in letters) / len(letters) >= 0.3:
        return "ar"
    return "en"


def clarification(language: str) -> str:
    """Return the canned clarification question for a detected language."""
    return message("clarification", language)
