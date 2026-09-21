import pytest

from voicetest.scoring import arabic_ratio, cer, keyword_recall, normalize, text_language


def test_normalize_arabic_forms_diacritics_and_punctuation() -> None:
    assert normalize("أَإِآُ ـ ى ة ؤ ئ،؛؟ «نَعَم»…") == "ااا ي ه و ي نعم"


def test_normalize_ascii_case_and_punctuation() -> None:
    assert normalize("Hello,  WORLD!") == "hello world"


def test_keyword_recall_accepts_variants_and_normalized_arabic() -> None:
    keywords = [["want", "عايز"], ["حاجه"], ["headache", "صداع"]]
    assert keyword_recall("عَايِز حاجة للـ headache", keywords) == 1.0


def test_keyword_recall_reports_partial_and_empty_groups() -> None:
    assert keyword_recall("oat drink", [["oat"], ["milk"]]) == 0.5
    assert keyword_recall("anything", []) is None


def test_cer() -> None:
    assert cer("Same text", "same text") == 0.0
    assert cer("abc", "") == 1.0
    assert cer("kitten", "sitten") == pytest.approx(1 / 6)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain English", "en"),
        ("كلام عربي", "ar"),
        ("عايز حاجة للـ headache", "mixed"),
        ("123 !?", "none"),
    ],
)
def test_text_language(text: str, expected: str) -> None:
    assert text_language(text) == expected


def test_arabic_ratio() -> None:
    assert arabic_ratio("English") == 0.0
    assert arabic_ratio("عربي") == 1.0
    assert 0.1 < arabic_ratio("عايز حاجة للـ headache") < 0.7
    assert arabic_ratio("123") == 0.0
