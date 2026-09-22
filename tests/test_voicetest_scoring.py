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


def _frames(*loud):
    """10 ms frames at 24 kHz: loud (amplitude 3000) or silent."""
    import array
    out = b""
    for is_loud in loud:
        out += array.array("h", [3000 if is_loud else 0] * 240).tobytes()
    return out


def test_speech_span_ignores_leading_and_trailing_silence():
    from voicetest.audio import speech_span
    assert speech_span(_frames(False, False, True, False, True, False, False)) == (2, 4)


def test_speech_span_of_silence_covers_the_whole_clip():
    from voicetest.audio import speech_span
    assert speech_span(_frames(False, False, False)) == (0, 2)


def test_pcm_rms_reads_int16_frames_as_livekit_sends_them():
    from array import array

    from voicetest.audio import pcm_rms

    samples = array("h", [1000, -1000] * 120)
    assert pcm_rms(memoryview(samples)) == pytest.approx(1000.0)
    assert pcm_rms(samples.tobytes()) == pytest.approx(1000.0)
