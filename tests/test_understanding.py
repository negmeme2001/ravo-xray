from ravo_core.understanding import EMPTY, FILLER_ONLY, PASS, clarification, classify, script_language


def test_classify_input() -> None:
    expected = {
        "": EMPTY,
        "  ": EMPTY,
        "؟": EMPTY,
        "🙂": EMPTY,
        "umm": FILLER_ONLY,
        "umm… يعني…": FILLER_ONLY,
        "Uhh, well": FILLER_ONLY,
        "يعني طيب": FILLER_ONLY,
        "milk?": PASS,
        "صداع؟": PASS,
        "Beyti": PASS,
        "do you have the": PASS,
        "asdf": PASS,
    }

    for text, result in expected.items():
        assert classify(text) == result


def test_script_language() -> None:
    assert script_language("عندكم لبن شوفان؟") == "ar"
    assert script_language("Do you have oat milk?") == "en"
    assert script_language("عايز حاجة للـ headache") == "ar"


def test_arabic_clarification() -> None:
    assert clarification("ar") == (
        "معلش، مفهمتش قصدك. بتدور على منتج زي اللبن أو الدوا، ولا عندك سؤال عن "
        "الترجيع أو التوصيل؟"
    )
