from seed.generate_catalog import BRANDS
from eval import metrics


def test_product_type_strips_single_word_brand():
    assert metrics.product_type("Juhayna Greek Yogurt", BRANDS) == "Greek Yogurt"


def test_product_type_strips_two_word_brand():
    assert metrics.product_type("Premium Select Whole Milk", BRANDS) == "Whole Milk"


def test_product_type_returns_name_when_no_brand_matches():
    assert metrics.product_type("Whole Milk", BRANDS) == "Whole Milk"


def test_is_relevant_product_with_relevant_names_takes_precedence():
    case = {"relevant_names": ["Juhayna Greek Yogurt"], "relevant_types": ["Greek Yogurt"]}
    assert metrics.is_relevant_product("Juhayna Greek Yogurt", case, BRANDS) is True
    assert metrics.is_relevant_product("Almarai Greek Yogurt", case, BRANDS) is False


def test_is_relevant_product_falls_back_to_relevant_types():
    case = {"relevant_types": ["Greek Yogurt"]}
    assert metrics.is_relevant_product("Almarai Greek Yogurt", case, BRANDS) is True
    assert metrics.is_relevant_product("Almarai Whole Milk", case, BRANDS) is False


def test_is_relevant_chunk_is_case_insensitive_and_whitespace_normalized():
    assert metrics.is_relevant_chunk("Refund the  difference\nplease", "refund   the difference")
    assert metrics.is_relevant_chunk("LINE\tONE\nLINE  TWO", "line one line two")


def test_is_relevant_chunk_returns_false_when_phrase_absent():
    assert metrics.is_relevant_chunk("some unrelated text", "refund the difference") is False


def test_reciprocal_rank_top_one():
    assert metrics.reciprocal_rank([True, False, False]) == 1.0


def test_reciprocal_rank_later_hit():
    assert metrics.reciprocal_rank([False, True, False]) == 0.5


def test_reciprocal_rank_no_hit():
    assert metrics.reciprocal_rank([False, False, False]) == 0.0


def test_hit_at_uses_rank_k():
    flags = [False, True, False]
    assert metrics.hit_at(flags, 1) == 0.0
    assert metrics.hit_at(flags, 2) == 1.0
    assert metrics.hit_at(flags, 3) == 1.0


def test_hit_at_zero_k():
    assert metrics.hit_at([True, False], 0) == 0.0


def test_diversity_counts_distinct_types():
    assert metrics.diversity(["Cola", "Cola", "Ice Cream"]) == 2
    assert metrics.diversity([]) == 0


def test_p50_odd_length_median():
    assert metrics.p50([3.5, 1.25, 2.75]) == 2.8


def test_p50_even_length_average_rounded():
    assert metrics.p50([1.0, 3.0, 2.0, 4.0]) == 2.5


def test_p50_empty():
    assert metrics.p50([]) == 0.0