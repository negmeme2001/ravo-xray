from seed.ingest_pdfs import chunk_text
from seed.ingest_products import product_text


def test_product_text_full_mode():
    p = {"name": "Almarai Whole Milk", "category": "Dairy", "description": "fresh whole milk (1L)"}
    assert product_text(p, "full") == "Almarai Whole Milk | Dairy | fresh whole milk (1L)"


def test_product_text_name_mode():
    p = {"name": "Almarai Whole Milk", "category": "Dairy", "description": "fresh whole milk (1L)"}
    assert product_text(p, "name") == "Almarai Whole Milk"


def test_chunk_text_sliding_window():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks == [
        "word0 word1 word2 word3 word4",
        "word3 word4 word5 word6 word7",
        "word6 word7 word8 word9",
    ]


def test_chunk_text_keeps_stub_at_least_half_chunk():
    text = " ".join(f"word{i}" for i in range(8))
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks == [
        "word0 word1 word2 word3 word4",
        "word3 word4 word5 word6 word7",
        "word6 word7",
    ]