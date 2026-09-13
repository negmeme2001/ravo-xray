"""Pure ranking and relevance metrics for the retrieval eval (no I/O)."""


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def product_type(name: str, brands: list[str]) -> str:
    """Strip the leading brand from a product name and return the rest.

    Brands can span multiple words (e.g. "Premium Select"), so longer
    matches are tried first.
    """
    for brand in sorted(brands, key=len, reverse=True):
        prefix = f"{brand} "
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def is_relevant_product(name: str, case: dict, brands: list[str]) -> bool:
    if case.get("relevant_names"):
        return name in case["relevant_names"]
    return product_type(name, brands) in case["relevant_types"]


def is_relevant_chunk(content: str, phrase: str) -> bool:
    return _normalize(phrase) in _normalize(content)


def reciprocal_rank(flags: list[bool]) -> float:
    for i, flag in enumerate(flags):
        if flag:
            return 1.0 / (i + 1)
    return 0.0


def hit_at(flags: list[bool], k: int) -> float:
    return 1.0 if any(flags[:k]) else 0.0


def diversity(types: list[str]) -> int:
    return len(set(types))


def p50(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 1)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 1)