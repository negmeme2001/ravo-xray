"""
Generate a synthetic grocery catalog big enough that indexing and latency
actually mean something. Ravo's real catalogs are retailer product feeds
(ingested via the mapping + ai-integration services); this stands in for one.

Writes catalog.jsonl — one product per line.

Usage:
    uv run python -m seed.generate_catalog       # writes seed/catalog.jsonl
"""

import json
import random
from pathlib import Path

random.seed(42)

# (category, [products], [descriptor words]) — enough variety that semantic
# search has something to be smart about.
CATEGORIES = {
    "Dairy": (
        ["Whole Milk", "Skimmed Milk", "Almond Milk", "Oat Milk", "Soy Milk",
         "Greek Yogurt", "Cheddar Cheese", "Mozzarella", "Butter", "Cream Cheese",
         "Cottage Cheese", "Feta", "Sour Cream", "Whipping Cream", "Labneh"],
        ["fresh", "chilled", "organic", "low-fat", "full-fat", "lactose-free"],
    ),
    "Bakery": (
        ["White Bread", "Whole Wheat Bread", "Baguette", "Croissant", "Bagel",
         "Pita Bread", "Sourdough Loaf", "Dinner Rolls", "Muffin", "Donut"],
        ["freshly baked", "soft", "crusty", "artisan", "sliced"],
    ),
    "Beverages": (
        ["Orange Juice", "Apple Juice", "Cola", "Sparkling Water", "Mineral Water",
         "Green Tea", "Black Coffee Beans", "Instant Coffee", "Energy Drink",
         "Iced Tea", "Lemonade", "Coconut Water"],
        ["refreshing", "sugar-free", "cold", "sparkling", "natural"],
    ),
    "PainRelief": (
        ["Paracetamol Tablets", "Ibuprofen Tablets", "Aspirin", "Headache Relief Gel",
         "Migraine Relief Caplets", "Cold & Flu Powder", "Muscle Rub Cream"],
        ["fast-acting", "extra-strength", "for headaches and pain", "non-drowsy"],
    ),
    "Cleaning": (
        ["Dish Soap", "Laundry Detergent", "Floor Cleaner", "Glass Cleaner",
         "Bleach", "Sponges", "Trash Bags", "Fabric Softener"],
        ["antibacterial", "lemon-scented", "concentrated", "eco-friendly"],
    ),
    "Snacks": (
        ["Potato Chips", "Chocolate Bar", "Salted Peanuts", "Granola Bar",
         "Rice Crackers", "Cookies", "Popcorn", "Trail Mix", "Pretzels"],
        ["crunchy", "sweet", "salty", "healthy", "family-size"],
    ),
    "Frozen": (
        ["Frozen Pizza", "Ice Cream", "Frozen Vegetables", "Frozen Berries",
         "Fish Fingers", "Frozen Fries", "Frozen Peas"],
        ["frozen", "ready-to-cook", "family pack"],
    ),
    "Household": (
        ["Insulated Cooler Bag", "Paper Towels", "Aluminum Foil", "Cling Film",
         "Batteries AA", "Light Bulb", "Scented Candle", "Reusable Water Bottle"],
        ["durable", "value pack", "everyday", "keeps things cold"],
    ),
}

BRANDS = ["Juhayna", "Almarai", "President", "Nestle", "Domty", "Beyti",
          "Panda", "Lamar", "Generic", "Premium Select"]


def make_catalog(target: int = 1500) -> list[dict]:
    rows, pc = [], 10000
    while len(rows) < target:
        cat = random.choice(list(CATEGORIES))
        products, descriptors = CATEGORIES[cat]
        name = random.choice(products)
        brand = random.choice(BRANDS)
        desc = random.choice(descriptors)
        size = random.choice(["250ml", "500ml", "1L", "250g", "500g", "1kg", "pack of 6", "single"])
        pc += 1
        rows.append({
            "code": f"PC{pc}",
            "name": f"{brand} {name}",
            "category": cat,
            "brand": brand,
            "description": f"{desc} {name.lower()} ({size})",
            "price_egp": round(random.uniform(15, 500), 2),
            "stock": random.choice([0, 0, 3, 12, 40, 120, 200]),  # ~28% out of stock
        })
    return rows


if __name__ == "__main__":
    rows = make_catalog()
    out_path = Path(__file__).parent / "catalog.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    in_stock = sum(1 for r in rows if r["stock"] > 0)
    print(f"Wrote catalog.jsonl — {len(rows)} products, {in_stock} in stock, "
          f"{len(rows) - in_stock} out of stock, across {len(CATEGORIES)} categories.")
