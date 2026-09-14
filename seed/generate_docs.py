"""
Generate 3 sample PDF documents (a store policy, a safety manual, an FAQ) so the
lab runs without you having to supply your own files. Each PDF is written to docs/.

In Ravo, "collections" are real documents like these — retailer guides, menus,
store FAQs — pulled in through the catalog pipeline and stored in the
searchable document store. These three stand in for that.

    uv run python -m seed.generate_docs
"""

import os
from fpdf import FPDF

DOCS = {
    "store_policy": {
        "title": "Store Policy",
        "sections": [
            (
                "Opening Hours",
                "Our store is open seven days a week to serve you. From Monday through "
                "Saturday we are open from eight in the morning until ten at night. On "
                "Sundays we open at nine and close at eight in the evening. The pharmacy "
                "counter follows its own schedule and is staffed between nine in the "
                "morning and nine at night on weekdays, with shorter hours on Sunday. "
                "During public holidays the store operates on reduced hours, which are "
                "published on our website and at the main entrance at least one week in "
                "advance.",
            ),
            (
                "Returns and Refunds",
                "We accept returns and issue refunds within thirty days of purchase when "
                "you present the original receipt. Electronics and small appliances must "
                "be returned within fourteen days and must be in their original packaging "
                "with all accessories included. Food and perishable items that have been "
                "opened cannot be returned for hygiene reasons. Perishable goods that are "
                "unopened and still within their sell-by date can be exchanged for the "
                "same product. Refunds are returned to the original payment method and "
                "normally appear within five business days. Gift receipts are accepted "
                "and the refund is issued as a store credit.",
            ),
            (
                "Price Match",
                "If you find a lower price for an identical item at a competitor within "
                "seven days of your purchase, bring us the competitor ad or a photo of the "
                "shelf label and we will refund the difference. The item must be in stock "
                "at the competitor and sold under the same brand and size. Price matching "
                "does not apply to clearance items, online marketplace sellers, or "
                "temporary promotional offers.",
            ),
            (
                "Loyalty Program",
                "Our loyalty program is free to join and can be activated at any checkout. "
                "You earn one point for every hundred EGP you spend, and points are added "
                "to your account within one day. One hundred points can be redeemed for a "
                "fifty EGP voucher, which can be used on any purchase except gift cards "
                "and tobacco. Members receive double points during their birthday month "
                "and a ten percent discount on the last Tuesday of every month. Vouchers "
                "are valid for thirty days from the date they are issued.",
            ),
            (
                "Contact and Lost Property",
                "The customer service desk is located near the main entrance and is "
                "staffed during all opening hours. You can reach us by phone on the main "
                "line listed on our website, or by email at customerservice at our store "
                "domain. Lost property is kept at the customer service desk for thirty "
                "days. High value items such as phones and wallets are stored in a secure "
                "cabinet and released only when the owner can describe the item. After "
                "thirty days unclaimed items are donated to charity.",
            ),
        ],
    },
    "safety_manual": {
        "title": "Staff Safety Manual",
        "sections": [
            (
                "Fire Exits",
                "Every aisle has a clearly marked fire exit, and the quickest route is "
                "signposted above the shelves. In case of fire, evacuate immediately "
                "using the nearest exit and never use the lifts. The assembly point is "
                "behind the store in the staff car park. When you hear the alarm, stop "
                "what you are doing, guide any shoppers near you toward the exits, and "
                "move to the assembly point. The shift manager calls the fire service "
                "and checks the staff register once everyone is outside.",
            ),
            (
                "Spills",
                "A wet floor sign must be placed at the scene of any spill within thirty "
                "seconds. If the liquid is not hazardous, absorb it with paper towels and "
                "throw them in the bin. For hazardous liquids, alert the duty manager on "
                "the radio and use the spill kit kept in the cleaning closet. Do not walk "
                "through a spill or ask shoppers to step over it. Every spill, whether "
                "cleaned or reported, must be logged in the incident book so patterns "
                "can be spotted.",
            ),
            (
                "First Aid",
                "The first aid kit is located behind the customer service desk. A "
                "defibrillator is mounted on the wall in the main lobby next to the "
                "vending machines. For serious injuries call the emergency number first "
                "and then the on-site first aider, whose name and shift are posted in "
                "the staff room. Minor cuts should be cleaned with running water and "
                "covered with a plaster from the kit. Gloves are mandatory whenever "
                "blood or body fluid is involved.",
            ),
            (
                "Manual Handling",
                "Lift with your legs, not your back, and keep the load close to your "
                "body. Anything heavier than twenty kilograms must be carried by two "
                "people or moved with a trolley, never alone. Stack products no higher "
                "than the top shelf bar and keep the heaviest items on the bottom. When "
                "restocking a shelf, stand on a step ladder and never reach from the "
                "floor. Report any back or shoulder pain to your manager on the same "
                "shift.",
            ),
            (
                "Power Outage",
                "When the power goes out, the emergency lighting comes on automatically "
                "and the fridges switch to a battery backup for up to two hours. Stand "
                "at the main entrance and guide shoppers toward the exits using a "
                "flashlight. The checkouts go offline immediately, so stop new sales "
                "and write down the last transaction number. Keep the doors closed to "
                "preserve the cold chain and call the building manager to report the "
                "outage.",
            ),
            (
                "Suspicious Packages",
                "If you see an unattended bag or package, do not touch it and do not "
                "open it. Tell the duty manager immediately and keep shoppers at least "
                "fifteen meters away from the item. If the item appears to have wires "
                "or is making a noise, call the emergency number. In all other cases "
                "the manager decides whether to call security. Always use your radio "
                "to report the item so the announcement channel stays clear.",
            ),
        ],
    },
    "faq": {
        "title": "Shopper FAQ",
        "sections": [
            (
                "Finding Products",
                "The quickest way to find a product is to ask the kiosk at the entrance "
                "by speaking to it, for example 'where are the batteries'. The kiosk "
                "will tell you the aisle and draw a map. Every aisle also has a hanging "
                "sign at eye level with the category written in large letters. If you "
                "still cannot find an item, any member of staff can look it up on the "
                "handheld scanner and walk you to the shelf.",
            ),
            (
                "Ordering Ahead",
                "You can order ahead by phone or on the kiosk screen and pick your order "
                "up at the click and collect counter near the exit. Orders placed before "
                "six in the evening are ready within two hours. There is no charge for "
                "pickup. Large orders such as cases of drinks or building materials can "
                "be loaded directly into your car from the loading bay at the back of "
                "the store.",
            ),
            (
                "Delivery",
                "We deliver within a ten kilometer radius of the store. The delivery "
                "fee is forty EGP and it is free for orders over five hundred EGP. "
                "Delivery slots are available every day between ten in the morning and "
                "eight at night, and you can choose a two hour window when you book. "
                "Cold items travel in insulated bags and arrive still chilled. The "
                "delivery driver will call you when they are fifteen minutes away.",
            ),
            (
                "Parking",
                "The store has a free car park with two hundred spaces. Parking is free "
                "for two hours when you spend any amount and keep your receipt. The "
                "ticket machine accepts coins and cards, and you can also pay by "
                "scanning the QR code with your phone. Disabled parking bays are located "
                "directly next to the main entrance, and family spaces are marked near "
                "the trolley bays.",
            ),
            (
                "Bakery",
                "The in-store bakery bakes fresh bread every morning starting at seven "
                "o'clock, and the shelves are restocked through the afternoon. You can "
                "order custom cakes for birthdays and other occasions with forty eight "
                "hours notice from the bakery counter. Ask about gluten free and vegan "
                "options, which are baked in a separate area to avoid contamination. "
                "Leftover bread from the day is sold at half price after eight in the "
                "evening.",
            ),
            (
                "Payment Methods",
                "We accept cash, all major debit and credit cards, contactless payment, "
                "and the national mobile payment apps. Gift cards can be bought at the "
                "customer service desk in any amount between fifty and five thousand EGP. "
                "Self-checkout lanes accept cards and contactless payments only. If you "
                "prefer to pay in cash, use the staffed lanes, where you can also pay "
                "with loyalty vouchers and coupons.",
            ),
        ],
    },
}


def build_pdf(name: str, spec: dict, out_dir: str) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    pdf.set_font("helvetica", "B", 18)
    pdf.cell(0, 12, spec["title"])
    pdf.ln(14)

    for heading, body in spec["sections"]:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, heading)
        pdf.ln(9)
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(0, 5.5, body)
        pdf.ln(3)

    path = os.path.join(out_dir, f"{name}.pdf")
    pdf.output(path)
    return path


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "docs")
    os.makedirs(out_dir, exist_ok=True)
    for name, spec in DOCS.items():
        path = build_pdf(name, spec, out_dir)
        words = sum(len(b.split()) for _, b in spec["sections"])
        print(f"Wrote {path}  ({len(spec['sections'])} sections, ~{words} words)")
