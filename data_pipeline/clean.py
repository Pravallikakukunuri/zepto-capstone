import re
import statistics

FIXED_GBP_TO_INR = 105.50

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

_PRICE_NUMBER_RE = re.compile(r"[\d]+\.?[\d]*")


def _parse_price(price_text):
    try:
        match = _PRICE_NUMBER_RE.search(price_text.replace(",", ""))
        if not match:
            return None
        return float(match.group())
    except (ValueError, AttributeError, TypeError):
        return None


def _parse_rating(star_word):
    return RATING_WORDS.get(star_word)


def _parse_availability(availability_text):
    if availability_text is None:
        return None
    text = availability_text.strip().lower()
    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False
    return None


def clean_books(raw_books):
    parsed_rows = []
    for row in raw_books:
        parsed_rows.append({
            "title": row.get("title"),
            "category": row.get("category"),
            "price_gbp": _parse_price(row.get("price")),
            "rating": _parse_rating(row.get("star_rating")),
            "in_stock": _parse_availability(row.get("availability")),
        })

    valid_prices = [r["price_gbp"] for r in parsed_rows if r["price_gbp"] is not None]
    median_price = statistics.median(valid_prices) if valid_prices else 0.0

    n_price_imputed = 0
    n_dropped = 0
    cleaned_books = []

    for row in parsed_rows:
        if row["price_gbp"] is None:
            row["price_gbp"] = median_price
            n_price_imputed += 1

        if row["rating"] is None or row["in_stock"] is None or not row["title"]:
            n_dropped += 1
            continue

        row["price_inr"] = round(row["price_gbp"] * FIXED_GBP_TO_INR, 2)
        cleaned_books.append(row)

    report = {
        "input_rows": len(raw_books),
        "output_rows": len(cleaned_books),
        "price_gbp_median_imputed_count": n_price_imputed,
        "rows_dropped_categorical": n_dropped,
        "median_price_used": median_price,
        "fixed_rate_gbp_to_inr": FIXED_GBP_TO_INR,
    }
    return cleaned_books, report