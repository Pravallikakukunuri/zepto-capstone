import os
import sys

from scraper import scrape_books
from clean import clean_books
from database import build_database

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books_catalog.db")


def main():
    print("Step 1/3: Scraping books.toscrape.com ...")
    raw_books = scrape_books(min_total=60, min_categories=3)
    print(f"  -> scraped {len(raw_books)} raw rows")

    print("Step 2/3: Cleaning + converting currency ...")
    cleaned_books, report = clean_books(raw_books)
    print(f"  -> {report}")

    if len(cleaned_books) < 60:
        print("WARNING: fewer than 60 clean rows remain after dropping bad rows.", file=sys.stderr)

    print("Step 3/3: Loading into SQLite ...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = build_database(cleaned_books, db_path=DB_PATH)
    conn.close()
    print(f"Done. Database written to {DB_PATH}")


if __name__ == "__main__":
    main()