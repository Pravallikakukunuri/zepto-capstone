# /data_pipeline - Data Pipeline Module

Scrapes books.toscrape.com, cleans the data, converts price to INR using a fixed rate of 1 GBP = 105.50 INR, loads it into a normalized SQLite database, and queries it with SQL and pandas.

## Files
- scraper.py - scrapes books.toscrape.com
- clean.py - cleans and types the scraped data, converts currency
- database.py - creates SQLite schema and loads data
- run_pipeline.py - runs scraper + clean + database in order (creates books_catalog.db)
- queries.py - runs the required SQL queries and pandas comparison

## How to run
cd data_pipeline
pip install -r requirements.txt
python run_pipeline.py
python queries.py

## Fixed conversion rate
1 GBP = 105.50 INR (fixed project-defined constant, not a live rate, stated here as required).

## Cleaning decisions
- price_gbp: numeric field. Missing/unparseable values are median-imputed rather than dropping the row, since a median is a reasonable stand-in for a single missing numeric value.
- rating and in_stock: categorical/boolean fields. Rows with unparseable values are dropped instead of imputed, since there is no statistically sound "average" category to guess.

## Schema
categories(category_id PK, category_name UNIQUE)
books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK -> categories)