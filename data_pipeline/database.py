import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def create_schema(conn):
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def get_or_create_category(conn, category_name):
    cur = conn.execute(
        "SELECT category_id FROM categories WHERE category_name = ?",
        (category_name,),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO categories (category_name) VALUES (?)", (category_name,)
    )
    conn.commit()
    return cur.lastrowid


def load_books(conn, cleaned_books):
    for book in cleaned_books:
        category_id = get_or_create_category(conn, book["category"])
        conn.execute(
            """
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                book["title"],
                book["price_gbp"],
                book["price_inr"],
                book["rating"],
                int(book["in_stock"]),
                category_id,
            ),
        )
    conn.commit()


def build_database(cleaned_books, db_path="books_catalog.db"):
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    load_books(conn, cleaned_books)
    return conn