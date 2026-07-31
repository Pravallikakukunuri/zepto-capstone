import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books_catalog.db")

QUERIES = {
    "Q1_five_star_priciest": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating = 5
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    "Q2_distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories;
    """,
    "Q3_price_between_20_and_40": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,
    "Q4_rating_in_4_or_5": """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title;
    """,
    "Q5_top3_per_category_JOIN": """
        SELECT category_name, title, rating
        FROM (
            SELECT c.category_name AS category_name,
                   b.title         AS title,
                   b.rating        AS rating,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.category_name
                       ORDER BY b.rating DESC, b.title
                   ) AS rn
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
        )
        WHERE rn <= 3
        ORDER BY category_name, rating DESC;
    """,
    "Q6_avg_price_per_category_bonus": """
        SELECT c.category_name,
               COUNT(*) AS n_books,
               ROUND(AVG(b.price_inr), 2) AS avg_price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY avg_price_inr DESC;
    """,
}


def run_all_queries(conn):
    results = {}
    for name, sql in QUERIES.items():
        cur = conn.execute(sql)
        rows = cur.fetchall()
        col_names = [d[0] for d in cur.description]
        results[name] = (sql.strip(), col_names, rows)
    return results


def print_results(results):
    for name, (sql, cols, rows) in results.items():
        print("=" * 70)
        print(name)
        print("-" * 70)
        print(sql)
        print(f"columns: {cols}")
        for r in rows:
            print(r)
        print(f"({len(rows)} rows)\n")


def pandas_read_sql_demo(conn):
    df_q1 = pd.read_sql(QUERIES["Q1_five_star_priciest"], conn)
    df_join = pd.read_sql(QUERIES["Q5_top3_per_category_JOIN"], conn)
    return df_q1, df_join


def pandas_merge_demo(conn):
    books_df = pd.read_sql("SELECT * FROM books;", conn)
    categories_df = pd.read_sql("SELECT * FROM categories;", conn)

    merged = pd.merge(books_df, categories_df, on="category_id", how="inner")
    merged = merged.sort_values(["category_name", "rating", "title"], ascending=[True, False, True])
    merged["rn"] = merged.groupby("category_name")["rating"].rank(method="first", ascending=False)
    top3 = (
        merged[merged["rn"] <= 3][["category_name", "title", "rating"]]
        .sort_values(["category_name", "rating"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return top3


def main():
    conn = sqlite3.connect(DB_PATH)

    results = run_all_queries(conn)
    print_results(results)

    df_q1, df_join_sql = pandas_read_sql_demo(conn)
    df_join_pandas = pandas_merge_demo(conn)

    print("=" * 70)
    print("pd.read_sql result for the JOIN query (Q5):")
    print(df_join_sql.reset_index(drop=True))

    print("\npd.merge-only (in-memory, no SQL) reproduction of the same JOIN:")
    print(df_join_pandas)

    a = (
        df_join_sql[["category_name", "title", "rating"]]
        .sort_values(["category_name", "rating", "title"])
        .reset_index(drop=True)
    )
    b = df_join_pandas.sort_values(["category_name", "rating", "title"]).reset_index(drop=True)
    print(f"\nSQL JOIN result and pandas merge result match: {a.equals(b)}")

    conn.close()


if __name__ == "__main__":
    main()