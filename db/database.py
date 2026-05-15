"""
db/database.py
──────────────
PURPOSE: Save processed posts to a SQLite database and read them back.

WHAT YOU ALREADY KNOW: In Colab you'd do df.to_csv("results.csv").
SQLite does the same thing but the file is a proper database, meaning:
  - You can query it with SQL: SELECT * WHERE sentiment_label = 'Negative'
  - It handles duplicate rows automatically (we check by post id)
  - It's much faster to read specific slices than reading a whole CSV

WHY NOT JUST CSV: Dashboards need to filter/aggregate data on the fly.
SQL is 10x faster and cleaner than reading a CSV and filtering in pandas.
Also — every pipeline run appends new rows rather than overwriting.

The database is one file: data/sentiment.db
"""

import sqlite3
import pandas as pd
import os

# Path to the database file (created automatically if it doesn't exist)
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sentiment.db")


def get_connection() -> sqlite3.Connection:
    """Open and return a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_table():
    """
    Create the posts table if it doesn't already exist.
    This is like defining your DataFrame schema upfront.

    Columns mirror what cleaner.py + sentiment.py produce.
    """
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id               TEXT PRIMARY KEY,
            text             TEXT,
            clean_text       TEXT,
            author           TEXT,
            source           TEXT,
            topic            TEXT,
            created_at       TEXT,
            date             TEXT,
            hour             INTEGER,
            day_of_week      TEXT,
            upvotes          INTEGER,
            comments         INTEGER,
            word_count       INTEGER,
            sentiment_score  REAL,
            subjectivity     REAL,
            sentiment_label  TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_posts(df: pd.DataFrame) -> int:
    """
    Insert new posts into the database.
    Skips rows whose 'id' already exists (no duplicates on re-runs).

    Returns the number of NEW rows actually inserted.

    ML ANALOGY: Like saving your model predictions to a results file,
    but smarter — won't overwrite if you run the pipeline twice.
    """
    create_table()

    cols = [
        "id", "text", "clean_text", "author", "source", "topic",
        "created_at", "date", "hour", "day_of_week", "upvotes",
        "comments", "word_count", "sentiment_score", "subjectivity",
        "sentiment_label"
    ]

    # Only keep columns that exist in our DataFrame
    save_df = df[[c for c in cols if c in df.columns]].copy()
    save_df["created_at"] = save_df["created_at"].astype(str)
    save_df["date"]       = save_df["date"].astype(str)

    conn = get_connection()
    before = pd.read_sql("SELECT COUNT(*) as n FROM posts", conn).iloc[0]["n"]

    # INSERT OR IGNORE = skip if id already exists
    save_df.to_sql("posts_temp", conn, if_exists="replace", index=False)
    conn.execute("""
        INSERT OR IGNORE INTO posts
        SELECT * FROM posts_temp
    """)
    conn.execute("DROP TABLE posts_temp")
    conn.commit()

    after = pd.read_sql("SELECT COUNT(*) as n FROM posts", conn).iloc[0]["n"]
    conn.close()

    return int(after - before)


def load_posts(
    topic: str = None,
    label: str = None,
    days: int  = 7,
) -> pd.DataFrame:
    """
    Read posts from the database back into a DataFrame.

    Parameters:
        topic : Filter by topic name (or None for all)
        label : Filter by 'Positive', 'Negative', 'Neutral' (or None)
        days  : How many past days to load

    Returns: DataFrame ready for the dashboard to use.

    ML ANALOGY: Like pd.read_csv() but with built-in filtering.
    """
    conn   = get_connection()
    query  = f"""
        SELECT *
        FROM   posts
        WHERE  date(created_at) >= date('now', '-{days} days')
    """
    if topic:
        query += f" AND topic = '{topic}'"
    if label:
        query += f" AND sentiment_label = '{label}'"

    df = pd.read_sql(query, conn)
    conn.close()

    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["date"]       = pd.to_datetime(df["date"]).dt.date

    return df


def get_topic_summary() -> pd.DataFrame:
    """Returns per-topic aggregate stats — used by dashboard charts."""
    conn = get_connection()
    df   = pd.read_sql("""
        SELECT
            topic,
            COUNT(*)                     AS total_posts,
            ROUND(AVG(sentiment_score), 3) AS avg_score,
            SUM(CASE WHEN sentiment_label='Positive' THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN sentiment_label='Neutral'  THEN 1 ELSE 0 END) AS neutral,
            SUM(CASE WHEN sentiment_label='Negative' THEN 1 ELSE 0 END) AS negative
        FROM posts
        GROUP BY topic
        ORDER BY avg_score DESC
    """, conn)
    conn.close()
    return df


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from ingest.mock_fetcher import fetch_posts
    from transform.cleaner   import clean_posts
    from analyze.sentiment   import score_sentiment

    df      = score_sentiment(clean_posts(fetch_posts(count=30)))
    inserted = save_posts(df)
    print(f"Inserted {inserted} new rows")

    loaded  = load_posts()
    print(f"Total in DB: {len(loaded)} rows")
    print(get_topic_summary().to_string(index=False))