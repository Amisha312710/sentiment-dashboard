"""
db/database.py
──────────────
PURPOSE: Save processed posts to a SQLite database and read them back.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sentiment.db")


def get_connection() -> sqlite3.Connection:
    """Open and return a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_table():
    """
    Create the posts table if it doesn't already exist.
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
            sentiment_label  TEXT,
            vader_compound   REAL
        )
    """)
    try:
        conn.execute("ALTER TABLE posts ADD COLUMN vader_compound REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def save_posts(df: pd.DataFrame) -> int:
    create_table()

    cols = [
        "id", "text", "clean_text", "author", "source", "topic",
        "created_at", "date", "hour", "day_of_week", "upvotes",
        "comments", "word_count", "sentiment_score", "subjectivity",
        "sentiment_label", "vader_compound"
    ]

    save_df = df[[c for c in cols if c in df.columns]].copy()
    save_df["created_at"] = save_df["created_at"].astype(str)
    save_df["date"]       = save_df["date"].astype(str)

    conn = get_connection()
    before = pd.read_sql("SELECT COUNT(*) as n FROM posts", conn).iloc[0]["n"]

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
    create_table()
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

    try:
        df = pd.read_sql(query, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if not df.empty:
        try:
            df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")
        except Exception:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        except Exception:
            pass

    return df


def get_topic_summary() -> pd.DataFrame:
    create_table()
    conn = get_connection()
    try:
        df = pd.read_sql("""
            SELECT
                topic,
                COUNT(*)                     AS total_posts,
                ROUND(AVG(COALESCE(vader_compound, sentiment_score)), 3) AS avg_score,
                SUM(CASE WHEN sentiment_label='Positive' THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN sentiment_label='Neutral'  THEN 1 ELSE 0 END) AS neutral,
                SUM(CASE WHEN sentiment_label='Negative' THEN 1 ELSE 0 END) AS negative
            FROM posts
            GROUP BY topic
            ORDER BY avg_score DESC
        """, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df