"""
transform/cleaner.py
────────────────────
PURPOSE: Clean and normalise raw post data before analysis.

WHAT YOU ALREADY KNOW: This is 100% identical to what you do in Colab
before feeding text to an ML model:
  - Remove noise (URLs, special characters, extra spaces)
  - Normalise (lowercase, strip whitespace)
  - Drop nulls
  - Convert types (timestamps as datetime objects)

The only difference from Colab: it's a function instead of loose cells.
"""

import pandas as pd
import re
from datetime import datetime


def clean_posts(raw_posts: list[dict]) -> pd.DataFrame:
    """
    Takes raw posts from the fetcher, returns a clean DataFrame.

    Steps:
      1. Load list of dicts → DataFrame  (like pd.read_csv)
      2. Drop nulls
      3. Clean the text column
      4. Parse timestamps to proper datetime
      5. Add a 'text_length' feature column

    ML ANALOGY: This is your feature engineering step.
    clean_text is the feature you'll feed into the sentiment model.
    """

    # ── Step 1: Load into DataFrame ───────────────────────────────────────────
    df = pd.DataFrame(raw_posts)

    # ── Step 2: Drop rows where text is missing ────────────────────────────────
    df.dropna(subset=["text"], inplace=True)
    df = df[df["text"].str.strip() != ""]

    # ── Step 3: Clean the text ─────────────────────────────────────────────────
    def clean_text(text: str) -> str:
        text = str(text)
        text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
        text = re.sub(r"@\w+", "", text)                   # remove @mentions
        text = re.sub(r"#\w+", "", text)                   # remove #hashtags
        text = re.sub(r"[^a-zA-Z0-9\s'.,!?]", "", text)   # keep readable chars
        text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
        return text

    df["clean_text"] = df["text"].apply(clean_text)

    # ── Step 4: Parse timestamps ──────────────────────────────────────────────
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Add useful time columns for charting (like feature engineering in Colab)
    df["date"]       = df["created_at"].dt.date
    df["hour"]       = df["created_at"].dt.hour
    df["day_of_week"]= df["created_at"].dt.day_name()

    # ── Step 5: Add basic feature columns ────────────────────────────────────
    df["word_count"]    = df["clean_text"].apply(lambda x: len(x.split()))
    df["char_count"]    = df["clean_text"].apply(len)

    # Reset index cleanly
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append("..")
    from ingest.mock_fetcher import fetch_posts

    raw = fetch_posts(count=10)
    df  = clean_posts(raw)
    print(df[["topic", "clean_text", "word_count", "date"]].head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")