"""
analyze/sentiment.py
────────────────────
PURPOSE: Run sentiment analysis on every cleaned post.

WHAT YOU ALREADY KNOW: This is model.predict() from your ML workflow.
TextBlob is a pre-trained NLP model. You pass it text, it returns:
  - polarity:    -1.0 (very negative) → 0.0 (neutral) → +1.0 (very positive)
  - subjectivity: 0.0 (objective fact) → 1.0 (personal opinion)

You don't train anything here — it's inference only, same as using
a pre-trained sklearn model or a HuggingFace pipeline.
"""

import pandas as pd
from textblob import TextBlob


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sentiment columns to the DataFrame.

    Input:  cleaned DataFrame (from cleaner.py)
    Output: same DataFrame + these new columns:
              sentiment_score  → float, -1 to +1
              subjectivity     → float, 0 to 1
              sentiment_label  → "Positive" / "Neutral" / "Negative"
              sentiment_emoji  → for display in the dashboard

    ML ANALOGY:
      df["sentiment_score"] = model.predict(df["clean_text"])
      That's literally what this does.
    """

    def analyse(text: str) -> tuple:
        blob       = TextBlob(str(text))
        polarity   = blob.sentiment.polarity      # -1 to +1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        return polarity, subjectivity

    # Apply to every row (same as df["col"].apply(func) in Colab)
    results = df["clean_text"].apply(analyse)
    df["sentiment_score"]  = results.apply(lambda x: round(x[0], 4))
    df["subjectivity"]     = results.apply(lambda x: round(x[1], 4))

    # Convert numeric score → human-readable label
    # (like mapping class index → class name after model.predict)
    def label(score: float) -> str:
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment_label"] = df["sentiment_score"].apply(label)

    # Emoji version — purely for the dashboard UI
    emoji_map = {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"}
    df["sentiment_label"] = df["sentiment_label"]  # already set above

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute aggregate stats for the dashboard KPI cards.
    Returns a dict of headline numbers — avg score, breakdown counts, etc.

    ML ANALOGY: Like a classification report — but for the dashboard.
    """
    total = len(df)
    counts = df["sentiment_label"].value_counts()

    return {
        "total_posts":      total,
        "avg_score":        round(df["sentiment_score"].mean(), 3),
        "positive_count":   int(counts.get("Positive", 0)),
        "neutral_count":    int(counts.get("Neutral",  0)),
        "negative_count":   int(counts.get("Negative", 0)),
        "positive_pct":     round(counts.get("Positive", 0) / total * 100, 1),
        "neutral_pct":      round(counts.get("Neutral",  0) / total * 100, 1),
        "negative_pct":     round(counts.get("Negative", 0) / total * 100, 1),
        "avg_subjectivity": round(df["subjectivity"].mean(), 3),
        "most_positive_topic": (
            df.groupby("topic")["sentiment_score"].mean().idxmax()
        ),
        "most_negative_topic": (
            df.groupby("topic")["sentiment_score"].mean().idxmin()
        ),
    }


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from ingest.mock_fetcher import fetch_posts
    from transform.cleaner   import clean_posts

    df = clean_posts(fetch_posts(count=20))
    df = score_sentiment(df)

    print(df[["clean_text", "sentiment_score", "sentiment_label"]].head(5).to_string())
    print("\n── Summary stats ──")
    stats = get_summary_stats(df)
    for k, v in stats.items():
        print(f"  {k}: {v}")