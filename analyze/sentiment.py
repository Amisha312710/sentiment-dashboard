"""
analyze/sentiment.py
────────────────────
PURPOSE: Run multi-model sentiment analysis (TextBlob + VADER) on text.
Provides dual NLP models:
  - TextBlob: lexicon & pattern-based polarity (-1.0 to +1.0) & subjectivity (0 to 1)
  - VADER: rule-based model specifically tuned for social media (emojis, caps, slang)
"""

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds multi-model sentiment columns to the DataFrame.
    """
    def analyse(text: str) -> tuple:
        s_text = str(text)
        # TextBlob
        blob = TextBlob(s_text)
        tb_polarity = blob.sentiment.polarity
        tb_subjectivity = blob.sentiment.subjectivity

        # VADER
        vader_scores = _vader.polarity_scores(s_text)
        vader_compound = vader_scores["compound"]
        vader_pos = vader_scores["pos"]
        vader_neg = vader_scores["neg"]
        vader_neu = vader_scores["neu"]

        return tb_polarity, tb_subjectivity, vader_compound, vader_pos, vader_neg, vader_neu

    results = df["clean_text"].apply(analyse)
    df["sentiment_score"]  = results.apply(lambda x: round(x[0], 4))
    df["subjectivity"]     = results.apply(lambda x: round(x[1], 4))
    df["vader_compound"]   = results.apply(lambda x: round(x[2], 4))
    df["vader_pos"]        = results.apply(lambda x: round(x[3], 4))
    df["vader_neg"]        = results.apply(lambda x: round(x[4], 4))
    df["vader_neu"]        = results.apply(lambda x: round(x[5], 4))

    # Hybrid sentiment label (using VADER compound score for social media accuracy)
    def label(score: float) -> str:
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment_label"] = df["vader_compound"].apply(label)
    return df


def analyze_single_text(text: str) -> dict:
    """
    Real-time inference function for custom user input in the dashboard.
    Returns comprehensive multi-model sentiment metrics.
    """
    if not text or not text.strip():
        return {
            "text": "",
            "label": "Neutral",
            "vader_compound": 0.0,
            "vader_pos": 0.0,
            "vader_neg": 0.0,
            "vader_neu": 1.0,
            "tb_polarity": 0.0,
            "tb_subjectivity": 0.0,
        }

    s_text = text.strip()
    blob = TextBlob(s_text)
    tb_polarity = round(blob.sentiment.polarity, 4)
    tb_subjectivity = round(blob.sentiment.subjectivity, 4)

    vader_scores = _vader.polarity_scores(s_text)
    compound = round(vader_scores["compound"], 4)

    if compound >= 0.05:
        lbl = "Positive"
    elif compound <= -0.05:
        lbl = "Negative"
    else:
        lbl = "Neutral"

    return {
        "text": s_text,
        "label": lbl,
        "vader_compound": compound,
        "vader_pos": round(vader_scores["pos"], 3),
        "vader_neg": round(vader_scores["neg"], 3),
        "vader_neu": round(vader_scores["neu"], 3),
        "tb_polarity": tb_polarity,
        "tb_subjectivity": tb_subjectivity,
    }


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute aggregate stats for the dashboard KPI cards.
    """
    total = len(df)
    if total == 0:
        return {
            "total_posts": 0, "avg_score": 0.0, "positive_count": 0, "neutral_count": 0,
            "negative_count": 0, "positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0,
            "avg_subjectivity": 0.0, "most_positive_topic": "N/A", "most_negative_topic": "N/A"
        }

    counts = df["sentiment_label"].value_counts()
    score_col = "vader_compound" if "vader_compound" in df.columns else "sentiment_score"

    return {
        "total_posts":      total,
        "avg_score":        round(df[score_col].mean(), 3),
        "positive_count":   int(counts.get("Positive", 0)),
        "neutral_count":    int(counts.get("Neutral",  0)),
        "negative_count":   int(counts.get("Negative", 0)),
        "positive_pct":     round(counts.get("Positive", 0) / total * 100, 1),
        "neutral_pct":      round(counts.get("Neutral",  0) / total * 100, 1),
        "negative_pct":     round(counts.get("Negative", 0) / total * 100, 1),
        "avg_subjectivity": round(df["subjectivity"].mean(), 3),
        "most_positive_topic": (
            df.groupby("topic")[score_col].mean().idxmax()
        ),
        "most_negative_topic": (
            df.groupby("topic")[score_col].mean().idxmin()
        ),
    }