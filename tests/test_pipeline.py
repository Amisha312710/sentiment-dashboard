"""
tests/test_pipeline.py
──────────────────────
Unit tests verifying ETL pipeline stages, data transformations, and sentiment inference.
"""

import unittest
import pandas as pd
from datetime import datetime

from ingest.mock_fetcher import fetch_posts
from transform.cleaner import clean_posts
from analyze.sentiment import score_sentiment, analyze_single_text
from db.database import save_posts, load_posts


class TestSentimentPipeline(unittest.TestCase):

    def test_fetch_posts_schema(self):
        posts = fetch_posts(count=10, use_live_api=False)
        self.assertEqual(len(posts), 10)
        required_keys = {"id", "text", "author", "source", "topic", "created_at"}
        for p in posts:
            self.assertTrue(required_keys.issubset(p.keys()))

    def test_clean_posts(self):
        raw = [{
            "id": "test_1",
            "text": "Check out this link https://example.com @user #awesome! Great product.",
            "author": "tester",
            "source": "Reddit",
            "topic": "AI & Technology",
            "created_at": datetime.now().isoformat(),
            "upvotes": 10,
            "comments": 2
        }]
        df = clean_posts(raw)
        self.assertFalse(df.empty)
        self.assertNotIn("https://example.com", df.iloc[0]["clean_text"])
        self.assertNotIn("@user", df.iloc[0]["clean_text"])
        self.assertIn("word_count", df.columns)

    def test_sentiment_scoring(self):
        raw = [
            {"id": "t1", "text": "This product is fantastic and I love it!", "author": "a", "source": "s", "topic": "Tech", "created_at": datetime.now().isoformat(), "upvotes": 1, "comments": 0},
            {"id": "t2", "text": "Terrible failure, absolute disaster and waste of money.", "author": "b", "source": "s", "topic": "Tech", "created_at": datetime.now().isoformat(), "upvotes": 1, "comments": 0}
        ]
        df = clean_posts(raw)
        scored = score_sentiment(df)
        self.assertIn("vader_compound", scored.columns)
        self.assertIn("sentiment_label", scored.columns)
        
        pos_row = scored[scored["id"] == "t1"].iloc[0]
        neg_row = scored[scored["id"] == "t2"].iloc[0]
        self.assertEqual(pos_row["sentiment_label"], "Positive")
        self.assertEqual(neg_row["sentiment_label"], "Negative")

    def test_single_text_inference(self):
        res_pos = analyze_single_text("Excellent work! Truly amazing and inspiring.")
        self.assertEqual(res_pos["label"], "Positive")
        self.assertGreater(res_pos["vader_compound"], 0.05)

        res_neg = analyze_single_text("Horrible experience, completely broken and useless.")
        self.assertEqual(res_neg["label"], "Negative")
        self.assertLess(res_neg["vader_compound"], -0.05)


if __name__ == "__main__":
    unittest.main()
