"""
pipeline.py
───────────
PURPOSE: Run the full ETL pipeline in one go.
  Fetch → Clean → Score → Save

WHAT YOU ALREADY KNOW: This is like clicking "Run All" in Colab.
The difference is it can also run automatically on a timer using APScheduler.
You can also just call run_pipeline() manually to load fresh data.

HOW TO RUN:
  python3 pipeline.py          → runs once and exits
  python3 pipeline.py --loop   → runs every hour forever (scheduler mode)
"""

import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from ingest.mock_fetcher import fetch_posts
from transform.cleaner   import clean_posts
from analyze.sentiment   import score_sentiment
from db.database         import save_posts

# ── Logging setup ─────────────────────────────────────────────────────────────
# In Colab you'd just use print(). In a real project, logging is better
# because it includes timestamps and can write to a file.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_pipeline(post_count: int = 120) -> dict:
    """
    Execute every stage of the pipeline.

    Returns a summary dict so the dashboard can show pipeline status.
    """
    log.info("═" * 50)
    log.info("Pipeline starting")
    start = datetime.now()

    try:
        # ── Stage 1: Ingest ───────────────────────────────────────────────────
        log.info("Stage 1/4 → Fetching posts...")
        raw_posts = fetch_posts(count=post_count)
        log.info(f"  Fetched {len(raw_posts)} raw posts")

        # ── Stage 2: Transform ────────────────────────────────────────────────
        log.info("Stage 2/4 → Cleaning & transforming...")
        df = clean_posts(raw_posts)
        log.info(f"  Clean DataFrame shape: {df.shape}")

        # ── Stage 3: Analyze ──────────────────────────────────────────────────
        log.info("Stage 3/4 → Running sentiment analysis...")
        df = score_sentiment(df)
        label_counts = df["sentiment_label"].value_counts().to_dict()
        log.info(f"  Labels: {label_counts}")

        # ── Stage 4: Load ─────────────────────────────────────────────────────
        log.info("Stage 4/4 → Saving to database...")
        inserted = save_posts(df)
        log.info(f"  Inserted {inserted} new rows")

        elapsed = (datetime.now() - start).total_seconds()
        log.info(f"Pipeline complete in {elapsed:.1f}s")

        return {
            "status":    "success",
            "fetched":   len(raw_posts),
            "inserted":  inserted,
            "elapsed_s": round(elapsed, 2),
            "run_at":    start.isoformat(),
            **label_counts,
        }

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    if "--loop" in sys.argv:
        # Scheduler mode: run every 60 minutes
        try:
            import schedule, time
            log.info("Scheduler mode: running every 60 minutes")
            schedule.every(60).minutes.do(run_pipeline)
            run_pipeline()  # run immediately on start
            while True:
                schedule.run_pending()
                time.sleep(30)
        except ImportError:
            log.warning("'schedule' not installed. Running once.")
            run_pipeline()
    else:
        result = run_pipeline()
        print("\nPipeline result:", result)