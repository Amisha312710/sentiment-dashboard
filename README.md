# Social Media Sentiment Dashboard 📊

> A real-time sentiment analysis pipeline with an interactive Streamlit dashboard — built across 5 domains of social media data.

## What is this?

A full ETL pipeline that fetches social media posts, cleans them, runs sentiment analysis using NLP, stores results in a database, and visualizes everything in a dark-themed interactive dashboard.

---

## Features

- **ETL Pipeline** — Fetch → Clean → Analyze → Store, all in one command
- **Sentiment Analysis** — TextBlob-powered NLP scoring (Positive / Neutral / Negative)
- **5 Topic Domains** — AI & Technology, Stock Market, Climate, Sports, Health
- **Interactive Dashboard** — Streamlit + Plotly with dark UI
- **Scheduler Mode** — Auto-runs pipeline every 60 minutes
- **KPI Cards** — Average sentiment score, breakdown by label, subjectivity
- **Filters** — By topic, sentiment label, and date range

---

## Tech Stack

- **Python** — core language
- **TextBlob** — pre-trained NLP sentiment model
- **Streamlit** — dashboard UI
- **Plotly** — interactive charts
- **SQLite** — local database
- **Pandas** — data transformation

---

## Project Structure
sentiment-dashboard/

├── pipeline.py          # Main ETL orchestrator

├── ingest/

│   └── mock_fetcher.py  # Fetches social media posts

├── transform/

│   └── cleaner.py       # Cleans and normalizes text

├── analyze/

│   └── sentiment.py     # TextBlob sentiment scoring

├── db/

│   └── database.py      # SQLite save/load

├── dashboard/

│   └── app.py           # Streamlit dashboard

└── README.md
---

## Getting Started

### Install dependencies

```bash
pip install streamlit plotly textblob pandas
python -m textblob.download_corpora
```

### Run the pipeline once

```bash
python3 pipeline.py
```

### Run pipeline on auto-scheduler (every 60 min)

```bash
python3 pipeline.py --loop
```

### Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## How it works
Social Media Posts

↓

Fetch (ingest)

↓

Clean (transform)

↓

Sentiment Score (analyze)

↓

Save to SQLite (db)

↓

Visualize (dashboard)
---

## Sentiment Scoring

Uses TextBlob's pre-trained model:

| Score | Label |
|---|---|
| > 0.05 | ✅ Positive |
| -0.05 to 0.05 | ➖ Neutral |
| < -0.05 | ❌ Negative |

---

## License

MIT — free to use and build on.

---

*Built as part of an NLP/ML portfolio — demonstrates real-world data pipeline architecture.*
