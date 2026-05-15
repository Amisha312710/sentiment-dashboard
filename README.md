# Social Media Sentiment Dashboard

An end-to-end ETL data pipeline that collects social media posts, runs sentiment analysis, stores results in a database, and visualises them on an interactive real-time dashboard.

> Built as a portfolio project to demonstrate data engineering skills: pipeline architecture, ETL, NLP inference, SQL storage, and dashboard development.

## Architecture
Data Source (Mock/API)
|
v
ingest/mock_fetcher.py      <- Stage 1: Extract raw posts
|
v
transform/cleaner.py        <- Stage 2: Clean & normalise (pandas)
|
v
analyze/sentiment.py        <- Stage 3: Sentiment scoring (TextBlob)
|
v
db/database.py              <- Stage 4: Load into SQLite
|
v
dashboard/app.py            <- Visualise with Streamlit + Plotly
## Features

- End-to-end ETL pipeline across 5 topics: AI, Markets, Climate, Sports, Health
- Sentiment scoring using TextBlob: polarity -1 to +1
- Interactive dark-themed dashboard with 4 chart views
- Filters by topic, sentiment, and date range
- One-click pipeline refresh from the dashboard itself

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | Python, Pandas |
| Sentiment model | TextBlob |
| Database | SQLite |
| Dashboard | Streamlit |
| Charts | Plotly |

## Setup

```bash
pip3 install streamlit pandas plotly textblob schedule
python3 pipeline.py
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Resume Bullets

- Built end-to-end ETL pipeline ingesting 120+ posts/run across 5 topics, processing with Pandas, storing in SQLite
- Automated NLP sentiment scoring using TextBlob with 3-class labelling across 500+ records
- Developed interactive Streamlit dashboard with 4 analysis views, real-time filters, and Plotly visualisations
- Architected modular Python project with clean separation between ingestion, transformation, analysis, and presentation layers

## Author

Built by Amisha Bhatia
