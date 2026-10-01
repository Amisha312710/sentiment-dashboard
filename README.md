# Social Media Sentiment Analytics Engine 📊

> An end-to-end automated ETL pipeline & real-time multi-model NLP sentiment intelligence dashboard across 5 industry domains.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLP-VADER%20%2B%20TextBlob-green.svg)](https://github.com/cjhutto/vaderSentiment)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)

---

## 🚀 Key Highlights & Architecture

- **Automated ETL Pipeline** — Ingests live API posts (HackerNews public API) + multi-domain streams with automatic deduplication into SQLite.
- **Dual NLP Sentiment Engines** — Combines **VADER** (specifically tuned for social media, punctuation, and capitalization) with **TextBlob** polarity & subjectivity metrics.
- **⚡ Real-Time Live Playground** — Interactive UI tab allowing recruiters and visitors to enter custom text and receive instant sentiment inference and gauge metrics.
- **5 Market Domains** — AI & Technology, Stock Market, Climate & Environment, Sports, Health & Wellness.
- **Full Plotly Dark Theme UI** — Velocity curves, hourly distribution, sector radar charts, and CSV dataset export.
- **Auto-Seeding On Deployment** — Automatic database initialization on fresh cloud deploys.
- **CI & Unit Tested** — Automated test suite covering ingestion, data cleaning, and NLP inference.

---

## 🛠 Tech Stack

- **Data Processing & Storage**: Python, Pandas, SQLite
- **NLP Sentiment Engines**: VADER Sentiment Intensity Analyzer, TextBlob
- **Dashboard & Visualizations**: Streamlit, Plotly Express & Graph Objects
- **APIs & Testing**: Requests, HackerNews Firebase API, Python Unittest

---

## 📂 Project Structure

```text
sentiment-dashboard/
├── analyze/
│   └── sentiment.py         # Multi-model NLP scoring (VADER + TextBlob) & single inference
├── dashboard/
│   └── app.py               # Streamlit interactive dark-theme dashboard & live playground
├── db/
│   └── database.py          # SQLite schema, upsert logic & aggregate queries
├── ingest/
│   └── mock_fetcher.py      # Hybrid fetcher: HackerNews Live API + fallback generator
├── tests/
│   └── test_pipeline.py     # Automated unit test suite
├── transform/
│   └── cleaner.py           # Text preprocessing, regex cleaning, and feature engineering
├── pipeline.py              # Central ETL orchestration script
├── requirements.txt         # Production dependencies for cloud deployment
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/sentiment-dashboard.git
cd sentiment-dashboard
pip install -r requirements.txt
```

### 2. Run the ETL Pipeline
```bash
# Execute single ETL cycle
python pipeline.py

# Or run on continuous 60-min scheduler
python pipeline.py --loop
```

### 3. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
Open **`http://localhost:8501`** in your browser.

### 4. Run Unit Tests
```bash
python -m unittest discover tests
```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push this repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select this repository and set:
   - **Main file path**: `dashboard/app.py`
4. Click **Deploy!** The application will auto-install `requirements.txt` and automatically seed sample data on first launch.

---

## 📄 License
MIT License. Free to use and build upon.
