# Zepto Capstone Project — Certificate Program in AI/ML

One connected platform made of three modules: a data-engineering pipeline,
an analytics/modeling pipeline, and a GenAI support assistant.

## Modules

| Module | Folder | Marks | Description |
|---|---|---|---|
| 1. Data Pipeline | `/data_pipeline` | 25 | Scrapes books.toscrape.com, cleans and converts currency, loads into a normalized SQLite database, queries with SQL and pandas. |
| 2. Analytics | `/analytics` | 50 | Profiles and cleans the Titanic dataset, tells a visual EDA story, then trains/evaluates/tunes 3 classifiers plus a regression side-task. |
| 3. Support Assistant | `/support_assistant` | 25 | A RAG-based GenAI support assistant answering Zepto policy questions, using LangGraph, ChromaDB, and FastAPI. Runs fully offline in mock mode. |

## How to run each module

Each module folder contains its own `README.md` with detailed install/run
steps and design decisions. Summary:

### /data_pipeline
cd data_pipeline
pip install -r requirements.txt
python run_pipeline.py
python queries.py

### /analytics

cd analytics
pip install -r requirements.txt
python 01_eda.py
python 02_modeling.py

### /support_assistant
cd support_assistant
pip install -r requirements.txt
python ingest.py
uvicorn main:app --reload


## Setup notes

- Each module has its own `requirements.txt`; install per-module as shown above.
- No paid services are required anywhere in this project. All external
  services used (books.toscrape.com, Seaborn's dataset loader, sentence-transformers,
  ChromaDB) are free and require no signup or API key.
- `/support_assistant` runs in a fully offline, deterministic mock mode by
  default (MOCK_LLM unset) -- no LLM API key is required to earn full marks
  on that module.

## Design decisions

See each module's own README for full details:
- `/data_pipeline/README.md` -- cleaning/imputation decisions, fixed currency rate, schema
- `/analytics/README.md` -- missing-value handling, EDA findings, model comparison, final recommendation
- `/support_assistant/README.md` -- RAG architecture description, example call transcripts, MOCK_LLM toggle behavior