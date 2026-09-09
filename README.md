# BonNuit

A semantic search engine for Paris hotel reviews, implemented in Python 3.

Given a query like *"An affordable hotel with a view of the Eiffel Tower"*, the code embeds the text and compares it against hotel review embeddings using similarity search, then returns the top-k most relevant matches.

## Setup

Requires **Python 3.11+**. Dependencies are listed in `requirements.txt` and
installed with pip. If the system Python is older (common on shared servers),
get 3.11 via conda without needing root:

```bash
conda create -n bonnuit python=3.11
conda activate bonnuit
pip install -r requirements.txt
```

Or, on a machine that already has a 3.11+ interpreter:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The package is not installed — run it with `src/` on the import path
(`PYTHONPATH=src`). `pytest` handles this itself via `conftest.py`.

## Usage

```bash
# 1. ingest: embed reviews and upsert them into an embedded Qdrant at ./data/qdrant
PYTHONPATH=src python -m bonnuit.ingest --locality Paris

# 2. query it
PYTHONPATH=src python -m bonnuit.demo "a quiet hotel near the Louvre" -k 5

# tests (similarity + chunking + vector store; need only numpy + qdrant-client)
pytest
```

To use a running Qdrant server instead of the embedded one, pass
`--qdrant-url http://localhost:6333` to both commands. If no store exists at
`data/qdrant`, `demo.py` builds an in-memory one on the fly.

## API

```bash
PYTHONPATH=src uvicorn backend.app:app --reload
```

- `POST /api/search` — ranked review hits for a query
- `POST /api/ask` — a generated, grounded answer with citations
- `GET  /api/health`
- interactive docs at `/docs`, OpenAPI schema at `/openapi.json`

The handlers currently return stub data; the request/response models in
`backend/schemas.py` are the contract the frontend builds against. Allowed CORS
origins default to `localhost:3000` / `localhost:5173` and can be overridden with
`BONNUIT_CORS_ORIGINS`.

## Architecture

```
React frontend  --HTTPS/JSON-->  FastAPI (backend/app.py)
                                   |-- bonnuit: embed + search
                                   |-- Qdrant (embedded or server)
                                   `-- Claude (answer generation, /api/ask)
```

All search and RAG logic is Python and server-side. The frontend only calls the
JSON API — it never touches embeddings, Qdrant, or the model. The contract
between them is `backend/schemas.py`'s Pydantic models, published as OpenAPI at
`/openapi.json`; the frontend generates its TypeScript types from that.

## Tech stack

**Backend**

- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) — JSON API
- [Sentence Transformers](https://www.sbert.net/) — text embeddings (`all-MiniLM-L6-v2`)
- [Qdrant](https://qdrant.tech/) — vector database (embedded or server), cosine distance
- [NumPy](https://numpy.org/) — cosine similarity reference implementation
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — hotel review dataset
- [pandas](https://pandas.pydata.org/) — filtering and data cleaning

**Frontend** (planned)

- [React](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- [Tailwind CSS](https://tailwindcss.com/) — styling
- [TanStack Query](https://tanstack.com/query) — data fetching / caching against the API
- types generated from the backend's OpenAPI schema

## Embedding Model

The project uses:

```python

SentenceTransformer("all-MiniLM-L6-v2")

```

This model converts text into dense vector embeddings suitable for semantic search.

## Dataset

Reviews come from the [traversaal-ai-hackathon/hotel_datasets](https://huggingface.co/datasets/traversaal-ai-hackathon/hotel_datasets) dataset on Hugging Face.

This repo will still grow

## Todos
- Run the pipeline end to end on the Paris data; add an ingest test
- Wire `/api/search` to the real retriever (drop the stub)
- Drop the `--locality` / `locality` plumbing (single-city, Paris)
- Create a React search page against the API
- Add the RAG generation step (`generator.py`) behind `/api/ask`
- Build the front-end MVP

