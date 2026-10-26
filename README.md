# Hotel Search Engine

A semantic search engine for finding Hotels in Paris, implemented as a RAG.

Given a query like *"An affordable hotel with a view of the Eiffel Tower"*, the code embeds the text and compares it against hotel review embeddings using similarity search, then returns the top-k most relevant matches.

## Setup

Requires **Python 3.11+**. Dependencies are declared in `pyproject.toml` and
installed with pip. If the system Python is older (common on shared servers),
get 3.11 via conda without needing root:

```bash
conda create -n hotel-search-engine python=3.11
conda activate hotel-search-engine
pip install -e ".[dev]"
```

Or, on a machine that already has a 3.11+ interpreter:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

The editable install provides the `hotel-search-engine-ingest` and
`hotel-search-engine-search` commands.

## Usage

```bash
# 1. ingest: embed reviews and upsert them into an embedded Qdrant at ./data/qdrant
hotel-search-engine-ingest --locality Paris

# 2. query it
hotel-search-engine-search "a quiet hotel near the Louvre" -k 5

# tests (similarity + chunking + vector store; need only numpy + qdrant-client)
pytest
```

To use a running Qdrant server instead of the embedded one, pass
`--qdrant-url http://localhost:6333` to both commands. If no store exists at
`data/qdrant`, `demo.py` builds an in-memory one on the fly.

## API

```bash
uvicorn hotel_search_engine.api.app:app --reload
```

- `POST /api/search` — ranked review hits for a query
- `POST /api/ask` — a generated, grounded answer with citations
- `GET  /api/health`
- interactive docs at `/docs`, OpenAPI schema at `/openapi.json`

The handlers currently return stub data; the request/response models in
`src/hotel_search_engine/api/schemas.py` is the contract the frontend builds against. Allowed CORS
origins default to `localhost:3000` / `localhost:5173` and can be overridden with
`HOTEL_SEARCH_ENGINE_CORS_ORIGINS`.

## Architecture

```
React frontend  --HTTPS/JSON-->  FastAPI (hotel_search_engine/api/app.py)
                                   |-- hotel search engine: embed + search
                                   |-- Qdrant (embedded or server)
                                   `-- Claude (answer generation, /api/ask)
```

All search and RAG logic is Python and server-side. The frontend only calls the
JSON API — it never touches embeddings, Qdrant, or the model. The contract
between them is `hotel_search_engine/api/schemas.py`'s Pydantic models, published as OpenAPI at
`/openapi.json`; the frontend generates its TypeScript types from that.

## Tech stack

**Backend**

- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) — JSON API
- [Sentence Transformers](https://www.sbert.net/) — text embeddings (`all-MiniLM-L6-v2`)
- [Qdrant](https://qdrant.tech/) — vector database (embedded or server), cosine distance
- [NumPy](https://numpy.org/) — cosine similarity reference implementation
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — hotel review dataset
- [pandas](https://pandas.pydata.org/) — filtering and data cleaning

**Frontend**

- [React](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/) + [Vite](https://vite.dev/)
- [TanStack Query](https://tanstack.com/query) — API request state
- responsive CSS with accessible form and result states

Run the frontend after starting the API:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the app at `http://localhost:5173` and proxies `/api` requests to
FastAPI at `http://localhost:8000`.

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
- Add the RAG generation step (`generator.py`) behind `/api/ask`
- Generate frontend types from the backend OpenAPI schema
