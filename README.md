# Hotel Search Engine

A full-stack semantic search application for discovering hotels in Paris through
real guest reviews. Users describe the stay they want in natural language, and
the application returns the most relevant review passages with their hotel names
and similarity scores.

## Quick start

### Prerequisites

- Python 3.11 or newer
- Node.js and npm
- Internet access during the initial dataset and model download

### 1. Install the backend

```bash
git clone https://github.com/YacineDeghaies/hotel-search-engine.git
cd hotel-search-engine

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Build the search index

```bash
hotel-search-engine-ingest --locality Paris
```

This downloads the review dataset and embedding model, then writes the local
Qdrant index to `data/qdrant`. Both downloads are cached for subsequent runs.

### 3. Verify search from the command line

```bash
hotel-search-engine-search "a quiet hotel near the Louvre" -k 5
```

### 4. Start the API

```bash
uvicorn hotel_search_engine.api.app:app --reload
```

The API is available at `http://localhost:8000`. Interactive documentation is
available at `http://localhost:8000/docs`.

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. During development, Vite proxies `/api` requests
to FastAPI at `http://localhost:8000`.

## Key features

- Search across Paris hotel reviews using NLP
- Review chunking and SBERT embeddings
- Cosine-similarity retrieval backed by Qdrant
- Persistent local indexes with optional remote Qdrant support
- FastAPI search endpoint with an OpenAPI contract
- React and TypeScript frontend with loading and error states
- CLI tools for ingestion and search
- Unit tests for search primitives, storage, and the API

## Architecture

```text
React + TypeScript
        |
        | POST /api/search
        v
     FastAPI
        |
        | Embed query and retrieve nearest vectors
        v
      Qdrant <---- Sentence Transformers
        |
        v
Ranked hotel reviews
```

The browser communicates only with FastAPI. Dataset loading, embedding, vector
storage, and retrieval remain in the Python backend.

## Technology stack

### Backend

- Python 3.11+
- FastAPI and Uvicorn
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Qdrant
- Hugging Face Datasets
- pandas and NumPy
- pytest and Ruff

### Frontend

- React 19
- TypeScript
- Vite
- TanStack Query
- CSS

## API

| Method | Endpoint | Status | Description |
| --- | --- | --- | --- |
| `GET` | `/api/health` | Ready | Service health and API version |
| `POST` | `/api/search` | Ready | Ranked hotel-review matches |
| `POST` | `/api/ask` | Placeholder | Planned grounded answer with citations |

### Search example

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"a quiet hotel near the Louvre","k":5}'
```

Example response:

```json
{
  "query": "a quiet hotel near the Louvre",
  "hits": [
    {
      "hotel_name": "Hotel Moliere",
      "review": "One of the quietest rooms I ever slept in in Paris...",
      "score": 0.805,
      "locality": "Paris"
    }
  ]
}
```

## Configuration

The API supports the following environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOTEL_SEARCH_ENGINE_QDRANT_PATH` | `data/qdrant` | Local Qdrant index path |
| `HOTEL_SEARCH_ENGINE_QDRANT_URL` | Unset | Remote Qdrant server URL |
| `HOTEL_SEARCH_ENGINE_COLLECTION` | `hotel_reviews` | Qdrant collection name |
| `HOTEL_SEARCH_ENGINE_CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated browser origins |

When `HOTEL_SEARCH_ENGINE_QDRANT_URL` is set, the API uses the remote server
instead of the embedded local database.

The ingestion and CLI commands also accept `--qdrant-url`, `--qdrant-path`, and
`--collection` options. Run either command with `--help` for the full list.

## Project structure

```text
hotel-search-engine/
├── frontend/                       # React and TypeScript application
├── src/hotel_search_engine/
│   ├── api/                        # FastAPI routes and schemas
│   ├── search/                     # Chunking, embeddings, retrieval, and Qdrant
│   ├── data.py                     # Dataset loading and cleaning
│   ├── ingest.py                   # Index-building pipeline
│   └── demo.py                     # Search CLI
├── tests/                          # Unit and integration tests
├── pyproject.toml                  # Python package and tool configuration
└── README.md
```

The installable project name is `hotel-search-engine`. Python imports use
`hotel_search_engine` because Python module names cannot contain hyphens.

## Development

Run backend tests and lint checks from the repository root:

```bash
pytest
ruff check .
```

Build the frontend:

```bash
cd frontend
npm run build
```

## Data and embeddings

Reviews come from the
[Traversaal AI hotel dataset](https://huggingface.co/datasets/traversaal-ai-hackathon/hotel_datasets).
The ingestion pipeline filters the dataset to Paris, removes empty reviews,
splits reviews into overlapping passages, and embeds them with
`sentence-transformers/all-MiniLM-L6-v2`.

Generated indexes and downloaded data are not committed to Git.

## Roadmap

- Add an end-to-end ingestion test
- Implement grounded answer generation behind `/api/ask`
- Generate frontend types from the OpenAPI schema
- Remove the remaining locality option from this Paris-only application
