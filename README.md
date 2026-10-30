# Hotel Recommendation Engine

A full-stack semantic search application for discovering hotels in Paris through
real guest reviews. Users describe the stay they want in natural language, and
the application ranks matching hotels and shows the strongest matching review
passage for each hotel, along with its similarity score.

## Quick start

### Prerequisites

- Python 3.11 or newer
- Node.js and npm

### 1. Install the backend

```bash
git clone https://github.com/YacineDeghaies/hotel_recommendation_engine.git hotel-search-engine
cd hotel-search-engine

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 2. Build the search index

```bash
hotel-search-engine-ingest --locality Paris
```

This downloads the review dataset and embedding model, then writes the local
Qdrant index to `data/qdrant`. Both downloads are cached for subsequent runs.
Ingestion recreates the selected collection by default; use `--no-recreate` to
upsert into an existing collection.

### 3. Verify search from the CLI

```bash
hotel-search-engine-search "a quiet hotel near the Louvre" -k 5
```

The CLI returns review-level matches, so a hotel can appear more than once.
The API and frontend group matches by hotel.

### 4. Start the API

```bash
uvicorn hotel_recommendation_engine.api.app:app --reload
```

The API is available at `http://localhost:8000`. Interactive documentation is
available at `http://localhost:8000/docs`.

Finish ingestion and CLI searches before starting the API when using local
Qdrant storage. Stop the API before rebuilding that index; use a Qdrant server
if multiple processes need to access the same collection concurrently.

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. During development, Vite proxies `/api` requests
to FastAPI at `http://localhost:8000`.

## Key features

- Search across Paris hotel reviews using natural language
- Unique hotel results in the API and frontend, reranked by a cross-encoder
- Semantic matching beyond exact keywords
- Review chunking and SBERT embeddings
- Cosine-similarity retrieval backed by Qdrant
- Cross-encoder reranking of retrieved review passages
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
        | Rerank candidate passages with a cross-encoder
        v
Ranked hotels with matching review passages
```

The browser communicates only with FastAPI. Dataset loading, embedding, vector
storage, and retrieval remain in the Python backend.

The reranking stage follows the query-passage cross-encoder approach described in
Nogueira and Cho's [*Passage Re-ranking with BERT*](https://arxiv.org/abs/1901.04085).
The implementation uses the [MS MARCO MiniLM-L6-v2 cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
not the BERT model evaluated in that paper.

## Technology stack

### Backend

- Python 3.11+
- FastAPI and Uvicorn
- Sentence Transformers (`all-MiniLM-L6-v2` for retrieval and
  `cross-encoder/ms-marco-MiniLM-L6-v2` for reranking)
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
| `POST` | `/api/search` | Ready | Ranked hotels with their best matching review passages |
| `POST` | `/api/ask` | Placeholder | Planned grounded answer with citations |

### Search example

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"a quiet hotel near the Louvre","k":5}'
```

Illustrative response (actual results depend on the indexed reviews):

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

`query` is required and must contain search text. `k` defaults to `5` and accepts
values from `1` to `50`. The API retrieves up to five times `k` review candidates
(at most 250), reranks them with a cross-encoder, keeps the strongest reranked
passage per hotel, and returns up to `k` hotels. Fewer results are possible when
candidates belong to the same hotels. The cross-encoder model downloads on the
first search request and is cached for later requests.

The optional `locality` request field is currently ignored; choose the locality
when building the index. The returned `score` is the original cosine similarity;
result order follows the cross-encoder's relevance scores. Neither score is a
calibrated recommendation probability. `/api/health` reports API availability without
checking the index, and `/api/ask` currently returns a fixed placeholder response.

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
These commands use their flags rather than the API environment variables. Keep
their storage location and collection aligned with the API configuration, and
run local commands from the repository root so relative paths resolve consistently.

## Project structure

```text
hotel-search-engine/
├── frontend/                       # React and TypeScript application
├── src/hotel_recommendation_engine/
│   ├── api/                        # FastAPI routes and schemas
│   ├── search/                     # Chunking, embeddings, retrieval, and Qdrant
│   ├── data.py                     # Dataset loading and cleaning
│   ├── ingest.py                   # Index-building pipeline
│   └── demo.py                     # Search CLI
├── tests/                          # Unit tests and integration-test scaffold
├── pyproject.toml                  # Python package and tool configuration
└── README.md
```

The installable project name is `hotel-recommendation-engine`. Python imports use
`hotel_recommendation_engine` because Python module names cannot contain hyphens.

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
The ingestion pipeline defaults to Paris, removes empty reviews,
splits reviews into overlapping passages, and embeds them with
`sentence-transformers/all-MiniLM-L6-v2`.

## Planned search quality evaluation

The next feature is a reproducible evaluation dashboard for hotel search. It will
use a fixed set of realistic queries with human-labeled relevant hotels to compare
vector search alone against vector search followed by the cross-encoder reranker.
The report should show Recall@5, ranking quality, and search latency for both
versions, plus examples where reranking improves or worsens the results. These
measurements are not implemented yet.

## Roadmap

- Build the labeled query set and search quality evaluation dashboard
- Add an end-to-end ingestion test
- Implement grounded answer generation behind `/api/ask`
- Generate frontend types from the OpenAPI schema
- Remove the remaining locality option from this Paris-only application
