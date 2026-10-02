# The Way Recognition Service

Recognizes The Way cards from photos and returns the card ID, name and match
metrics. Recognition uses SIFT features, FLANN candidate selection and RANSAC
geometric verification.

## Required data

Copy the reference images and card metadata to the server before starting the
service. These files are not included in the repository or Docker image.

```text
data/
├── gt/png/
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── json/
    ├── 1.json
    ├── 2.json
    └── ...
```

Each card needs a reference image and a JSON file with the same ID. For example,
`data/gt/png/41.png` pairs with `data/json/41.json`:

```json
{
  "name": "ATANAZ ALEXANDRIJSKY"
}
```

The JSON must contain a nonempty `name`; additional fields are allowed.
The service validates the catalog and builds its search index at startup.
Missing or invalid references prevent startup. Restart after changing the data.

## Run with Docker

Place the data in the directories above, then run from the project root:

```bash
make build
make up
```

The service is available at `http://localhost:8000`. Docker Compose mounts both
data directories read-only. To use different directories, update the volume
mounts and data paths in `docker/docker-compose.yml`.

```bash
make logs   # View logs
make down   # Stop the service
```

## Run locally

Install dependencies with [uv](https://docs.astral.sh/uv/), then start the service:

```bash
uv sync --no-dev
uv run --no-dev uvicorn src.main:app --host 0.0.0.0 --port 8000
```

For development with automatic reload:

```bash
uv run run.py
```

## API

Interactive API documentation: `http://localhost:8000/docs`.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/recognize-card` | Recognize a photo uploaded as multipart field `file` |
| `GET /health` | Check that the service is running |

Upload a photo:

```bash
curl -F "file=@/path/to/card.jpg" http://localhost:8000/api/v1/recognize-card
```

Example response:

```json
{
  "is_card": true,
  "confidence": "high",
  "card": {
    "id": "32",
    "name": "PATRIK",
    "sift_match_score": 45.0,
    "inliers": 80,
    "inlier_ratio": 0.75,
    "coverage": 0.5625,
    "match_margin": 3.4
  }
}
```

| Field | Meaning |
| --- | --- |
| `is_card` | A match to a card in the reference catalog was accepted |
| `confidence` | Match strength: `high`, `medium`, `low` or `none`; not a probability |
| `card.id`, `card.name` | ID and name of the accepted card |
| `card.sift_match_score` | Ranking score: `inliers × inlier_ratio × sqrt(coverage)` |
| `card.inliers` | Number of geometrically verified feature matches |
| `card.inlier_ratio` | Fraction of candidate feature matches that passed geometric verification |
| `card.coverage` | Fraction of the reference image area spanned by verified points |
| `card.match_margin` | Best valid score divided by the runner-up score; null when there is no positive runner-up score |

Unrecognized or ambiguous photos return `is_card: false`, `confidence: "none"`
and null `card.id`/`card.name`. Match metrics may describe the rejected candidate.
Empty, invalid or corrupt uploads return HTTP 400; a missing `file` returns 422.
An unavailable recognition index returns 503.

## Configuration

For local execution, configuration is read from environment variables or `.env`.
See [.env.example](.env.example) for all settings. For Docker, set environment
variables in `docker/docker-compose.yml`.

| Setting | Default |
| --- | --- |
| `REFERENCE_IMAGE_DIR` | `data/gt/png` under the project root |
| `CARD_METADATA_DIR` | `data/json` under the project root |
| `MAX_IMAGE_DIM` | `900` |
| `SIFT_NFEATURES` | `1200` |
| `SIFT_SHORTLIST_SIZE` | `10` |
| `SIFT_MIN_INLIERS` | `12` |
| `SIFT_MIN_INLIER_RATIO` | `0.45` |
| `SIFT_MIN_COVERAGE` | `0.04` |
| `SIFT_MIN_MARGIN` | `1.25` |

## Tests

```bash
uv run pytest -q
```

Synthetic tests run without reference data. Real-photo tests run when their
reference catalog and test photos are available; otherwise they are skipped.
