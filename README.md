# The Way Recognition Service

CPU-only recognition of The Way cards using SIFT, FLANN candidate selection and
RANSAC homography. No GPU, CLIP, OCR or database is required.

Each worker loads the reference catalog once at startup. A photo is resized to
900 pixels on its longest side, then SIFT features vote for ten candidates in a
shared FLANN index. Those candidates are verified against their reference
features using a ratio test and RANSAC. The best candidate must pass the minimum
inlier, inlier ratio, coverage and geometry checks, and beat the runner-up by at
least 1.25 times. Ambiguous images and images without enough features are rejected.

## Setup

```bash
uv sync
```

Card metadata goes in `data/json/<id>.json`, with at least a nonempty `name`.
Reference images go in `data/gt/png/<id>.png`. Their IDs must match exactly; an
incomplete catalog fails startup with an actionable error. The existing
`data/cards.csv` can generate the metadata:

```bash
uv run -m scripts.csv_to_json_schema
```

For the existing two ZIP archives of numbered PDF cards, place them in `data/`
and render the reference images:

```bash
uv run -m scripts.prepare_references
```

The same command also supports numbered PDFs in `data/pdf/`. It renders one-page
PDFs at a maximum dimension of 1400 pixels and rejects duplicate/missing IDs.
It overwrites generated PNGs when rerun. Keep the PDFs/ZIPs for rebuilding; keep
the PNGs available to the service. Data is local and excluded from Git.

Start the service:

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

For development with reload, `uv run run.py` remains available. Configuration is
read from `.env`; see `.env.example`. Defaults match the fast notebook: 1200
features, ten candidates, one OpenCV thread. The service serializes recognition
within each worker because the SIFT/matcher objects are shared. CPU work runs in
a thread pool so health checks and other async routes remain responsive. Extra
workers each build their own index and consume their own memory. Restart after
changing reference data.

## API

- `POST /api/v1/recognize-card` — multipart upload named `file`.
- `GET /health` — available once startup has completed.
- `/docs` — Swagger UI.

```bash
curl -F file=@tests/test_data/patrik.jpg http://localhost:8000/api/v1/recognize-card
```

Example accepted response (values are illustrative):

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

`is_card` means an accepted match to the known catalog, not a general detector of
all cards. Rejected images return `is_card: false`, `confidence: "none"`, and null
`id`/`name`. Diagnostic metrics may describe the rejected best candidate.

`sift_match_score = inliers × inlier_ratio × sqrt(coverage)` is a ranking score,
not a probability or a value limited to 0–1. Coverage is the area spanned by the
inlier reference points divided by reference image area. `match_margin` is the
best valid score divided by the runner-up valid score; null means there is no
positive runner-up score. Confidence labels are heuristics: `high` requires at
least 30 inliers, inlier ratio at least 0.6, and a margin of at least 2 (or no
positive runner-up); `medium` requires the strong margin; other accepted matches
are `low`. These labels are not calibrated on a large dataset.

**API change from the OCR/CLIP implementation:** `card.text_match_score` and
`card.embedding_match_score` are removed. Use the SIFT diagnostic fields above.
The endpoint, `is_card`, `confidence` and `card.name` remain. Clients displaying or
validating the old score fields must update when adopting this branch.

Bad, empty or corrupt uploads return HTTP 400; a missing `file` returns 422. An
unavailable index returns 503. Old OCR/CLIP `.env` keys are ignored.

## Docker

Prepare the PNGs and JSON metadata on the host first, then:

```bash
make build
make up
```

Compose mounts both directories read-only. The image contains only the runtime
code and dependencies, with headless OpenCV, no GPU libraries or Tesseract. The
healthcheck uses Python's standard library. For custom data paths, update the
Compose mounts and corresponding environment variables.

## Verification and experiments

```bash
uv run pytest -q
```

Tests use an in-process FastAPI client. Synthetic fixtures cover uploads,
rotation, rejection, ambiguous candidates, catalog validation and concurrent
requests. The real-photo regression tests additionally check the independently
labeled 18 local photos against the full catalog. Missing local assets are
reported as skips; synthetic tests do not need those assets.

Notebook dependencies are separate from runtime:

```bash
uv sync --group notebooks
```

Open `prototypes/sift_fast_experiment.ipynb` in VS Code and use the local Jupyter
kernel. The original `sift_experiment.ipynb` is the full-catalog baseline. The fast
experiment selects ten candidates, benchmarks both variants, and visualizes all
photos in `tests/test_data`. Add labels via `data/sift_queries.csv` with columns
`path,expected_id`; paths are relative to the project, empty IDs are negatives.

The initial eight real photos achieved 8/8 accepted correct matches, with median
latency around 0.55 seconds on one CPU thread, versus 1.75 seconds for full-catalog
verification. A further ten photos were visually checked successfully in the
notebook. This small set does not establish general accuracy; test more glare,
blur, distant cards, similar editions and real negative images before deployment.
The old `cards.db` and historical notebooks may remain locally; the running
service does not read the database or use embedding files.
