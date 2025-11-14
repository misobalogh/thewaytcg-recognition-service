# The Way Recognition Service

This project recognizes collectible card images and returns structured card data using OCR and image embeddings. You send a photo of a card, and the service tells you which card it is, how confident it is, and gives you match scores.

## How it works

You upload a card image to the API. The service runs OCR to extract text, then compares the text to a database of known cards using Levenshtein similarity. It also generates an image embedding using CLIP and compares it to stored embeddings. The service combines both scores to pick the best match and returns the result as JSON.

### Service Architecture:

```

                +-------------------+
                |   Client (User)   |
                +--------+----------+
                         |
                         v
                +-----------------------------+
                |  FastAPI Server             |
                |  (src/main.py)              |
                +--------+--------------------+
                         |
                         v
                +-----------------------------+
                |  /api/v1/recognize-card     |
                |  (Recognition Endpoint)     |
                +--------+--------------------+
                         |
                         v
                +-----------------------------+
                |  Image Preprocessing        |
                |  (utils/image.py)           |
                +--------+--------------------+
                         |
                         v
                +-----------------------------+
                |  OCR Service                |
                |  (core/ocr.py)              |
                +--------+--------------------+
                         |
                         v
                +-----------------------------+
                |  Card Matcher               |
                |  (core/matching.py)         |
                +--------+--------------------+
                         |
                         v
                +-----------------------------+
                |  Database (SQLite)          |
                |  (db/database.py, models.py)|
                +-----------------------------+
                         ^
                         |
                +--------+--------------------+
                |  Embedding Service          |
                |  (core/embeddings.py, CLIP) |
                +-----------------------------+
```

Data flow:
1. User uploads card image.
2. API preprocesses image.
3. OCR extracts text.
4. Card matcher compares OCR text and image embedding to database.
5. Service returns best match and scores.

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/misobalogh/thewayccg-recognition-service
cd thewayccg-recognition-service
uv sync
```

You can use [uv](https://docs.astral.sh/uv/) for dependency management. If you prefer Python only, run:

```bash
pip install -r requirements.txt
```

Set up a virtual environment if you want isolation:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Prepare data

1. Export your card data from Excel to CSV.
2. Save the CSV file as `data/cards.csv`.

Generate JSON schemas for each card:

```bash
uv run -m scripts.csv_to_json_schema
```

This creates JSON files for each card in `data/gt/json/` as `1.json`, `2.json`, and so on.

Download all card images as PDFs into `data/pdf/` (name them `1.pdf`, `2.pdf`, etc).

Convert PDFs to PNG images:

```bash
uv run -m scripts.pdf_to_png
```

This saves PNGs to `data/gt/png/`.

Generate image embeddings:

```bash
uv run -m scripts.get_embeddings
```

This creates `.npy` embedding files in `data/gt/npy/`.

You can now delete the PDFs and PNGs if you want to save space.

### 3. Insert data into the database

Load all card data and embeddings into the database:

```bash
uv run -m scripts.insert_cards
```

Check the database contents:

```bash
uv run -m scripts.view_cards
```

### 4. Run the service

Start the API server:

```bash
uv run run.py
```

The API is now available at `http://localhost:8000`.

## Docker

You can run the service in Docker if you prefer.

Build the Docker image:

```bash
make build
```

Start the container:

```bash
make up
```

The service will be available at `http://localhost:8000`.

Show logs:

```bash
make logs
```

Stop the container:

```bash
make down
```

Other commands:

```bash
make clean   # Remove image and volumes
make restart # Restart the container
```

## API Documentation

Swagger docs are available at `http://localhost:8000/docs` when the service is running.

### Endpoints

```
/api/v1/recognize-card/  # Accepts a card image (multipart/form-data), returns JSON with recognition result
/docs                    # Swagger documentation
/health                  # Health check
```

### Example response

```json
{
  "card": {
    "embedding_match_score": 0.92,
    "name": "Example Card",
    "text_match_score": 0.87
  },
  "confidence": "high",
  "is_card": true
}
```

You get the best match, match scores, and a confidence level for each request.
