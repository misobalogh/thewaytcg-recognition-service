"""In-process HTTP tests; no separately running server or notebook is required."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.the_way_recognition.config import Settings
from src.the_way_recognition.core.sift import Candidate, SIFTRecognitionService

ROOT = Path(__file__).resolve().parents[1]


def png(image):
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


@pytest.fixture(scope="session")
def catalog(tmp_path_factory):
    root = tmp_path_factory.mktemp("catalog")
    images, metadata = root / "png", root / "json"
    images.mkdir()
    metadata.mkdir()
    rng = np.random.default_rng(12)
    for card_id in range(1, 4):
        image = rng.integers(0, 256, (500, 350), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0.6)
        cv2.putText(image, str(card_id), (90, 270), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)
        (images / f"{card_id}.png").write_bytes(png(image))
        (metadata / f"{card_id}.json").write_text(json.dumps({"name": f"Card {card_id}"}))
    return Settings(_env_file=None, REFERENCE_IMAGE_DIR=images, CARD_METADATA_DIR=metadata)


@pytest.fixture(scope="session")
def service(catalog):
    return SIFTRecognitionService(catalog)


@pytest.fixture
def client(monkeypatch, service):
    import src.main as main
    monkeypatch.setattr(main, "SIFTRecognitionService", lambda settings: service)
    with TestClient(main.app) as client:
        yield client


def test_recognition_through_http(client, catalog):
    image = cv2.imdecode(np.frombuffer((catalog.REFERENCE_IMAGE_DIR / "2.png").read_bytes(), np.uint8), 0)
    # A transformed view, rather than a byte-identical reference upload.
    image = cv2.rotate(image, cv2.ROTATE_180)
    response = client.post("/api/v1/recognize-card", files={"file": ("rotated.png", png(image), "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert data["is_card"] is True
    assert data["card"]["id"] == "2"
    assert data["card"]["name"] == "Card 2"
    assert data["card"]["inliers"] >= 12
    assert data["card"]["sift_match_score"] > 0
    assert "embedding_match_score" not in data["card"]


@pytest.mark.parametrize("contents", [b"", b"not an image", b"\x89PNG\r\n\x1a\ncorrupt"])
def test_invalid_upload(client, contents):
    response = client.post("/api/v1/recognize-card", files={"file": ("bad.png", contents)})
    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]


def test_missing_upload(client):
    assert client.post("/api/v1/recognize-card").status_code == 422


def test_blank_image_is_rejected(client):
    response = client.post("/api/v1/recognize-card", files={"file": ("blank.png", png(np.full((600, 800), 160, np.uint8)))})
    assert response.status_code == 200
    data = response.json()
    assert data["is_card"] is False
    assert data["confidence"] == "none"
    assert data["card"]["id"] is None
    assert data["card"]["name"] is None
    assert data["card"]["inliers"] == 0


def test_noise_is_rejected(service):
    image = np.random.default_rng(42).integers(0, 256, (600, 800), dtype=np.uint8)
    assert service.recognize(png(image)).accepted is False


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_unready_index_returns_503(client, service):
    client.app.state.recognition_service = None
    try:
        response = client.post("/api/v1/recognize-card", files={"file": ("x.png", b"image")})
        assert response.status_code == 503
    finally:
        client.app.state.recognition_service = service


def test_missing_references_fail_startup(catalog, tmp_path):
    invalid = catalog.model_copy(update={"REFERENCE_IMAGE_DIR": tmp_path})
    with pytest.raises(ValueError, match="missing images"):
        SIFTRecognitionService(invalid)


def test_ambiguous_matches_are_rejected(service):
    ref = service.references[0]
    other = service.references[1]
    result = service._select([Candidate(ref, 40, 0.8, 0.5, 30, True), Candidate(other, 39, 0.8, 0.5, 29, True)])
    assert not result.accepted
    assert result.confidence == "none"


def test_concurrent_requests_share_index_safely(service, catalog):
    contents = (catalog.REFERENCE_IMAGE_DIR / "1.png").read_bytes()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(service.recognize, [contents] * 4))
    assert all(r.accepted and r.candidate.reference.card_id == "1" for r in results)
    assert len({r.candidate.score for r in results}) == 1


# Labels verified from the card names in the photos, independent of matcher output.
REAL_CASES = [
    ("apostol_pavol.jpg", "2"), ("damian.jpg", "17"), ("jan_pavol.jpg", "49"),
    ("jozef.jpg", "20"), ("martin.jpg", "34"), ("maximilian.jpg", "4"),
    ("patrik.jpg", "32"), ("terezia.jpg", "36"),
    ("valentin.jpg", "35"), ("ludmila.jpg", "37"),
    ("atanaz.jpg", "41"), ("lukas.jpg", "7"),
    ("don_bosco.jpg", "22"), ("lucia.jpg", "43"),
    ("cyril_a_metod.jpg", "16"), ("maria_gorretti.jpg", "19"),
    ("katarina_alexandrijska.jpg", "31"), ("anton.jpg", "18"),
]


@pytest.fixture(scope="session")
def real_service():
    config = Settings(_env_file=None)
    if not config.REFERENCE_IMAGE_DIR.is_dir() or not config.CARD_METADATA_DIR.is_dir():
        pytest.skip("Local reference catalog is not available")
    return SIFTRecognitionService(config)


@pytest.mark.parametrize("filename, expected_id", REAL_CASES)
def test_real_photo_identification(real_service, filename, expected_id):
    path = ROOT / "tests/test_data" / filename
    if not path.exists():
        pytest.skip(f"Local photo missing: {filename}")
    result = real_service.recognize(path.read_bytes())
    assert result.accepted, f"Photo rejected: {filename}"
    assert result.candidate.reference.card_id == expected_id
