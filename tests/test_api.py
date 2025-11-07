import pytest
import requests
from pathlib import Path
import io
from PIL import Image

API_URL = "http://127.0.0.1:8000/api/v1/recognize-card"
SAMPLES_DIR = Path("data/")
TEST_IMAGE = "1.jpg"
TIMEOUT = 30  # seconds


@pytest.fixture
def api_url():
    return API_URL


class TestRecognizeCardEndpoint:

    @pytest.mark.parametrize("image_num", range(1, 9))
    def test_recognize_card_all_samples(self, api_url, image_num):
        image_path = SAMPLES_DIR / f"{image_num}.jpg"

        if not image_path.exists():
            pytest.skip(f"Sample image {image_path} not found")

        with open(image_path, "rb") as img_file:
            files = {"file": (f"{image_num}.jpg", img_file, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=TIMEOUT)

        assert response.status_code == 200, f"Failed for image {image_num}.jpg"
        data = response.json()

        assert "is_card" in data
        assert isinstance(data["is_card"], bool)
        assert "card" in data
        assert "name" in data["card"]
        assert "text_match_score" in data["card"]
        assert "embedding_match_score" in data["card"]

        assert 0.0 <= data["card"]["text_match_score"] <= 1.0
        assert 0.0 <= data["card"]["embedding_match_score"] <= 1.0

    def test_response_structure(self, api_url):
        image_path = SAMPLES_DIR / TEST_IMAGE

        if not image_path.exists():
            pytest.skip("Sample image not found")

        with open(image_path, "rb") as img_file:
            files = {"file": (TEST_IMAGE, img_file, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert set(data.keys()) == {"is_card", "card", "confidence"}

        assert set(data["card"].keys()) == {
            "name",
            "text_match_score",
            "embedding_match_score",
        }

        assert isinstance(data["is_card"], bool)
        assert isinstance(data["card"]["name"], (str, type(None)))
        assert isinstance(data["card"]["text_match_score"], (int, float))
        assert isinstance(data["card"]["embedding_match_score"], (int, float))

    def test_invalid_file_format(self, api_url):
        fake_file = io.BytesIO(b"This is not an image file")
        files = {"file": ("test.txt", fake_file, "text/plain")}

        response = requests.post(api_url, files=files, timeout=TIMEOUT)

        assert response.status_code == 400
        assert "Invalid image file" in response.json()["detail"]

    def test_corrupted_image(self, api_url):
        corrupted_data = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"corrupted data")
        files = {"file": ("corrupted.png", corrupted_data, "image/png")}

        response = requests.post(api_url, files=files, timeout=TIMEOUT)

        assert response.status_code == 400

    def test_missing_file_parameter(self, api_url):
        response = requests.post(api_url, timeout=TIMEOUT)

        assert response.status_code == 422  # Unprocessable Entity

    def test_empty_file(self, api_url):
        empty_file = io.BytesIO(b"")
        files = {"file": ("empty.jpg", empty_file, "image/jpeg")}

        response = requests.post(api_url, files=files, timeout=TIMEOUT)

        assert response.status_code == 400

    @pytest.mark.parametrize(
        "format,mime",
        [
            ("JPEG", "image/jpeg"),
            ("PNG", "image/png"),
            ("BMP", "image/bmp"),
            ("WEBP", "image/webp"),
        ],
    )
    def test_different_image_formats(self, api_url, format, mime):
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format)
        img_bytes.seek(0)

        files = {"file": (f"test.{format.lower()}", img_bytes, mime)}
        response = requests.post(api_url, files=files, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        assert "is_card" in data
        assert "card" in data

    @pytest.mark.parametrize(
        "width,height",
        [
            (50, 50),  # Very small
            (500, 500),  # Medium
            (2000, 2000),  # Large (should be thumbnailed)
            (100, 2000),  # Tall
            (2000, 100),  # Wide
        ],
    )
    def test_different_image_sizes(self, api_url, width, height):
        img = Image.new("RGB", (width, height), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {"file": (f"test_{width}x{height}.jpg", img_bytes, "image/jpeg")}
        response = requests.post(api_url, files=files, timeout=TIMEOUT)

        assert response.status_code == 200

    def test_response_time(self, api_url):
        image_path = SAMPLES_DIR / TEST_IMAGE

        if not image_path.exists():
            pytest.skip("Sample image not found")

        with open(image_path, "rb") as img_file:
            files = {"file": (TEST_IMAGE, img_file, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=TIMEOUT)

        assert response.elapsed.total_seconds() < TIMEOUT
        assert response.status_code == 200

    def test_card_detection_logic(self, api_url):
        image_path = SAMPLES_DIR / TEST_IMAGE

        if not image_path.exists():
            pytest.skip("Sample image not found")

        with open(image_path, "rb") as img_file:
            files = {"file": (TEST_IMAGE, img_file, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=TIMEOUT)

        data = response.json()

        if data["is_card"]:
            assert (
                data["card"]["text_match_score"] >= 0.5
                or data["card"]["embedding_match_score"] >= 0.5
            )

        if not data["is_card"]:
            assert data["card"]["name"] is None

    def test_multiple_consecutive_requests(self, api_url):
        image_path = SAMPLES_DIR / TEST_IMAGE

        if not image_path.exists():
            pytest.skip("Sample image not found")

        for i in range(3):
            with open(image_path, "rb") as img_file:
                files = {"file": (TEST_IMAGE, img_file, "image/jpeg")}
                response = requests.post(api_url, files=files, timeout=TIMEOUT)

            assert response.status_code == 200
            data = response.json()
            assert "is_card" in data

    def test_score_precision(self, api_url):
        image_path = SAMPLES_DIR / TEST_IMAGE

        if not image_path.exists():
            pytest.skip("Sample image not found")

        with open(image_path, "rb") as img_file:
            files = {"file": (TEST_IMAGE, img_file, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=TIMEOUT)

        data = response.json()

        text_score_str = str(data["card"]["text_match_score"])
        emb_score_str = str(data["card"]["embedding_match_score"])

        if "." in text_score_str:
            assert len(text_score_str.split(".")[1]) <= 4
        if "." in emb_score_str:
            assert len(emb_score_str.split(".")[1]) <= 4

    def test_confidence_based_detection(self, api_url):
        image_path = SAMPLES_DIR / TEST_IMAGE

        if not image_path.exists():
            pytest.skip("Sample image not found")

        with open(image_path, "rb") as img_file:
            files = {"file": (TEST_IMAGE, img_file, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=TIMEOUT)

        data = response.json()

        if data["is_card"]:
            if (
                data["card"]["text_match_score"] >= 0.75
                or data["card"]["embedding_match_score"] >= 0.75
            ):
                assert data["card"]["name"] is not None

        if (
            data["card"]["text_match_score"] < 0.3
            and data["card"]["embedding_match_score"] < 0.3
        ):
            assert data["is_card"] is False

    def test_consensus_detection(self, api_url):
        results = []

        for i in range(1, 9):
            image_path = SAMPLES_DIR / f"{i}.jpg"
            if not image_path.exists():
                continue

            with open(image_path, "rb") as img_file:
                files = {"file": (f"{i}.jpg", img_file, "image/jpeg")}
                response = requests.post(api_url, files=files, timeout=TIMEOUT)

            if response.status_code == 200:
                data = response.json()
                results.append(
                    {
                        "image": i,
                        "is_card": data["is_card"],
                        "text_score": data["card"]["text_match_score"],
                        "emb_score": data["card"]["embedding_match_score"],
                        "name": data["card"]["name"],
                    }
                )

        assert len(results) > 0, "No test images were processed"

    @pytest.mark.parametrize("image_num", range(1, 9))
    def test_no_false_negatives_on_samples(self, api_url, image_num):
        image_path = SAMPLES_DIR / f"{image_num}.jpg"

        if not image_path.exists():
            pytest.skip(f"Sample image {image_path} not found")

        with open(image_path, "rb") as img_file:
            files = {"file": (f"{image_num}.jpg", img_file, "image/jpeg")}
            response = requests.post(api_url, files=files, timeout=TIMEOUT)

        data = response.json()

        has_some_signal = (
            data["card"]["text_match_score"] > 0.5
            or data["card"]["embedding_match_score"] > 0.5
        )
        assert has_some_signal, f"Image {image_num}.jpg shows no recognition signal"


class TestAPIHealth:

    def test_api_is_running(self, api_url):
        try:
            image_path = SAMPLES_DIR / TEST_IMAGE
            if image_path.exists():
                with open(image_path, "rb") as img_file:
                    files = {"file": (TEST_IMAGE, img_file, "image/jpeg")}
                    response = requests.post(api_url, files=files, timeout=5)
                assert response.status_code in [200, 400, 422]
        except requests.exceptions.ConnectionError:
            pytest.fail("API server is not running or not accessible")
