import pytesseract
from PIL import Image
from src.the_way_recognition.config import settings

class OCRService:
    @staticmethod
    def extract_text(image: Image.Image) -> str:
        return pytesseract.image_to_string(
            image,
            config=settings.TESSERACT_CONFIG,
            lang=settings.TESSERACT_LANG
        )
