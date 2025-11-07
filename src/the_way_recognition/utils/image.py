from fastapi import UploadFile
from PIL import Image
from io import BytesIO
from src.the_way_recognition.config import settings


async def preprocess_image(file: UploadFile) -> Image.Image:
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Resize if too large
        if max(image.size) > settings.MAX_IMAGE_DIM:
            image.thumbnail(
                (settings.MAX_IMAGE_DIM, settings.MAX_IMAGE_DIM), Image.LANCZOS
            )

        return image
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")
