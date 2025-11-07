from functools import lru_cache
import clip
import torch
import numpy as np
from PIL import Image
from src.the_way_recognition.config import settings

class EmbeddingService:
    def __init__(self):
        self.model, self.preprocess = self._load_model()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():
        return clip.load(settings.CLIP_MODEL, device=settings.DEVICE)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        img_tensor = self.preprocess(image).unsqueeze(0).to(settings.DEVICE)
        with torch.no_grad():
            embedding = self.model.encode_image(img_tensor).cpu().numpy().flatten()
        return embedding
