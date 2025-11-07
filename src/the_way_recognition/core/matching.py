from dataclasses import dataclass
from typing import Optional, Tuple, List
import Levenshtein
import numpy as np
from src.the_way_recognition.config import settings
from src.the_way_recognition.db.models import Card
from src.the_way_recognition.core.embeddings import EmbeddingService


@dataclass
class MatchResult:
    card: Optional[Card]
    text_score: float
    embedding_score: float
    is_card: bool
    confidence: str


class CardMatcher:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def get_best_text_match(
        self, ocr_text: str, cards: List[Card]
    ) -> Tuple[Optional[Card], float]:
        best_card = None
        best_score = 0.0

        for card in cards:
            similarity = Levenshtein.ratio(ocr_text, card.gt_text)

            if similarity > best_score:
                best_score = similarity
                best_card = card

        return best_card, best_score

    def get_best_embedding_match(
        self, image, cards: List[Card]
    ) -> Tuple[Optional[Card], float]:
        query_embedding = self.embedding_service.encode_image(image)

        best_card = None
        best_score = -1

        for card in cards:
            if card.gt_embedding:
                db_emb = np.frombuffer(card.gt_embedding, dtype=np.float32)
                score = self._cosine_similarity(query_embedding, db_emb)

                if score > best_score:
                    best_score = score
                    best_card = card

        return best_card, best_score

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def calculate_combined_score(
        self, text_score: float, emb_score: float, same_card: bool = False
    ) -> float:
        combined = text_score * settings.TEXT_WEIGHT + emb_score * settings.EMBED_WEIGHT

        # Apply consensus boost if both methods agree
        if same_card and text_score > 0 and emb_score > 0:
            combined = min(1.0, combined + settings.CONSENSUS_BOOST)

        return combined

    def select_best_match(
        self,
        text_card: Optional[Card],
        text_score: float,
        emb_card: Optional[Card],
        emb_score: float,
    ) -> MatchResult:

        # Case 1: Both methods agree on the same card
        if text_card and emb_card and text_card.name == emb_card.name:
            combined = self.calculate_combined_score(
                text_score, emb_score, same_card=True
            )

            if combined >= settings.CONFIDENCE_HIGH:
                return MatchResult(text_card, text_score, emb_score, True, "high")
            elif combined >= settings.CONFIDENCE_MEDIUM:
                return MatchResult(text_card, text_score, emb_score, True, "medium")
            elif combined >= settings.CONFIDENCE_LOW:
                return MatchResult(text_card, text_score, emb_score, True, "low")

        # Case 2: Only one method has high confidence
        if text_score >= settings.CONFIDENCE_HIGH:
            if emb_score < settings.CONFIDENCE_LOW or not emb_card:
                return MatchResult(text_card, text_score, text_score, True, "medium")

        if emb_score >= settings.CONFIDENCE_HIGH:
            if text_score < settings.CONFIDENCE_LOW or not text_card:
                return MatchResult(emb_card, emb_score, emb_score, True, "medium")

        # Case 3: Methods disagree but both have medium confidence
        if (
            text_card
            and emb_card
            and text_card.name != emb_card.name
            and text_score >= settings.CONFIDENCE_MEDIUM
            and emb_score >= settings.CONFIDENCE_MEDIUM
        ):
            if text_score > emb_score:
                return MatchResult(text_card, text_score, emb_score, True, "low")
            else:
                return MatchResult(emb_card, text_score, emb_score, True, "low")

        # Case 4: One method has medium confidence, other is low
        if text_score >= settings.CONFIDENCE_MEDIUM and text_card:
            if emb_score < settings.CONFIDENCE_MEDIUM:
                return MatchResult(text_card, text_score, emb_score, True, "low")

        if emb_score >= settings.CONFIDENCE_MEDIUM and emb_card:
            if text_score < settings.CONFIDENCE_MEDIUM:
                return MatchResult(emb_card, text_score, emb_score, True, "low")

        # Case 5: Both methods have some signal but below medium threshold
        combined = self.calculate_combined_score(text_score, emb_score, same_card=False)
        if combined >= settings.CONFIDENCE_MEDIUM:
            if text_score >= emb_score and text_card:
                return MatchResult(text_card, text_score, emb_score, True, "low")
            elif emb_card:
                return MatchResult(emb_card, text_score, emb_score, True, "low")

        # Case 6: No reliable match found
        return MatchResult(None, text_score, emb_score, False, "none")
