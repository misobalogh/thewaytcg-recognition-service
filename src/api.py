from fastapi import FastAPI, File, UploadFile, HTTPException
from src.models import Card
from io import BytesIO
from src.database import SessionLocal
import pytesseract
from PIL import Image
import numpy as np
import clip
import torch
import Levenshtein

MAX_DIM = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-B/32"


class Confidence:
    HIGH = 0.75
    MEDIUM = 0.5
    LOW = 0.3


class ScoreWeights:
    # Boost when both methods agree - additive
    CONSENSUS_BOOST = 0.15

    # Weights for combined score
    TEXT_WEIGHT = 0.4
    EMBED_WEIGHT = 0.6


class TesseractConfig:
    PSM_SINGLE_LINE = "--psm 7"
    PSM_BLOCK = "--psm 6"
    LANG_SLK = "slk"
    LANG_ENG = "eng"


model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)

app = FastAPI()


def get_best_text_match(ocr_text, cards):
    best_card = None
    best_score = 0.0

    for card in cards:
        similarity = Levenshtein.ratio(ocr_text, card.gt_text)

        if similarity > best_score:
            best_score = similarity
            best_card = card

    return best_card, best_score


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_best_embedding_match(image, cards):
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        query_embedding = model.encode_image(img_tensor).cpu().numpy().flatten()

    best_card = None
    best_score = -1

    for card in cards:
        if card.gt_embedding:
            db_emb = np.frombuffer(card.gt_embedding, dtype=np.float32)
            score = cosine_similarity(query_embedding, db_emb)

            if score > best_score:
                best_score = score
                best_card = card

    return best_card, best_score


def calculate_combined_score(text_score, emb_score, same_card=False):
    combined = (
        text_score * ScoreWeights.TEXT_WEIGHT + emb_score * ScoreWeights.EMBED_WEIGHT
    )

    # Apply consensus boost if both methods agree
    if same_card and text_score > 0 and emb_score > 0:
        combined = min(1.0, combined + ScoreWeights.CONSENSUS_BOOST)

    return combined


def select_best_match(text_card, text_score, emb_card, emb_score):
    # Case 1: Both methods agree on the same card
    if text_card and emb_card and text_card.name == emb_card.name:
        combined = calculate_combined_score(text_score, emb_score, same_card=True)

        if combined >= Confidence.HIGH:
            return text_card, text_score, emb_score, True, "high"
        elif combined >= Confidence.MEDIUM:
            return text_card, text_score, emb_score, True, "medium"
        elif combined >= Confidence.LOW:
            return text_card, text_score, emb_score, True, "low"

    # Case 2: Only one method has high confidence
    if text_score >= Confidence.HIGH:
        if emb_score < Confidence.LOW or not emb_card:
            # Text is confident, embedding is weak - trust text
            return text_card, text_score, text_score, True, "medium"

    if emb_score >= Confidence.HIGH:
        if text_score < Confidence.LOW or not text_card:
            # Embedding is confident, text is weak - trust embedding
            return emb_card, emb_score, emb_score, True, "medium"

    # Case 3: Methods disagree but both have medium confidence
    if (
        text_card
        and emb_card
        and text_card.name != emb_card.name
        and text_score >= Confidence.MEDIUM
        and emb_score >= Confidence.MEDIUM
    ):

        # Choose the method with higher score, but lower confidence
        if text_score > emb_score:
            return text_card, text_score, emb_score, True, "low"
        else:
            return emb_card, text_score, emb_score, True, "low"

    # Case 4: One method has medium confidence, other is low
    if text_score >= Confidence.MEDIUM and text_card:
        if emb_score < Confidence.MEDIUM:
            return text_card, text_score, emb_score, True, "low"

    if emb_score >= Confidence.MEDIUM and emb_card:
        if text_score < Confidence.MEDIUM:
            return emb_card, text_score, emb_score, True, "low"

    # Case 5: Both methods have some signal but below medium threshold
    combined = calculate_combined_score(text_score, emb_score, same_card=False)
    if combined >= Confidence.MEDIUM:
        # Use the method with higher individual score
        if text_score >= emb_score and text_card:
            return text_card, text_score, emb_score, True, "low"
        elif emb_card:
            return emb_card, text_score, emb_score, True, "low"

    # Case 6: No reliable match found
    return None, text_score, emb_score, False, "none"


@app.post("/recognize-card")
async def recognize_card(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        if max(image.size) > MAX_DIM:
            image.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    ocr_text = pytesseract.image_to_string(
        image, config=TesseractConfig.PSM_BLOCK, lang=TesseractConfig.LANG_SLK
    )

    with SessionLocal() as session:
        cards = session.query(Card).all()

    best_text_card, best_text_score = get_best_text_match(ocr_text, cards)
    best_emb_card, best_emb_score = get_best_embedding_match(image, cards)

    # Use smart selection logic
    selected_card, final_text_score, final_emb_score, is_card, confidence = (
        select_best_match(
            best_text_card, best_text_score, best_emb_card, best_emb_score
        )
    )

    return {
        # ============ FOR DEBUGGING =============
        # "best_text_match": {
        #     "card_name": best_text_card.name if best_text_card else None,
        #     "score": f"{float(best_text_score):.4f}",
        #     "ocr_text": ocr_text,
        #     "gt_text": best_text_card.gt_text if best_text_card else None,
        # },
        # "best_embedding_match": {
        #     "card_name": best_emb_card.name if best_emb_card else None,
        #     "score": f"{float(best_emb_score):.4f}",
        # },
        # ======================================
        "is_card": is_card,
        "card": {
            "name": selected_card.name if selected_card else None,
            "text_match_score": round(float(final_text_score), 4),
            "embedding_match_score": round(float(final_emb_score), 4),
        },
        "confidence_level": confidence,
    }
