from src.models import Card, Base
from src.the_way_recognition.database import engine, SessionLocal
import numpy as np
import json
from src.ocr_recognition.utils import card_json_to_text

Base.metadata.create_all(bind=engine)

def load_embedding(path):
    arr = np.load(path)
    return arr.tobytes()

def insert_card(json_path, embedding_path):
    with SessionLocal() as session:
        with open(json_path, 'r', encoding='utf-8') as f:
            card_data = json.load(f)
        name = card_data.get('name', '')
        edition = card_data.get('edition', '')
        rarity = card_data.get('rarity', '')
        gt_text = card_json_to_text(json_path)
        gt_embedding = load_embedding(embedding_path)
        card = Card(
            name=name,
            edition=edition,
            rarity=rarity,
            gt_text=gt_text,
            gt_embedding=gt_embedding,
        )
        session.add(card)
        session.commit()

if __name__ == "__main__":
    import os

    gt_path = 'data/gt/'
    npy_path = os.path.join(gt_path, "npy")
    json_path = os.path.join(gt_path, "json")
    for filename in os.listdir(json_path):
        if filename.endswith('.json'):
            json_file = os.path.join(json_path, filename)
            emb_file = os.path.join(npy_path, filename.replace('.json', '.npy'))
            insert_card(json_file, emb_file)
