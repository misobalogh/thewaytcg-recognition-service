import json
from pathlib import Path

import numpy as np

from src.the_way_recognition.db.database import SessionLocal, engine
from src.the_way_recognition.db.models import Card, Base
from src.the_way_recognition.db.repositories.card_repository import \
    CardRepository
from src.the_way_recognition.utils.json_to_text import card_json_to_text

Base.metadata.create_all(bind=engine)


def load_embedding(path):
    arr = np.load(path)
    return arr.tobytes()


def insert_card(json_path, embedding_path, repo: CardRepository):
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
    repo.create(card)


if __name__ == "__main__":

    gt_path = 'data/gt/'
    npy_path = Path(gt_path) / "npy"
    json_path = Path(gt_path) / "json"
    with SessionLocal() as session:
        repo = CardRepository(session)
        for filename in json_path.iterdir():
            if filename.suffix == '.json':
                json_file = filename
                emb_file = npy_path / filename.with_suffix('.npy').name
                insert_card(json_file, emb_file, repo)
