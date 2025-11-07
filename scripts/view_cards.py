from src.models import Card
from src.the_way_recognition.database import SessionLocal

def view_cards():
    with SessionLocal() as session:
        cards = session.query(Card).all()
        for card in cards:
            print(f"Name: {card.name}\nEdition: {card.edition}\nRarity: {card.rarity}")
            print(f"GT Text:\n{card.gt_text}")
            print()
            print("-" * 40)
            # print(card)

if __name__ == "__main__":
    view_cards()
