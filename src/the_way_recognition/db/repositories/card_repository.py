from typing import List, Optional
from sqlalchemy.orm import Session
from src.the_way_recognition.db.models import Card


class CardRepository:

    def __init__(self, session: Session):
        self.session = session

    def get_all(self) -> List[Card]:
        return self.session.query(Card).all()

    def get_by_id(self, card_id: int) -> Optional[Card]:
        return self.session.query(Card).filter(Card.id == card_id).first()

    def get_by_name(self, name: str) -> Optional[Card]:
        return self.session.query(Card).filter(Card.name == name).first()

    def create(self, card: Card) -> Card:
        self.session.add(card)
        self.session.commit()
        self.session.refresh(card)
        return card

    def update(self, card: Card) -> Card:
        self.session.commit()
        self.session.refresh(card)
        return card

    def delete(self, card_id: int) -> bool:
        card = self.get_by_id(card_id)
        if card:
            self.session.delete(card)
            self.session.commit()
            return True
        return False
