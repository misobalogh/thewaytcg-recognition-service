from sqlalchemy import Column, String, LargeBinary
from src.the_way_recognition.db.database import Base

class Card(Base):
    __tablename__ = 'cards'
    name = Column(String, nullable=False, primary_key=True)
    edition = Column(String)
    rarity = Column(String)
    gt_text = Column(String)
    gt_embedding = Column(LargeBinary)

    def __repr__(self):
        return f"<Card(name='{self.name}', edition='{self.edition}', rarity='{self.rarity}')>"
