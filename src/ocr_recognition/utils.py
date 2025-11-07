import json


def card_json_to_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        card = json.load(f)
    name = card.get("name", "")
    description = card.get("description", "")
    index = card.get("index", "")
    rarity = card.get("rarity", "")
    footer = card.get("footer", "")
    text_block = f"{name}\n\n{description}\n\n{index} {rarity}\n{footer}"
    return text_block
