import csv
import json
from pathlib import Path
import unicodedata

CSV_PATH = Path("data") / "cards.csv"
OUTPUT_DIR = Path("data") / "json"

TOTAL_CARDS = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def remove_diacritics(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def csv_row_to_json(row):
    name_no_diacritics = remove_diacritics(row["name"]).upper()
    edition_number = row["edition"][0]
    index = f"{row[' card index']}/{TOTAL_CARDS}"
    rarity = row["rarity"][0].upper()

    description = row['rules']
    if description:
        description += f"\n{row['quote']}"
    else:
        description = row['quote']
    description = f'{row["type"]}\n{description}'

    footer = f'ED{edition_number} + SK {row["Ilustruje"]} "& © 2025 The Way'

    return {
        "name": name_no_diacritics,
        "edition": edition_number,
        "index": index,
        "rarity": rarity,
        "description": description,
        "footer": footer
    }

with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= TOTAL_CARDS:
            break
        card_json = csv_row_to_json(row)
        out_path = OUTPUT_DIR / f"{i+1}.json"
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(card_json, out_f, ensure_ascii=False, indent=2)
