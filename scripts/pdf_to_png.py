import os
from pathlib import Path

import tqdm
from pdf2image import convert_from_path

PDF_DIR = Path(__file__).resolve().parent.parent / 'data' / 'pdf'
PNG_DIR = Path(__file__).resolve().parent.parent / 'data' / 'gt' / 'png'

PNG_DIR.mkdir(parents=True, exist_ok=True)

for filename in tqdm.tqdm(os.listdir(PDF_DIR)):
    if filename.lower().endswith('.pdf'):
        pdf_path = PDF_DIR / filename
        images = convert_from_path(pdf_path)
        base_name = pdf_path.stem
        for i, image in enumerate(images):
            png_filename = f"{base_name}.png"
            png_path = PNG_DIR / png_filename
            image.save(png_path, 'PNG')
