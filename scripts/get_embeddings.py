from pathlib import Path

import clip
import numpy as np
import torch
import tqdm
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)


embeddings = []
filenames = []
img_path = "data/gt/png"
for img_file in tqdm.tqdm(Path(img_path).iterdir()):
    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        img = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(img)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embeddings.append(embedding.cpu().numpy().flatten())
            filenames.append(img_file)


save_dir = "data/gt/npy"
Path(save_dir).mkdir(parents=True, exist_ok=True)

print("Saving embeddings...")
for embedding, filename in zip(embeddings, filenames):
    card_id = filename.stem
    np.save(Path(save_dir) / f"{card_id}.npy", embedding)
