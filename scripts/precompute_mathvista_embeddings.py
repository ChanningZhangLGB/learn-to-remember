"""
Precompute CLIP (ViT-L/14) embeddings for MathVista_testmini_250.

For each sample:
  - Text:    CLIP text encoder on question (+ options if multi_choice)
  - Image:   CLIP image encoder on decoded_image
  - Combined: normalize(mean(text_emb, image_emb)) → 768-dim

Options are included in text only when choices is non-null (multi_choice samples).
Free_form samples embed question text only (no choices).

Saves to CSV with columns: input, embedding
Compatible with LeRe precomputed_embeddings_csv loader.

Usage:
  python scripts/precompute_mathvista_embeddings.py
"""

import csv
import io
import os
import re
import sys

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DATASET_ARROW = "./data/MathVista_testmini_250/data-00000-of-00001.arrow"
OUT_CSV       = "./embeddings/MathVista_testmini_250.csv"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
BATCH_SIZE = 16


def load_clip(device):
    print(f"Loading CLIP {CLIP_MODEL_NAME} on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    print("CLIP loaded.")
    return model, processor


def format_text(question: str, choices) -> str:
    """Question text + options if choices present; question only for free_form."""
    q = re.sub(r"<image\s*\d+>", "", question or "").strip()
    if choices:
        opts = "\n".join(f"({chr(65+j)}) {opt}" for j, opt in enumerate(choices))
        return f"{q}\nOptions:\n{opts}"
    return q


def get_pil_image(decoded_image):
    """Extract PIL Image from decoded_image field (dict with 'bytes' or raw bytes)."""
    if decoded_image is None:
        return None
    raw = decoded_image.get("bytes") if isinstance(decoded_image, dict) else decoded_image
    if not raw:
        return None
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def embed_batch_text(texts, model, processor, device):
    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        embs = model.get_text_features(**inputs).float()
        embs = F.normalize(embs, dim=-1)
    return embs.cpu().numpy()


def embed_batch_images(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs).float()
        embs = F.normalize(embs, dim=-1)
    return embs.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, processor = load_clip(device)

    reader = pa.ipc.open_stream(DATASET_ARROW)
    table = reader.read_all()
    n = table.num_rows
    print(f"Loaded {n} samples from {DATASET_ARROW}")

    rows_out = []

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = table.slice(start, end - start).to_pydict()

        texts = [
            format_text(batch["question"][i], batch["choices"][i])
            for i in range(end - start)
        ]

        # Text embeddings
        text_embs = embed_batch_text(texts, model, processor, device)

        # Image embeddings
        pil_images = [get_pil_image(batch["decoded_image"][i]) for i in range(end - start)]
        has_image  = [img is not None for img in pil_images]

        img_embs = np.zeros_like(text_embs)
        valid_indices = [i for i, h in enumerate(has_image) if h]
        if valid_indices:
            valid_pils    = [pil_images[i] for i in valid_indices]
            valid_img_embs = embed_batch_images(valid_pils, model, processor, device)
            for idx, emb in zip(valid_indices, valid_img_embs):
                img_embs[idx] = emb

        # Combined: normalize(mean(text, image)) if image present; text only otherwise
        combined = np.where(
            np.array(has_image)[:, None],
            (text_embs + img_embs) / 2.0,
            text_embs,
        )
        combined = combined / np.linalg.norm(combined, axis=1, keepdims=True).clip(min=1e-8)

        for i in range(end - start):
            rows_out.append({
                "input":     texts[i],
                "embedding": combined[i].tolist(),
            })

        mc = sum(1 for c in batch["choices"] if c)
        print(f"  [{end}/{n}] done — multi_choice={mc}/{end-start}, img_present={sum(has_image)}/{end-start}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "embedding"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSaved {len(rows_out)} embeddings → {OUT_CSV}")


if __name__ == "__main__":
    main()
