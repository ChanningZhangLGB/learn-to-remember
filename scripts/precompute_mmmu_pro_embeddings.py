"""
Precompute CLIP (ViT-L/14) embeddings for MMMU-Pro datasets.

For each sample:
  - Text embedding:  CLIP text encoder on (question + options), <image X> stripped
  - Image embedding: CLIP image encoder on image_1 (primary image)
  - Combined:        mean(text_emb, image_emb) → 768-dim, L2-normalized

Saves to CSV with columns: input, embedding
Compatible with LeRe precomputed_embeddings_csv loader.

Usage:
  python scripts/precompute_mmmu_pro_embeddings.py
"""

import ast
import csv
import io
import re
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_from_disk
from transformers import CLIPModel, CLIPProcessor

DATASETS = [
    (
        "./data/MMMU_Pro/MMMU_Pro_standard_4_250",
        "./embeddings/MMMU_Pro_standard_4_250.csv",
    ),
    (
        "./data/MMMU_Pro/MMMU_Pro_standard_10_250",
        "./embeddings/MMMU_Pro_standard_10_250.csv",
    ),
]

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
BATCH_SIZE = 32


def load_clip(device):
    print(f"Loading CLIP {CLIP_MODEL_NAME} on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    print("CLIP loaded.")
    return model, processor


def format_text(question: str, options_raw: str) -> str:
    """Build text input: stripped question + options."""
    q = re.sub(r"<image\s*\d+>", "", question).strip()
    try:
        options = ast.literal_eval(options_raw) if isinstance(options_raw, str) else options_raw
    except Exception:
        options = []
    opts = "\n".join(f"({chr(65+j)}) {opt}" for j, opt in enumerate(options))
    return f"{q}\nOptions:\n{opts}"


def get_primary_image(row_dict: dict):
    """Return PIL Image from image_1, or None if absent."""
    img = row_dict.get("image_1")
    if img is None:
        return None
    raw_bytes = img["bytes"] if isinstance(img, dict) else img
    try:
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
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
    """Embed a list of PIL Images. Returns (N, 768) array."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs).float()
        embs = F.normalize(embs, dim=-1)
    return embs.cpu().numpy()


def process_dataset(data_dir: str, out_csv: str, model, processor, device):
    print(f"\n=== Processing {data_dir} ===")
    ds = load_from_disk(data_dir)
    table = ds.data.table
    n = len(ds)
    print(f"  Samples: {n}")

    rows_out = []

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = table.slice(start, end - start).to_pydict()

        texts = [
            format_text(batch["question"][i], batch["options"][i])
            for i in range(end - start)
        ]

        # Text embeddings
        text_embs = embed_batch_text(texts, model, processor, device)  # (B, 768)

        # Image embeddings — embed available images, use text emb as fallback
        pil_images = [get_primary_image({k: batch[k][i] for k in batch}) for i in range(end - start)]
        has_image = [img is not None for img in pil_images]

        img_embs = np.zeros_like(text_embs)
        valid_indices = [i for i, h in enumerate(has_image) if h]
        if valid_indices:
            valid_pils = [pil_images[i] for i in valid_indices]
            valid_img_embs = embed_batch_images(valid_pils, model, processor, device)
            for idx, emb in zip(valid_indices, valid_img_embs):
                img_embs[idx] = emb

        # Combined: mean(text, image) for samples with image; text-only otherwise
        combined = np.where(
            np.array(has_image)[:, None],
            (text_embs + img_embs) / 2.0,
            text_embs,
        )
        combined = combined / np.linalg.norm(combined, axis=1, keepdims=True).clip(min=1e-8)

        for i in range(end - start):
            rows_out.append({
                "input": texts[i],
                "embedding": combined[i].tolist(),
            })

        print(f"  [{end}/{n}] done — img_present: {sum(has_image)}/{end-start}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "embedding"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  Saved {len(rows_out)} embeddings → {out_csv}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, processor = load_clip(device)

    for data_dir, out_csv in DATASETS:
        process_dataset(data_dir, out_csv, model, processor, device)

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
