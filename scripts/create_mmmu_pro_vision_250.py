"""
Create MMMU_Pro_vision_250: balanced 250-sample subset of MMMU_Pro_vision,
then compute CLIP ViT-L/14 embeddings (text=options, image=image column).

Since question is always None in the vision split, text input = formatted options only.
Combined embedding: mean(text_emb, image_emb) → L2-normalized 768-dim.

Saves:
  - Dataset:   data/MMMU_Pro/MMMU_Pro_vision_250
  - Embeddings: embeddings/MMMU_Pro_vision_250.csv
"""

import ast
import csv
import io
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_from_disk
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = "."
SRC_DATA = f"{BASE_DIR}/data/MMMU_Pro/MMMU_Pro_vision"
DST_DATA = f"{BASE_DIR}/data/MMMU_Pro/MMMU_Pro_vision_250"
DST_CSV  = f"{BASE_DIR}/embeddings/MMMU_Pro_vision_250.csv"

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
N_TOTAL  = 250
PER_SUBJ = N_TOTAL // 30          # 8
EXTRA    = N_TOTAL - PER_SUBJ * 30  # 10
SEED     = 42
BATCH_SIZE = 32


# ── Balanced sampling ────────────────────────────────────────────────────────

def balanced_sample(ds, seed=SEED):
    rng = random.Random(seed)
    table = ds.data.table
    subjects = table["subject"].to_pylist()

    from collections import defaultdict
    subj_indices = defaultdict(list)
    for i, s in enumerate(subjects):
        subj_indices[s].append(i)

    selected = []
    leftover = []
    for subj in sorted(subj_indices):
        pool = subj_indices[subj]
        rng.shuffle(pool)
        selected.extend(pool[:PER_SUBJ])
        leftover.extend(pool[PER_SUBJ:])

    rng.shuffle(leftover)
    selected.extend(leftover[:EXTRA])
    selected.sort()
    return selected


# ── CLIP helpers ─────────────────────────────────────────────────────────────

def load_clip(device):
    print(f"Loading CLIP {CLIP_MODEL_NAME} on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    print("CLIP loaded.")
    return model, processor


def format_text(options_raw) -> str:
    """Options-only text (question is None in vision split)."""
    try:
        options = ast.literal_eval(options_raw) if isinstance(options_raw, str) else options_raw
    except Exception:
        options = []
    return "\n".join(f"({chr(65+j)}) {opt}" for j, opt in enumerate(options))


def embed_texts(texts, model, processor, device):
    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        embs = model.get_text_features(**inputs).float()
        embs = F.normalize(embs, dim=-1)
    return embs.cpu().numpy()


def embed_images(pil_images, model, processor, device):
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs).float()
        embs = F.normalize(embs, dim=-1)
    return embs.cpu().numpy()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading source dataset from {SRC_DATA} ...")
    ds = load_from_disk(SRC_DATA)
    table = ds.data.table
    print(f"Source size: {len(ds)}")

    # Balanced sample
    indices = balanced_sample(ds)
    print(f"Selected {len(indices)} samples (balanced across 30 subjects)")

    # Save subset dataset
    subset = ds.select(indices)
    os.makedirs(os.path.dirname(DST_DATA), exist_ok=True)
    subset.save_to_disk(DST_DATA)
    print(f"Saved subset → {DST_DATA}")

    # Compute CLIP embeddings on the subset using Arrow table
    model, processor = load_clip(device)
    sub_table = table.take(indices)
    n = len(indices)
    rows_out = []

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = sub_table.slice(start, end - start).to_pydict()

        texts = [format_text(batch["options"][i]) for i in range(end - start)]
        text_embs = embed_texts(texts, model, processor, device)

        # Decode images from raw bytes
        pil_images = []
        for i in range(end - start):
            img_raw = batch["image"][i]
            raw = img_raw["bytes"] if isinstance(img_raw, dict) else img_raw
            pil_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

        img_embs = embed_images(pil_images, model, processor, device)

        # Mean → L2-normalize
        combined = (text_embs + img_embs) / 2.0
        combined = combined / np.linalg.norm(combined, axis=1, keepdims=True).clip(min=1e-8)

        for i in range(end - start):
            rows_out.append({"input": texts[i], "embedding": combined[i].tolist()})

        print(f"  [{end}/{n}] embedded")

    os.makedirs(os.path.dirname(DST_CSV), exist_ok=True)
    with open(DST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "embedding"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved {len(rows_out)} embeddings → {DST_CSV}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
