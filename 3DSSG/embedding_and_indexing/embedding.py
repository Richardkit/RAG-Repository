#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import SiglipVisionModel, SiglipImageProcessor


# ========= CONFIG =========
ROOT = Path("/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan")  # where all <scan_id> dirs live
OUTPUT_DIR = Path("/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan_embeddings/frame_embeddings")  # per-scan npz
TEST_TXT = Path("/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_testset.txt")

MODEL_CKPT = "google/siglip-base-patch16-224"
BATCH_SIZE = 16
# ==========================


def find_frame_jsons_for_scan(scan_dir: Path) -> List[str]:
    """Return list of frame stems (e.g., 'frame-000123') for a given <scan_id>."""
    sg_dir = scan_dir / "frame_sg"
    if not sg_dir.is_dir():
        return []
    return sorted([p.stem for p in sg_dir.glob("frame-*.json") if p.is_file()])


def guess_rgb_path_for_frame(scan_dir: Path, frame_id: int) -> Optional[Path]:
    """
    Try common 3RScan patterns to find the RGB path for a given frame id.
    We try .jpg/.png and with/without 'frame-' prefix.
    """
    seq = scan_dir / "sequence"
    fid = f"{frame_id:06d}"
    candidates = [
        seq / f"frame-{fid}.color.jpg",
        seq / f"frame-{fid}.color.png",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_model_and_processor(device: str = "cuda") -> tuple[SiglipVisionModel, SiglipImageProcessor]:
    model = SiglipVisionModel.from_pretrained(MODEL_CKPT).to(device)
    processor = SiglipImageProcessor.from_pretrained(MODEL_CKPT)
    return model, processor


def embed_images(model, processor, paths, device):
    embs_list = []
    kept_idx = []  # indices (w.r.t. `paths`) that succeeded
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[i:i+BATCH_SIZE]
            images, idx_map = [], []
            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as e:
                    print(f"❌ Cannot open image: {p} ({e})")
                    continue
                images.append(img)
                idx_map.append(i + j)  # original index in `paths`

            if not images:
                continue

            inputs = processor(images=images, do_resize=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            pooled = outputs.pooler_output.detach().cpu().numpy().astype(np.float32)  # [m,768]
            embs_list.append(pooled)
            kept_idx.extend(idx_map)

    if not embs_list:
        return np.zeros((0, 768), dtype=np.float32), []
    embs = np.concatenate(embs_list, axis=0)  # [M,768]
    return embs, kept_idx

# ========= Main =========
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🟢 Using device: {device}")

    model, processor = load_model_and_processor(device=device)

    # All scans
    scan_dirs = [p for p in ROOT.iterdir() if p.is_dir()]

    # Optional: exclude test set
    test_scan_ids = set()
    if TEST_TXT.is_file():
        with open(TEST_TXT, "r") as f:
            test_scan_ids = {line.strip() for line in f if line.strip()}

    # Accumulators for this run's NEW scans (only those we actually wrote)
    run_keys: list[str] = []
    run_vecs: list[np.ndarray] = []
    written_scans = 0

    for scan_dir in tqdm(scan_dirs, desc="Scans"):
        scan_id = scan_dir.name
        if scan_id in test_scan_ids:
            continue

        # Gather frames and RGBs
        frame_stems = find_frame_jsons_for_scan(scan_dir)
        if not frame_stems:
            print(f"⚠️  No frame_sg for {scan_id} (skipped)")
            continue

        frame_ids, rgb_paths = [], []
        for stem in frame_stems:
            try:
                frame_id = int(stem.split("-")[1])
            except Exception:
                continue
            rgb_path = guess_rgb_path_for_frame(scan_dir, frame_id)
            if rgb_path is None:
                print(f"⚠️  Missing RGB for {scan_id} frame-{frame_id:06d} (skipped)")
                continue
            frame_ids.append(frame_id)
            rgb_paths.append(rgb_path)

        if not rgb_paths:
            print(f"⚠️  No valid RGBs found for {scan_id} (skipped)")
            continue

        # Per-scan output dir
        out_dir = OUTPUT_DIR / scan_id
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "frame_rgb_embeddings.npz"
        json_path = out_dir / "index.json"

        # If per-scan already exists, skip doing it again (we only append NEW scans this run)
        if npz_path.exists() and json_path.exists():
            # the already existed embeddings will not count towards the 400 quota, nor will they be appended to the global chunk for this run
            continue

        # Compute embeddings
        embs, kept_idx = embed_images(model, processor, rgb_paths, device=device)
        if embs.shape[0] != len(rgb_paths):
            print(f"⚠️  Mismatch: got {embs.shape[0]} embeddings for {len(rgb_paths)} images in {scan_id}")

        # Keys for succeeded images
        keys = [f"frame-{frame_ids[i]:06d}" for i in kept_idx]
        rgb_paths = [rgb_paths[i] for i in kept_idx]

        # Save per-scan NPZ / index.json (overwrite semantics: always write fresh files)
        np.savez(npz_path, **{k: v for k, v in zip(keys, embs)})
        index = {
            "scan_id": scan_id,
            "count": len(keys),
            "frames": [
                {"key": k, "frame_id": int(k.split('-')[1]), "rgb_path": str(p)}
                for k, p in zip(keys, rgb_paths)
            ]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        print(f"✅ Saved {len(keys)} embeddings for {scan_id} to {npz_path}")

    print("🎉 Done.")


if __name__ == "__main__":
    main()
