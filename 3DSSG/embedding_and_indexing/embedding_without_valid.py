#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import pickle
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import SiglipVisionModel, SiglipImageProcessor


# ========= CONFIG =========
ROOT = Path("/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan")  # where all <scan_id> dirs live
OUTPUT_DIR = Path("/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_patch_embeddings_without_validset_siglip2/frame_embeddings")  # per-scan npz
VAL_DIR  = Path("/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/embedding_and_indexing/3RScan_validation_set.txt")  # optional: val set to exclude
MODEL_CKPT = "google/siglip2-base-patch16-224"
BATCH_SIZE = 16
GRID_ROWS = 4
GRID_COLS = 6
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


def load_model_and_processor(device="cuda"):
    model = SiglipVisionModel.from_pretrained(MODEL_CKPT).to(device).eval()
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

            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            # 1) one forward -> last_hidden_state

            out = model(pixel_values=pixel_values)
            tok = out.last_hidden_state  # [B, T, D]

            H = W = int(model.config.image_size // model.config.patch_size)  # usually 14
            B, T, D = tok.shape
            HW = H * W

            # Some versions of SigLIP2 have no CLS token, others do; determine by sequence length
            if T == HW:
                tokens = tok                      # no CLS
            elif T == HW + 1:
                tokens = tok[:, 1:, :]            # has CLS -> drop
            else:
                # fallback: don't drop anything and infer square grid if possible
                tokens = tok
                B, T, D = tokens.shape
                s = int(T ** 0.5)
                if s * s == T:
                    H = W = s
                    HW = T
                elif (T - 1) > 0 and int((T - 1) ** 0.5) ** 2 == (T - 1):
                    # maybe CLS + square
                    tokens = tokens[:, 1:, :]
                    T = tokens.shape[1]
                    s = int(T ** 0.5)
                    H = W = s
                    HW = T
                else:
                    raise RuntimeError(f"Cannot infer token grid: T={T}, expected {HW} or {HW+1}")

            # now tokens must be [B, HW, D]
            feat = tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]


            pooled = F.adaptive_avg_pool2d(feat, output_size=(GRID_ROWS, GRID_COLS))  # [B, D, R, C]
            pooled = pooled.permute(0, 2, 3, 1).reshape(B, GRID_ROWS*GRID_COLS, D)    # [B, R*C, D]

            # normalize for cosine
            pooled = torch.nn.functional.normalize(pooled, dim=-1)

            pooled = pooled.cpu().numpy().astype(np.float32)  # [B, 144, D]
            embs_list.append(pooled)
            kept_idx.extend(idx_map)

    if not embs_list:
        embed_dim = model.config.hidden_size
        return np.zeros((0, embed_dim), dtype=np.float32), []
    embs = np.concatenate(embs_list, axis=0)  # [M, 144, D]
    return embs, kept_idx

# ========= Main =========
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🟢 Using device: {device}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    assert 0 <= args.shard_id < args.num_shards

    with open(VAL_DIR, "r") as f:
        val_scan_ids = {line.strip() for line in f if line.strip()}

    model, processor = load_model_and_processor(device=device)

    # All scans
    scan_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda x: x.name)
    scan_dirs = scan_dirs[args.shard_id::args.num_shards]
    print(f"[Shard] shard_id={args.shard_id}/{args.num_shards}, scans={len(scan_dirs)}")


    # Optional: exclude test set
    # test_scan_ids = set()
    # if TEST_TXT.is_file():
    #     with open(TEST_TXT, "r") as f:
    #         test_scan_ids = {line.strip() for line in f if line.strip()}

    # Accumulators for this run's NEW scans (only those we actually wrote)
    run_keys: list[str] = []
    run_vecs: list[np.ndarray] = []
    written_scans = 0

    for scan_dir in tqdm(sorted(scan_dirs), desc="Scans"):

        if scan_dir.name in val_scan_ids:
            print(f"⚠️  Skipping validation scan {scan_dir.name}")
            continue
        
        scan_id = scan_dir.name
        # if scan_id in test_scan_ids:
        #     continue

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
        npy_path  = out_dir / "frame_patch_embeddings.npy"
        pkl_path  = out_dir / "patch_keys.pkl"
        json_path = out_dir / "patch_index.json"

        if npy_path.exists() and pkl_path.exists() and json_path.exists():
            continue

        # Compute embeddings
        embs, kept_idx = embed_images(model, processor, rgb_paths, device=device)
        if embs.shape[0] != len(kept_idx):
            print(f"⚠️  Mismatch: got {embs.shape[0]} embeddings for {len(kept_idx)} kept images in {scan_id}")

        # Keys for succeeded images
        keys = [f"frame-{frame_ids[i]:06d}" for i in kept_idx]
        rgb_paths = [rgb_paths[i] for i in kept_idx]

        patch_keys = []
        patches_meta = []

        # embs: [num_frames, 144, D]
        # reshape it to [num_frames*144, D], order consistent with the key loop below
        X = embs.reshape(-1, embs.shape[-1])  # float32

        # Build key list (length = num_frames*144)
        for frame_key in keys:
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    pk = f"{frame_key}|r{r}c{c}"
                    patch_keys.append(pk)
                    patches_meta.append({
                        "key": pk,
                        "frame_key": frame_key,
                        "frame_id": int(frame_key.split("-")[1]),
                        "row": r,
                        "col": c
                    })

        # Safety check: ensure the number of keys matches the number of matrix rows
        if X.shape[0] != len(patch_keys):
            raise RuntimeError(f"Patch count mismatch in {scan_id}: X={X.shape[0]} vs keys={len(patch_keys)}")

        # Save: float16 is recommended (fast, small file); convert to float32 when building the index
        X_to_save = X.astype(np.float16, copy=False)
        np.save(npy_path, X_to_save)

        with open(pkl_path, "wb") as f:
            pickle.dump(patch_keys, f)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"scan_id": scan_id, "patches": patches_meta}, f, indent=2)

        print(f"✅ Saved {len(keys)} frames / {len(patch_keys)} patches for {scan_id} to {npy_path}")


    print("🎉 Done.")


if __name__ == "__main__":
    main()
