#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_window_level_faiss_index_tmp304_weightedpool.py

临时 window-level embedding + FAISS index 构建脚本（仅对已完成 experience 的 scans）：
- 不再自行 plan_windows：直接读取每个 scan 的 {scan_id}_windows.json
- 支持 window 帧数 != WIN_SIZE（考虑 stride + 边界），允许可变长度 window
- 遍历 EXP_ROOT 下已处理 scans，自动跳过未完成 scan（305 中那 1 个）
- 每个 window 用其 RGB 帧的 SigLIP embedding 做 softmax 加权池化得到 window embedding
- 保存临时 FAISS index & keys

输出：
- OUT_DIR/window_index.faiss
- OUT_DIR/window_keys.pkl
- OUT_DIR/window_embeds.npy (可选)
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import faiss
from transformers import SiglipProcessor, SiglipModel


# =========================
# Config (edit here)
# =========================

EXP_ROOT = Path("/hkfs/work/workspace/scratch/rg5174-ragspace2/data/RAG_Rexplore_Qwen3")

# 你需要保证能找到：RGB_ROOT/scan_id/sequence/frame-xxxxxx.color.*
RGB_ROOT = Path("/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan")  # TODO: 改成你真实路径
SEQ_DIRNAME = "sequence"
RGB_GLOB = "frame-*.color.*"

# 仅用于 sanity print（不做强约束）
WIN_SIZE = 6
STRIDE = 4

# 为了避免冗余/算力浪费：每个 window 最多采样多少帧来做 embedding
MAX_FRAMES_PER_WINDOW = 8

# softmax 加权池化的 sharpness 系数
ALPHA = 12

SIGLIP_NAME = "google/siglip-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

OUT_DIR = Path("/hkfs/work/workspace/scratch/rg5174-ragspace2/data/tmp_window_index_304")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_EMBEDS_NPY = True


# =========================
# Utils
# =========================

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.maximum(np.sum(ex), 1e-12)


def uniform_subsample(paths: List[Path], max_n: int) -> List[Path]:
    """均匀采样（保留首尾），避免相邻帧冗余。"""
    n = len(paths)
    if n <= max_n:
        return paths
    idx = np.linspace(0, n - 1, num=max_n)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, n - 1)
    # 去重并保持顺序
    seen = set()
    out = []
    for i in idx.tolist():
        if i not in seen:
            out.append(paths[i])
            seen.add(i)
    return out


def load_all_rgb_paths(scan_id: str) -> List[Path]:
    seq_dir = RGB_ROOT / scan_id / SEQ_DIRNAME
    if not seq_dir.exists():
        raise FileNotFoundError(f"RGB sequence dir not found: {seq_dir}")
    rgb_paths = sorted(seq_dir.glob(RGB_GLOB))
    if len(rgb_paths) == 0:
        raise FileNotFoundError(f"No RGB frames found with pattern {RGB_GLOB} in {seq_dir}")
    return rgb_paths


def read_windows_json(scan_dir: Path, scan_id: str) -> List[Tuple[int, int]]:
    """
    读取 {scan_id}_windows.json，返回 [(start_idx, end_idx), ...]（end_idx 视为半开区间）
    文件里字段名按你现有实现：start_idx / end_idx
    """
    win_path = scan_dir / f"{scan_id}_windows.json"
    if not win_path.exists():
        raise FileNotFoundError(f"Missing windows json: {win_path}")
    with open(win_path, "r") as f:
        win_data = json.load(f)
    windows = [(int(w["start"]), int(w["end"])) for w in win_data]
    return windows


def exp_window_json_path(scan_dir: Path, win_idx: int) -> Path:
    return scan_dir / "exp" / f"window_{win_idx}_abstraction.json"


def is_scan_completed(scan_dir: Path, scan_id: str, windows: List[Tuple[int, int]]) -> bool:
    """
    完成判据：
    - 有 exp/ 目录
    - 对 windows.json 里每个 win_idx，都存在对应 abstraction json
    """
    exp_dir = scan_dir / "exp"
    if not exp_dir.exists():
        return False
    for i in range(len(windows)):
        if not exp_window_json_path(scan_dir, i).exists():
            return False
    return True


@torch.inference_mode()
def siglip_embed_images(model: SiglipModel, processor: SiglipProcessor, pil_images: List[Image.Image]) -> np.ndarray:
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    feats = model.get_image_features(**inputs)
    return feats.to(torch.float32).cpu().numpy()  # (N, D)


def embed_window_softmax_pool(
    model: SiglipModel,
    processor: SiglipProcessor,
    rgb_paths: List[Path],
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    softmax 加权池化（无训练）：
    1) 每帧 embedding: e_i
    2) 粗中心 mu = mean(e_i)
    3) 分数 s_i = cos(e_i, mu)
    4) 权重 w_i = softmax(alpha * s_i)
    5) window emb = sum_i w_i * e_i
    6) L2 normalize
    """
    # 可选：采样减少冗余
    rgb_paths = uniform_subsample(rgb_paths, MAX_FRAMES_PER_WINDOW)

    pil_imgs = [Image.open(str(p)).convert("RGB") for p in rgb_paths]
    feats = siglip_embed_images(model, processor, pil_imgs)  # (N, D)

    # 归一化后，dot 即 cosine
    feats = l2norm(feats.astype(np.float32))  # (N, D)
    mu = feats.mean(axis=0, keepdims=True).astype(np.float32)  # (1, D)
    mu = l2norm(mu)  # (1, D)

    scores = (feats @ mu.T).reshape(-1)  # (N,)
    w = softmax_np(alpha * scores).astype(np.float32)  # (N,)

    win = (w[:, None] * feats).sum(axis=0, keepdims=True)  # (1, D)
    win = l2norm(win).astype(np.float32)
    return win[0]


def main():
    print(f"[Info] DEVICE={DEVICE} DTYPE={DTYPE}")
    print(f"[Info] EXP_ROOT={EXP_ROOT}")
    print(f"[Info] RGB_ROOT={RGB_ROOT}")
    print(f"[Info] WIN_SIZE={WIN_SIZE} STRIDE={STRIDE} (for reference only)")
    print(f"[Info] MAX_FRAMES_PER_WINDOW={MAX_FRAMES_PER_WINDOW}")
    print(f"[Info] ALPHA={ALPHA}")
    print(f"[Info] OUT_DIR={OUT_DIR}")

    # 1) load SigLIP
    print(f"[Info] Loading SigLIP: {SIGLIP_NAME}")
    processor = SiglipProcessor.from_pretrained(SIGLIP_NAME)
    model = SiglipModel.from_pretrained(SIGLIP_NAME).to(DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model = model.to(DTYPE)

    # 2) iterate scans in EXP_ROOT, keep only completed ones
    scan_dirs = [p for p in sorted(EXP_ROOT.iterdir()) if p.is_dir()]
    print(f"[Info] Found {len(scan_dirs)} scan folders under EXP_ROOT (expect 305 with one incomplete).")

    keys: List[Dict[str, Any]] = []
    embeds: List[np.ndarray] = []

    rgb_cache: Dict[str, List[Path]] = {}

    num_skipped_incomplete = 0
    num_skipped_rgb_missing = 0
    num_used_scans = 0

    for scan_dir in tqdm(scan_dirs, desc="Scans"):
        scan_id = scan_dir.name

        # must have windows json
        try:
            windows = read_windows_json(scan_dir, scan_id)
        except Exception:
            num_skipped_incomplete += 1
            continue

        # must be "completed" (skip the one unfinished scan)
        if not is_scan_completed(scan_dir, scan_id, windows):
            print(f"[Warn] Skipping incomplete scan: {scan_id}")
            num_skipped_incomplete += 1
            continue

        # load rgb list
        try:
            if scan_id not in rgb_cache:
                rgb_cache[scan_id] = load_all_rgb_paths(scan_id)
            all_rgb = rgb_cache[scan_id]
        except Exception:
            num_skipped_rgb_missing += 1
            continue

        # embed each window
        for win_idx, (s, e) in enumerate(windows):
            # safety clamp to available frames
            s = max(0, min(int(s), len(all_rgb)))
            e = max(0, min(int(e), len(all_rgb)))
            if e <= s:
                continue

            rgb_paths = all_rgb[s:e]

            exp_json = exp_window_json_path(scan_dir, win_idx)
            if not exp_json.exists():
                continue

            try:
                win_emb = embed_window_softmax_pool(model, processor, rgb_paths, alpha=ALPHA)
            except Exception as ex:
                print(f"[Warn] embed failed scan={scan_id} win={win_idx}: {type(ex).__name__}: {ex}")
                continue

            embeds.append(win_emb)
            keys.append({
                "scan_id": scan_id,
                "win_idx": win_idx,
                "exp_json_path": str(exp_json),
                "rgb_paths": [str(p) for p in rgb_paths],
                "start_idx": int(s),
                "end_idx": int(e),
                "n_frames_raw": int(len(rgb_paths)),
                "n_frames_used": int(min(len(rgb_paths), MAX_FRAMES_PER_WINDOW)),
                "pool": "softmax_weighted",
                "alpha": float(ALPHA),
            })

        num_used_scans += 1

    if len(embeds) == 0:
        raise RuntimeError("No embeddings generated. Check RGB_ROOT / patterns / completed scans.")

    X = np.stack(embeds, axis=0).astype(np.float32)
    N, D = X.shape
    print(f"[Info] Generated window embeddings: N={N}, D={D}")
    print(f"[Info] Used scans={num_used_scans}, skipped_incomplete={num_skipped_incomplete}, skipped_rgb_missing={num_skipped_rgb_missing}")

    # 3) build FAISS (cosine) : IP with L2-normalized vectors
    index = faiss.IndexFlatIP(D)
    index.add(X)
    print(f"[Info] FAISS index ntotal={index.ntotal}")

    # 4) save
    index_path = OUT_DIR / "window_index.faiss"
    keys_path = OUT_DIR / "window_keys.pkl"
    faiss.write_index(index, str(index_path))
    with open(keys_path, "wb") as f:
        pickle.dump(keys, f)

    print(f"[OK] Saved index: {index_path}")
    print(f"[OK] Saved keys : {keys_path}")

    if SAVE_EMBEDS_NPY:
        emb_path = OUT_DIR / "window_embeds.npy"
        np.save(str(emb_path), X)
        print(f"[OK] Saved embeds: {emb_path}")

    # quick self-check
    rng = np.random.default_rng(0)
    test_ids = rng.choice(N, size=min(3, N), replace=False)
    Dists, Ids = index.search(X[test_ids], k=min(5, N))
    for qpos, qi in enumerate(test_ids):
        print(f"[Check] q={qi} -> hits={Ids[qpos].tolist()} sims={Dists[qpos].tolist()}")

    print("[Done]")


if __name__ == "__main__":
    main()
