#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用 SigLIP2 给 3RScan 的 frame-level scene graph 生成“note text”embedding（文本侧），并按 scan 保存。

Input:
  /hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan/{scan_id}/frame_sg/frame-XXXXXX.json

Output:
  /hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_fram_sg_embed_note_guided_wo_validset/frame_embeddings/{scan_id}/
    - frame_text_embeds.npz   (embeddings + frame_ids + json_paths)
    - frame_note_keys.jsonl   (mapping: scan_id, frame_id, json_path, note_text)

要点：
- note_text 仅由 frame_sg 里的 object.name 构造： "Objects: floor, wall(x6), bed, pillow(x5), ..."
- 使用 SigLIP2 的 text encoder：model.get_text_features(...)
- 自动按模型 config 决定 max token（防止你之前的 max_position_embeddings 报错）
"""

import os
import re
import json
import glob
import hashlib
from typing import List, Dict, Any
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, SiglipModel


# =========================
# Config (只改这里即可)
# =========================
ROOT_3RSCAN = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan"
FRAME_SG_SUBDIR = "frame_sg"

OUT_ROOT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_fram_sg_embed_note_guided_wo_validset/frame_embeddings"

# validation scans（存在则排除；不存在就全跑）
VAL_LIST_PATH = "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"

# 你说用 SigLIP2：这里用常见的 HF 名称（按你环境已有的为准）
MODEL_NAME = "google/siglip2-base-patch16-224"

BATCH_SIZE = 128

# note-text format
USE_COUNTS = True
MAX_OBJECTS_IN_TEXT = 60  # 先粗限，tokenizer 还会再 truncation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


# =========================
# IO helpers
# =========================
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def read_lines(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s:
                out.append(s)
    return out

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def atomic_save_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # tmp 必须也以 .npz 结尾，否则 numpy 会自动补 .npz 导致找不到
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_write_jsonl(path: str, lines: List[Dict[str, Any]]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# =========================
# Note text builder
# =========================
def normalize_obj_name(s: str) -> str:
    return (s or "").strip().lower()

def build_note_text_from_frame_sg(frame_sg: Dict[str, Any]) -> str:
    objs = frame_sg.get("objects", []) or []
    names = []
    for o in objs:
        nm = normalize_obj_name(o.get("name", ""))
        if nm:
            names.append(nm)

    if not names:
        return "Objects: none."

    cnt = Counter(names)
    # sort: 高频优先，再字母
    items = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))

    parts = []
    for name, c in items[:MAX_OBJECTS_IN_TEXT]:
        if USE_COUNTS and c > 1:
            parts.append(f"{name}(x{c})")
        else:
            parts.append(name)

    return "Objects: " + ", ".join(parts) + "."


# =========================
# Model max token helper (SigLIP2 safe)
# =========================
def infer_max_text_tokens(processor, model, fallback: int = 64) -> int:
    # 1) try model.config.text_config.max_position_embeddings
    mx = None
    cfg = getattr(model, "config", None)
    if cfg is not None:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            mx = getattr(text_cfg, "max_position_embeddings", None)

    # 2) try tokenizer.model_max_length
    tok_mx = None
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok_mx = getattr(tok, "model_max_length", None)
        # 有些 tokenizer 会给超大值（1e30），这里过滤一下
        try:
            if isinstance(tok_mx, int) and tok_mx > 0 and tok_mx < 100000:
                pass
            else:
                tok_mx = None
        except Exception:
            tok_mx = None

    cands = []
    if isinstance(mx, int) and mx > 0:
        cands.append(mx)
    if isinstance(tok_mx, int) and tok_mx > 0:
        cands.append(tok_mx)

    if cands:
        # 取更保守的最小值，避免炸
        return int(min(cands))

    return int(fallback)


# =========================
# SigLIP2 text embedding
# =========================
@torch.no_grad()
def encode_text_batch(tokenizer, model, texts, max_tokens):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    feats = model.get_text_features(**inputs)  # 关键：只需要 input_ids/attention_mask
    feats = feats.to(torch.float32)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
    return feats.cpu().numpy().astype(np.float32)

# =========================
# Per-scan processing
# =========================
def list_scan_ids(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    scan_ids = []
    for d in os.listdir(root):
        if UUID_RE.match(d) and os.path.isdir(os.path.join(root, d)):
            scan_ids.append(d)
    return sorted(scan_ids)

def list_frame_jsons(scan_id: str) -> List[str]:
    frame_dir = os.path.join(ROOT_3RSCAN, scan_id, FRAME_SG_SUBDIR)
    if not os.path.isdir(frame_dir):
        return []
    files = glob.glob(os.path.join(frame_dir, "frame-*.json"))

    def _fid(p):
        m = re.search(r"frame-(\d+)\.json$", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return sorted(files, key=_fid)

def parse_frame_id(path: str) -> int:
    m = re.search(r"frame-(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def process_one_scan(processor, model, scan_id: str, max_tokens: int):
    frame_jsons = list_frame_jsons(scan_id)
    if not frame_jsons:
        return

    out_dir = os.path.join(OUT_ROOT, scan_id)
    safe_mkdir(out_dir)

    out_npz = os.path.join(out_dir, "frame_text_embeds.npz")
    out_jsonl = os.path.join(out_dir, "frame_note_keys.jsonl")

    # resume：已有则跳过
    if os.path.exists(out_npz) and os.path.exists(out_jsonl):
        print(f"[Skip] {scan_id} already done: {out_dir}")
        return

    frame_ids: List[int] = []
    json_paths: List[str] = []
    note_texts: List[str] = []
    bad = 0

    for jp in frame_jsons:
        try:
            sg = load_json(jp)
            txt = build_note_text_from_frame_sg(sg)
            frame_ids.append(parse_frame_id(jp))
            json_paths.append(jp)
            note_texts.append(txt)
        except Exception:
            bad += 1

    if not note_texts:
        print(f"[Warn] {scan_id}: no valid frames. bad={bad}")
        return

    print(f"[Scan] {scan_id}: frames={len(note_texts)} bad={bad}")

    # encode
    all_embs = []
    for i in range(0, len(note_texts), BATCH_SIZE):
        batch = note_texts[i:i + BATCH_SIZE]
        em = encode_text_batch(processor, model, batch, max_tokens=max_tokens)
        all_embs.append(em)
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  [Embed] {scan_id}: {min(i+BATCH_SIZE, len(note_texts))}/{len(note_texts)}")

    embs = np.concatenate(all_embs, axis=0)
    assert embs.shape[0] == len(note_texts)

    # save npz
    atomic_save_npz(
        out_npz,
        embeddings=embs.astype(np.float32),
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        json_paths=np.asarray(json_paths, dtype=object),
    )

    # save mapping jsonl
    lines = []
    for fid, jp, txt in zip(frame_ids, json_paths, note_texts):
        lines.append({
            "scan_id": scan_id,
            "frame_id": int(fid),
            "json_path": jp,
            "note_text": txt,
        })
    atomic_write_jsonl(out_jsonl, lines)

    print(f"[OK] saved: {out_npz}")
    print(f"[OK] saved: {out_jsonl}")


def main():
    safe_mkdir(OUT_ROOT)

    # exclude validation scans if list exists
    val_scans = set(read_lines(VAL_LIST_PATH))
    if val_scans:
        print(f"[Info] loaded val scans to exclude: {len(val_scans)} from {VAL_LIST_PATH}")
    else:
        print(f"[Info] val list not found/empty, will process all scans: {VAL_LIST_PATH}")

    # load SigLIP2
    print(f"[Load] model={MODEL_NAME} device={DEVICE} dtype={DTYPE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = SiglipModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device=DEVICE, dtype=DTYPE)

    max_tokens = infer_max_text_tokens(tokenizer, model, fallback=64)
    print(f"[Info] inferred max_tokens={max_tokens} (for truncation)")

    scan_ids = list_scan_ids(ROOT_3RSCAN)

    n = 0
    for sid in scan_ids:
        if sid in val_scans:
            continue
        frame_dir = os.path.join(ROOT_3RSCAN, sid, FRAME_SG_SUBDIR)
        if not os.path.isdir(frame_dir):
            continue
        n += 1
        process_one_scan(tokenizer, model, sid, max_tokens=max_tokens)

    print(f"[Done] processed scans (excluding val): {n}")


if __name__ == "__main__":
    main()
