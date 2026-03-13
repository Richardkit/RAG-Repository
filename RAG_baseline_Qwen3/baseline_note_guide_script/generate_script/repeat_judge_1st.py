"""
Use qwen3-vl-8b to judge if the given images is a repeat one compared to image buffer,
image buffer contains images that are already kept during 2nd qwen inference with scene graph prediction.
"""

# RAG_whole_scan_sg.py
# End-to-end incremental local Scene Graph (no geometry) with resumable checkpoints.
# - Nodes: id (global), name, description, seen_in_frames
# - Edges: src, tgt, c_edge
# - Sliding window over RGB frames; resume from last completed window if interrupted.

import os
from pathlib import Path
from PIL import Image

job_id = os.environ.get("SLURM_JOB_ID", "manual")
tmpdir = Path(f"/scratch/{os.environ.get('USER', 'user')}/tmp_{job_id}")
tmpdir.mkdir(parents=True, exist_ok=True, mode=0o700)
os.environ["TMPDIR"] = str(tmpdir)
os.environ["SLURM_TMPDIR"] = str(tmpdir)
os.environ.pop("TMP", None)
os.environ.pop("TEMP", None)
os.environ.pop("XDG_RUNTIME_DIR", None)

# Pin torch compile caches to the per-job tmpdir to avoid stale paths
torch_cache = tmpdir / "torchinductor_cache"
torch_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(torch_cache)
os.environ["TORCH_COMPILE_CACHE_DIR"] = str(torch_cache)


import re
import json, re
import glob
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import asdict
import torch
# from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoModel
from qwen_vl_utils import process_vision_info
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available

import re as _re, json as _json

# =========================
# Config
# =========================
# MODEL_NAME = "Qwen/Qwen3-VL-32B"  # "Qwen/Qwen3-VL-8B"
# DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 6
STRIDE      = 4  # overlap = WINDOW_SIZE - STRIDE = 2

# IF_vLLM = False  # whether to use vllm for inference

# ---- Global LLM cache for vLLM ----
# _GLOBAL_VLLM = None

FRAME_RE = re.compile(r"frame-(\d+)\.color\.(?:jpg|jpeg|png)$", re.IGNORECASE)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# =========================
# Text helpers
# =========================
def normalize_text(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", (s or "").strip().lower())

def jaccard_token_sim(a: str, b: str) -> float:
    import re as _re
    A = set(_re.findall(r"[a-z0-9]+", normalize_text(a)))
    B = set(_re.findall(r"[a-z0-9]+", normalize_text(b)))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / max(1, union)

# =========================
# Pipeline (resumable)
# =========================
def prepare_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

# =========================
# Similarity computation
# =========================

def find_indices_by_substrings(tokens, frames):
    """
    get indices of tokens whose paths contain any of the given frames
    """
    out = []
    for f in frames:
        hit = [i for i, p in enumerate(tokens) if f in p]
        if not hit:
            print(f"[WARN] cannot find any path contains '{f}'")
            out.append(None)
        else:
            out.append(hit[0])
    return out

def late_interaction_sim_sym(Vi, Vj):
    """
    Vi, Vj: (T,D), assumed L2-normalized per token
    returns symmetric late interaction similarity in [-1,1]
    sim = 0.5*( mean_i max_j dot(Vi_i, Vj_j) + mean_j max_i dot(Vi_i, Vj_j) )
    """
    # token-token similarity matrix
    S = Vi @ Vj.T  # (Ti,Tj)
    s_i2j = S.max(axis=1).mean()
    s_j2i = S.max(axis=0).mean()
    return 0.5 * (s_i2j + s_j2i)

def l2norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / n

def cheap_image_proxy(tokens_3d, mode="mean"):
    """
    tokens_3d: (N,T,D) fp16/32
    return: (N,D) float32 normalized
    """
    X = tokens_3d.astype(np.float32, copy=False)
    if mode == "mean":
        v = X.mean(axis=1)           # (N,D)
    elif mode == "max":
        v = X.max(axis=1)
    else:
        raise ValueError(f"unknown CHEAP_PROXY={mode}")
    v = l2norm_rows(v)
    return v

def topM_by_cosine(proxy, M = 30):
    """
    proxy: (N,D) row-normalized
    returns indices: (N,M), sims: (N,M)
    """
    S = proxy @ proxy.T
    np.fill_diagonal(S, -1.0)
    M = min(M, S.shape[0]-1)
    idx = np.argpartition(-S, kth=M-1, axis=1)[:, :M]
    sims = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-sims, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    sims = np.take_along_axis(sims, order, axis=1)

    return idx, sims


def compute_sim_for_window(scan_id, frames: List[str], image_buffer: List[str], CHEAP_PROXY="mean", SIM_THRESH=0.45, topk= 5):

    # topk is the number of most similar images to return for each frame in frames
    sig_dir = os.path.join("/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan", scan_id, "siglip2_val")
    tokens_path = os.path.join(sig_dir, "patch_tokens_fp16.npy")
    paths_file_path = os.path.join(sig_dir, "paths.json")

    # open tokens file
    tokens = np.load(tokens_path, mmap_mode="r")  # (N,T,D)
    paths = json.load(open(paths_file_path, "r"))  # list of N paths
    N, T, D = tokens.shape

    tgt_idxs = find_indices_by_substrings(paths, frames)
    image_buffer = [os.path.basename(p) for p in image_buffer] # save absolute paths of finished query images
    buf_idxs = find_indices_by_substrings(paths, image_buffer)

    # pre-load tokens for targets into RAM (small)
    V = {}
    for i in tgt_idxs:
        if i is None:
            continue
        Vi = np.array(tokens[i], dtype=np.float32)  # (T,D)
        Vi = Vi / (np.linalg.norm(Vi, axis=1, keepdims=True) + 1e-8)  # L2-normalize
        V[i] = Vi

    proxy = cheap_image_proxy(tokens, mode=CHEAP_PROXY)  # (N,D)
    cand_idx, cand_sim = topM_by_cosine(proxy)
    out = {}
    
    for i in tgt_idxs:

        if i is None:
            continue

        Vi = np.array(tokens[i], dtype=np.float32)
        Vi = Vi / (np.linalg.norm(Vi, axis=1, keepdims=True) + 1e-8)
        scores = []

        for j, sim in zip(cand_idx[i],cand_sim[i]):

            if j in buf_idxs:

                if sim < SIM_THRESH:
                    continue

                scores.append((int(j), float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)

        topk_id = scores[:min(topk, len(scores))] 
        out[int(i)] = [paths[k[0]] for k in topk_id]

    return out, tgt_idxs

def colqwen_similarity(Vi, Vj):
    S = Vi @ Vj.T
    return 0.5 * (
        S.max(dim=1).values.mean().item() +
        S.max(dim=0).values.mean().item()
    )

@torch.no_grad()
def colqwen_score_image_to_images(processor, model, query_img_path: str, cand_img_paths: list[str]) -> list[float]:
    # query as 1-image batch
    q_img = Image.open(query_img_path).convert("RGB")
    c_imgs = [Image.open(p).convert("RGB") for p in cand_img_paths]

    batch_q = processor.process_images([q_img]).to(model.device)
    batch_c = processor.process_images(c_imgs).to(model.device)

    q_emb = model(**batch_q)      # multi-vector embeddings
    c_emb = model(**batch_c)

    # scores: (n_query=1, n_cand)
    scores = processor.score_multi_vector(q_emb, c_emb)[0].detach().cpu().tolist()
    return scores


@torch.no_grad()
def colqwen_encode_images(model, processor, image_paths, device):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = model(**inputs).last_hidden_state  # (N, T, D)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats

def load_colqwen(COLQWEN_NAME: str, device="cuda"):
    model = ColQwen2_5.from_pretrained(
        COLQWEN_NAME,
        torch_dtype=torch.bfloat16,   # or float16 if needed
        device_map="cuda:0" if device.startswith("cuda") else "cpu",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(COLQWEN_NAME)
    return processor, model

def img_if_new_check_colqwen(
    window_img_paths,
    scan_id,
    image_buffer,
    colqwen_model,
    colqwen_proc,
    device="cuda",
    SIM_THRESH=0.45,
    topk_buf=5,
):
    # Step 1: SigLIP prefilter (reuse your existing code)
    frames = [os.path.basename(p) for p in window_img_paths]
    rag_sim_imgs_window, tgt_idxs = compute_sim_for_window(
        scan_id, frames, image_buffer
    )

    keep_imgs = []

    for img_path, global_i in zip(window_img_paths, tgt_idxs):

        # Fail-open
        if global_i is None:
            keep_imgs.append(img_path)
            continue

        buf_imgs = rag_sim_imgs_window.get(global_i, [])[:topk_buf]

        # No buffer → new
        if len(buf_imgs) == 0:
            keep_imgs.append(img_path)
            continue

        # Step 2: ColQwen similarity
        scores = colqwen_score_image_to_images(colqwen_proc, colqwen_model, img_path, buf_imgs)
        max_sim = max(scores) if scores else -1e9
        if max_sim < SIM_THRESH:
            keep_imgs.append(img_path)
        # else: repeat → drop

    return keep_imgs
