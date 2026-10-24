# Force clean TMPDIR first (before any other imports)
import os
from pathlib import Path
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
import json
import glob
import base64
import pickle
import time
import logging


from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from copy import deepcopy
from contextlib import contextmanager
from vllm import LLM, SamplingParams
import re as _re, json as _json
import numpy as np
from PIL import Image
import torch
from repeat_judge_1st import img_if_new_check_colqwen, load_colqwen
from transformers import SiglipModel, SiglipImageProcessor
from transformers import AutoModelForImageTextToText, AutoProcessor
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Config
# =========================
CLASSES_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/data_GT/3RScan_3RScan160/classes.txt"
RELATIONS_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/data_GT/3RScan_3RScan160/relationships.txt"
KB_ROOT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_fram_sg_embed_note_guided_wo_validset/frame_embeddings"

QWEN_MODEL_NAME_2ND = "/hkfs/work/workspace/scratch/rg5174-ragspace/models/Qwen3-VL-32B-Instruct"
COLQWEN_NAME = "vidore/colqwen2.5-v0.1"

WINDOW_SIZE = 6
STRIDE      = 4  # overlap = WINDOW_SIZE - STRIDE = 2

FRAME_RE_SEQ = re.compile(r"frame-(\d+)\.color\.(?:jpg|jpeg|png)$", re.IGNORECASE)
FRAME_RE_KB  = re.compile(r"frame-(\d+)(?:\.color)?(?:\.(?:jpg|jpeg|png))?$", re.IGNORECASE)
UUID_RE      = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
IF_API       = False
IF_vLLM      = True

# ---- Global LLM cache for vLLM ----
_GLOBAL_VLLM = None

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# =========================
# IO utils
# =========================
def atomic_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def safe_read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_sequence_rgbs(sequence_dir: str) -> List[str]:
    paths = glob.glob(os.path.join(sequence_dir, "frame-*.color.jpg")) \
          + glob.glob(os.path.join(sequence_dir, "frame-*.color.png")) \
          + glob.glob(os.path.join(sequence_dir, "frame-*.color.jpeg"))
    def _idx(p):
        m = FRAME_RE_SEQ.search(os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(paths, key=_idx)

def plan_windows(n: int, win: int, stride: int) -> List[Tuple[int, int]]:
    out = []
    i = 0
    while i < n:
        j = min(i + win, n)
        out.append((i, j))
        if j == n:
            break
        i += stride
    return out

def load_vocab_list(path: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(s)
    return items

def setup_logger(log_path: str, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers if reusing the same logger_name (safe for resume)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


@contextmanager
def timed(logger: logging.Logger, name: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    logger.info(f"[TIMER] {name}: {dt:.3f}s")

# =========================
# Text helpers
# =========================
def normalize_text(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", (s or "").strip().lower())

def _extract_array_objects(raw, key):
    m = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if not m:
        return []
    i = m.end()
    objs, depth, start = [], 0, None
    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(raw[start:i+1])
                start = None
        elif ch == "]":
            if depth == 0:
                break
        i += 1
    return objs

def salvage_local_from_raw(raw_text, max_nodes=10, max_edges=30):
    if not raw_text:
        return None
    node_objs = _extract_array_objects(raw_text, "new_nodes") or _extract_array_objects(raw_text, "nodes")
    edge_objs = _extract_array_objects(raw_text, "new_edges") or _extract_array_objects(raw_text, "edges")

    nodes, edges = [], []
    for t in node_objs[:max_nodes]:
        try:
            n = json.loads(t)
            nid = str(n.get("id", "")).strip()
            if not nid:
                continue
            nodes.append({"id": nid, "name": n.get("name", ""), "description": n.get("description", "")})
        except Exception:
            continue

    keep_ids = {str(n["id"]) for n in nodes}
    for t in edge_objs:
        if len(edges) >= max_edges:
            break
        try:
            e = json.loads(t)
            src = str(e.get("src", "")).strip()
            tgt = str(e.get("tgt", "")).strip()
            ce = e.get("c_edge",None)
            if isinstance(ce, list):
                rel = [str(x).strip().lower() for x in ce if str(x).strip()]
            else:
                rel = str(ce or "").strip().lower()
            if src in keep_ids and tgt in keep_ids and rel:
                edges.append({"src": src, "tgt": tgt, "c_edge": rel})
        except Exception:
            continue

    if nodes:
        return {"nodes": nodes, "edges": edges}
    return None

# =========================
# Global graph merge (IDs are global)
# =========================

def _to_int_safe(x):
    try:
        return int(str(x))
    except Exception:
        return None


def build_id_map(local_nodes, global_nodes_by_id):
    """
    New build_id_map:
    - No longer does similarity matching based on name/description
    - Fully trusts the model-generated node["id"] as the "global id the model intends"
    - If the id already exists in global_nodes_by_id, treat it as the same object and reuse directly
    - If the id does not exist:
        * If the id is an integer larger than the current maximum used id, use it directly as a new global id
        * Otherwise, to avoid conflicts, assign a new incrementing id (next_id) and map the local id to it
    """
    id_map = {}
    new_global_nodes = []

    # Count integer ids currently occupied in the global graph
    existing_int_ids = []
    for gid in global_nodes_by_id.keys():
        v = _to_int_safe(gid)
        if v is not None:
            existing_int_ids.append(v)
    max_existing = max(existing_int_ids) if existing_int_ids else 0
    next_id = max_existing + 1

    used_ids = set(global_nodes_by_id.keys())  # string form

    for node in local_nodes:
        lid_raw = node.get("id")
        lid = str(lid_raw).strip()
        name = node.get("name", "")
        desc = node.get("description", "")

        # Case 1: the id given by the model already exists in the global graph -> treat as the same object
        if lid in global_nodes_by_id:
            id_map[lid] = lid
            # If you want to update the global description, you can merge description here too
            # e.g., append the new desc (optional)
            continue

        # Case 2: this id has not been used yet
        lid_int = _to_int_safe(lid)
        if lid_int is not None and lid_int > max_existing and lid not in used_ids:
            # The model gave an integer larger than all current ids, treat it as a valid new global id
            g_id = lid
            used_ids.add(g_id)
            # Also update max_existing / next_id to avoid future conflicts
            if lid_int > max_existing:
                max_existing = lid_int
                next_id = max_existing + 1
        else:
            # The model gave a small integer already in use / non-integer id / other unusual form,
            # to be safe, assign a truly unique new_id
            g_id = str(next_id)
            next_id += 1
            used_ids.add(g_id)

        id_map[lid] = g_id
        new_global_nodes.append({
            "id": g_id,
            "name": name,
            "description": desc,
        })

    return id_map, new_global_nodes


def merge_graphs(global_graph: Dict[str, Any], local_graph: Dict[str, Any]) -> Dict[str, Any]:

    if "nodes" not in global_graph:
        global_graph["nodes"] = []
    if "edges" not in global_graph:
        global_graph["edges"] = []

    global_nodes_by_id = {str(n["id"]): n for n in global_graph["nodes"]}

    local_nodes = (local_graph.get("nodes") or [])
    local_edges = (local_graph.get("edges") or [])

    # --- normalize local (no seen_in_frames) ---
    for n in local_nodes:
        n["id"] = str(n.get("id"))
        # keep only required fields
        n["name"] = n.get("name", "")
        n["description"] = n.get("description", "")

    for e in local_edges:
        # normalize relationship field
        if "c_edge" not in e:
            e["c_edge"] = e.get("predicate") or e.get("relation") or ""

    # Build local->global mapping + new nodes
    id_map, new_global_nodes = build_id_map(local_nodes, global_nodes_by_id)

    # ---- Merge nodes (no seen_in_frames) ----
    for ln in local_nodes:
        ln_id  = str(ln.get("id"))
        ln_name = ln.get("name", "")
        ln_desc = ln.get("description", "")
        gid = id_map[ln_id]

        if gid in global_nodes_by_id:
            cur = global_nodes_by_id[gid]
            if not isinstance(cur, dict):
                # fallback: if history is invalid/None, overwrite with the full node
                cur = {"id": gid, "name": ln_name, "description": ln_desc}
                global_nodes_by_id[gid] = cur

            # fill in name (force rename)
            if ln_name:
                cur["name"] = ln_name
            # only update description if new one is “richer”
            if len(normalize_text(ln_desc)) > len(normalize_text(cur.get("description", ""))):
                cur["description"] = ln_desc
        else:
            # new global node: only three fields
            global_nodes_by_id[gid] = {"id": gid, "name": ln_name, "description": ln_desc}

    # ---- Merge edges and deduplicate ----
    # prepare existing edges: map (src,tgt) -> edge index
    pair2idx = {}
    for idx, e in enumerate(global_graph["edges"]):
        src = str(e.get("src")).strip()
        tgt = str(e.get("tgt")).strip()
        pair2idx[(src, tgt)] = idx

    # merge
    for e in local_edges:
        src_g = id_map[str(e["src"]) if e["src"] is not None else ""]
        tgt_g = id_map[str(e["tgt"]) if e["tgt"] is not None else ""]

        if src_g == "" or tgt_g == "":
            continue
        
        preds = e["c_edge"]

        key = (src_g, tgt_g)
        if key in pair2idx:
            i = pair2idx[key]
            old_preds = global_graph["edges"][i]["c_edge"]
            if isinstance(old_preds, str): old_preds = [old_preds]
            if isinstance(preds, str): preds = [preds]
            merged = list(dict.fromkeys(old_preds + preds))

            global_graph["edges"][i]["c_edge"] = merged
        else:
            global_graph["edges"].append({
                "src": src_g,
                "tgt": tgt_g,
                "c_edge": preds
            })
            pair2idx[key] = len(global_graph["edges"]) - 1


    # Sort by numeric ID (global IDs are guaranteed to be integers)
    global_graph["nodes"] = sorted(global_nodes_by_id.values(), key=lambda x: int(x["id"]))
    return global_graph

# =========================
# vllm helpers
# =========================
def get_qwen_vllm():
    """
    Lazily create a single vLLM LLM instance and reuse it
    across all windows / scans within this python process.
    """
    global _GLOBAL_VLLM
    if _GLOBAL_VLLM is None:
        print("[vLLM] Initializing Qwen3-VL-32B engine once ...")
        _GLOBAL_VLLM = LLM(
            model=QWEN_MODEL_NAME_2ND,
            mm_encoder_tp_mode="data",
            tensor_parallel_size=max(torch.cuda.device_count(), 1),
            enable_expert_parallel=False,    # disable EP on single GPU
            gpu_memory_utilization=0.60,     # avoid OOM
            enforce_eager=False,              # Temporarily True to avoid compile timeout
            # executor="uniproc",            # keep this if you are currently using it
        )
    return _GLOBAL_VLLM

def prepare_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8192):
    from qwen_vl_utils import process_vision_info
    import re as _re, json as _json

    # === Build input ===
    text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        text_token_ids = qwen_proc.tokenizer.encode(text, add_special_tokens=False)
        text_tokens = len(text_token_ids)
    except Exception:
        text_tokens = -1

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16,
        return_video_kwargs=True, return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs
    },text_tokens

# =========================
# RAG: Note-guided Step1, wo Faiss
# =========================
def parse_frame_id_from_path(p: str) -> int:
    m = re.search(r"frame-(\d+)\.json$", os.path.basename(p))
    return int(m.group(1)) if m else -1

def load_siglip(model_name="google/siglip2-base-patch16-224", device="cuda"):
    processor = SiglipImageProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name).to(device).eval()
    return processor, model, device

_KB_CACHE = None  # (kb_mat_torch, kb_scan_ids)

def load_kb_frame_text_embs(kb_root: str, device: str = "cuda"):
    """
    Load ALL frame-level text embeddings (note_text) produced by frame_sg_emb.py.
    Expect per scan:
      {kb_root}/{scan_id}/frame_text_embeds.npz  with keys: embeddings
    Return:
      kb_mat: torch.FloatTensor [N, D] (L2 normalized), on device
      kb_scan_ids: List[str] length N
    """
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE

    npz_list = sorted(glob.glob(os.path.join(kb_root, "*", "frame_text_embeds.npz")))
    if not npz_list:
        raise FileNotFoundError(f"No frame_text_embeds.npz under: {kb_root}")

    all_embs = []
    all_scan_ids = []
    all_json_paths = []

    for npz_path in npz_list:
        scan_id = os.path.basename(os.path.dirname(npz_path))
        data = np.load(npz_path, allow_pickle=True)
        embs = data["embeddings"].astype(np.float32)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

        # key: json_paths is stored in the npz file
        json_paths = data["json_paths"].tolist()

        all_embs.append(embs)
        all_scan_ids.extend([scan_id] * embs.shape[0])
        all_json_paths.extend([str(p) for p in json_paths])

    kb = np.concatenate(all_embs, axis=0)
    kb_mat = torch.from_numpy(kb).to(device=device, dtype=torch.float32)

    _KB_CACHE = (kb_mat, all_scan_ids, all_json_paths)
    return _KB_CACHE

def _merge_reference_sgs(ref_jsons: List[Dict[str, Any]],
                         winning_scan: str,
                         keep_policy: str = "most_edges",
                         max_edges_per_object: Optional[int] = None) -> Dict[str, Any]:
    by_id: Dict[int, List[Tuple[Dict[str, Any], int]]] = defaultdict(list)
    frames_used = set()

    for r in ref_jsons:
        sg = r.get("json") or {}
        f_id = int(r.get("kb_frame_id"))
        frames_used.add(f_id)
        for obj in sg.get("objects", []) or []:
            oid = obj.get("id")
            if oid is None:
                continue
            by_id[oid].append((obj, f_id))

    kept_pre: Dict[int, Dict[str, Any]] = {}
    for oid, lst in by_id.items():

        if keep_policy == "most_edges":
            rep = max(lst, key=lambda x: len((x[0].get("edges") or [])))
        elif keep_policy == "min_frame":
            rep = min(lst, key=lambda x: x[1])
        else:
            rep = lst[0]
        kept_pre[oid] = deepcopy(rep[0])

        for i,x in enumerate(lst):
            for edge in x[0]["edges"]:
                if "edges" not in kept_pre[oid]:
                    kept_pre[oid]["edges"] = []
                kept_pre[oid]["edges"].append(edge)


    kept = {oid: o for oid, o in kept_pre.items()}
    kept_ids = set(kept.keys())
    # print(f"[Debug] kept object ids: {kept_ids}")
    # print(f"[Debug] kept objects: {kept}")
    new_edges, seen = [], set()
    for oid, obj in kept.items():
        edges = obj.get("edges", []) or []
        for e in edges:
            src_name = e.get("src_label") if "src_label" in e else ""
            src = e.get("src") if "src" in e else None
            tgt_name = e.get("tgt_label") if "tgt_label" in e else ""
            tgt = e.get("tgt") if "tgt" in e else None
            rel = (e.get("relationship") or e.get("pred_label") or e.get("c_edge") or "").strip()
            if not rel or src is None or tgt is None:
                continue
            if (src in kept_ids) and (tgt in kept_ids):
                key = (src_name, rel, tgt_name)
                if key not in seen:
                    new_edges.append({"src": str(src_name), "tgt": str(tgt_name), "c_edge": rel})
                    seen.add(key)

    if isinstance(max_edges_per_object, int) and max_edges_per_object > 0:
        # simple global cap
        new_edges = new_edges[:max_edges_per_object]

    merged = {
        "scan": winning_scan,
        "frames_used": sorted(frames_used),
        "edges": new_edges
    }
    return merged

def scan_sg_to_reference_edges(scan_sg: dict) -> list:
    """
    Return: List[{"src": str, "tgt": str, "c_edge": str}]
    """
    # 1) id -> name (adapt field names to your scan-level SG schema)
    id2name = {}
    for n in (scan_sg.get("nodes") or scan_sg.get("objects") or []):
        nid = n.get("id")
        name = n.get("name") or n.get("label") or n.get("class_name")
        if nid is not None and name:
            id2name[str(nid)] = str(name)

    # 2) edges
    out = []
    edges = scan_sg.get("edges") or scan_sg.get("relationships") or []
    for e in edges:
        src = e.get("src") or e.get("source") or e.get("subject_id")
        tgt = e.get("tgt") or e.get("target") or e.get("object_id")
        rel = e.get("c_edge") or e.get("relationship") or e.get("predicate") or e.get("relation")

        if src is None or tgt is None or rel is None:
            continue

        src_name = id2name.get(str(src), str(src))
        tgt_name = id2name.get(str(tgt), str(tgt))

        # rel may be a list (to keep one rel per edge in the final output)
        if isinstance(rel, list):
            for r in rel:
                r = str(r).strip()
                if r:
                    out.append({"src": str(src_name), "tgt": str(tgt_name), "c_edge": r})
        else:
            r = str(rel).strip()
            if r:
                out.append({"src": str(src_name), "tgt": str(tgt_name), "c_edge": r})
    return out

@torch.no_grad()
def run_rag_for_window(
    sequence_dir: str,
    kb_mat: torch.Tensor,          # [N, D]  (CPU or GPU)
    kb_scan_ids: list,             # len N
    kb_json_paths: list,           # len N, each is frame_sg json path
    query_rgbs: list,              # M images
    model_bundle,
    out_json_path: str,
    topk: int = 5,                 # final picked frames
) -> list:
    """
    New logic:
      1) encode all query images -> Q [M, D]
      2) R = kb_mat @ Q^T -> [N, M]
      3) for each query col j:
           pk_j = softmax(R[:,j])
           best_idx_j = argmax(pk_j)
      4) collect M best frames -> dedup
      5) choose top-5 among these by their pk score (if <5 because duplicates, keep all)
      6) load those frame-level SG jsons and merge to reference_edges (like your CoT script)
    """

    if not query_rgbs:
        raise FileNotFoundError(f"No query RGBs provided for {sequence_dir}")

    processor, model, device = model_bundle

    # ---- 1) encode all query images ----
    q_vecs = []
    for p in query_rgbs:
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        feat = model.get_image_features(pixel_values=pixel_values, normalize=True)  # [1, D]
        v = feat[0].detach().to(torch.float32).cpu()  # move to CPU to match kb_mat if kb_mat on CPU
        q_vecs.append(v)

    Q = torch.stack(q_vecs, dim=0)  # [M, D]

    # make devices consistent
    if kb_mat.device.type != Q.device.type:
        Q = Q.to(kb_mat.device)

    # ---- 2) R = kb_mat @ Q^T (matrix mul) ----
    # R: [N, M]
    R = kb_mat @ Q.t()

    # ---- 3) strict step1: pk = softmax(r) per query image ----
    # pk: [N, M]
    PK = torch.softmax(R, dim=0)

    # PK: [N, M], R: [N, M]
    best_pk, best_idx_pk = torch.max(PK, dim=0)  # [M] each query's best by pk

    # Note: theoretically best_idx_pk and best_idx_r should be consistent (softmax is monotone within a column),
    # but here we use best_idx_pk as index and take r at the same idx, for a stricter “re-sort by step1” approach.
    cands = []
    for qi in range(best_idx_pk.numel()):
        idx = int(best_idx_pk[qi].item())
        pk  = float(best_pk[qi].item())
        r_best = float(R[idx, qi].item())  # original r score at the same idx
        cands.append((idx, pk, r_best, qi))


    # dedup by kb_row: keep max pk if same row appears multiple times
    best_by_row = {}
    for idx, pk, r_best, qi in cands:
        if (idx not in best_by_row) or (r_best > best_by_row[idx][1]):
            best_by_row[idx] = (pk, r_best, qi)

    uniq = [(idx, v[0], v[1], v[2]) for idx, v in best_by_row.items()]
    # now uniq: (idx, pk, r_best, qi)
    # ---- 5) choose top-5 by pk ----
    uniq_sorted = sorted(uniq, key=lambda x: x[2], reverse=True)  # x[2] is r_best
    chosen = uniq_sorted[:topk]

    chosen_refs = []
    for idx, pk, r_best, qi in chosen:
        json_path = kb_json_paths[idx]
        if not os.path.isfile(json_path):
            print(f"[WARN] missing frame json -> skip: {json_path}")
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                sg = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load json ({json_path}): {e}")
            continue

        chosen_refs.append({
            "kb_row": idx,
            "kb_scan_id": kb_scan_ids[idx],
            "kb_frame_json_path": json_path,
            "pk": float(pk),
            "r_best": float(r_best),
            "query_index": int(qi),
            "json": sg,
        })


    # If chosen_refs is empty due to missing files, fall back to at least taking the top-1 row
    if not chosen_refs:
        idx0, pk0, r_best0, qi0 = uniq_sorted[0]
        json_path = kb_json_paths[idx0]
        with open(json_path, "r", encoding="utf-8") as f:
            sg = json.load(f)
        chosen_refs = [{
            "kb_row": idx0,
            "kb_scan_id": kb_scan_ids[idx0],
            "kb_frame_json_path": json_path,
            "pk": float(pk0),
            "r_best": float(r_best0),
            "query_index": int(qi0),
            "json": sg,
        }]

    # ---- 6) merge frame-level SGs to reference_edges ----
    # Directly reuse the merge output format from _merge_reference_sgs in CoT (edges only)
    merged_reference_sg = _merge_reference_sgs(
        ref_jsons=[
            {"kb_frame_id": parse_frame_id_from_path(r["kb_frame_json_path"]), "json": r["json"]}
            for r in chosen_refs
        ],
        winning_scan="frame_level_refs",
        keep_policy="most_edges",
        max_edges_per_object=200
    )
    merged_edges = merged_reference_sg.get("edges", [])

    out = {
        "step1": {
            "note": "matrix R=kb@Q^T; pk=softmax(R, dim=0); per-query argmax -> M frames; dedup; pick top-5 by r_best"
        },
        "query_num": int(Q.shape[0]),
        "picked_frames": [
            {
                "kb_row": r["kb_row"],
                "pk": r["pk"],
                "query_index": r["query_index"],
                "scan_id": r["kb_scan_id"],
                "frame_json_path": r["kb_frame_json_path"]
            } for r in chosen_refs
        ],
        "reference_edges": merged_edges,
    }
    atomic_write_json(out_json_path, out)
    return merged_edges



# =========================
# Qwen inference (window-level, with previous_global + ref_edges)
# =========================
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
        # Determine type based on file extension; default to jpg here
        return f"data:image/jpeg;base64,{base64_data}"
    
def build_prompt_window_rag(
    window_rgbs: List[str],
    prev_global: Optional[Dict[str, Any]],
    reference_edges: List[Dict[str, Any]],
    classes_list: List[str],
    relations_list: List[str],
) -> List[Dict[str, Any]]:
    
    import json as _json

    content: List[Dict[str, Any]] = []

    # --- visual inputs ---
    for p in window_rgbs:
        content.append({"type": "image", "image": p})

    # --- stringify context blocks (keep exactly your variable names) ---
    prev_str = _json.dumps(prev_global or {"nodes": [], "edges": []},
                           ensure_ascii=False, separators=(",", ":"))
    refs_str = _json.dumps({"edges": reference_edges or []},
                           ensure_ascii=False, separators=(",", ":"))


    # classes_txt kept for compatibility (even if not used in guide)
    classes_txt   = ", ".join(classes_list)
    relations_txt = ", ".join(relations_list)

    # --------- main guide (uses tags, no huge inline text) ----------
    guide = f"""
You are a reliable 3D scene graph annotator for indoor scenes.

You are given:
- multiple images of ONE scene window, which contains several images,
- a PREVIOUS global scene graph, it contains already detected nodes and edges of the scene same as images of this window (provided in <PREV_GRAPH>),
- reference relations from similar scenes (provided in <REF_EDGES>),
- the allowed predicate list (provided in <ALLOWED_PREDICATES>).
- the allowed object class list (see <ALLOWED_CLASSES>).

You MUST internally follow THREE sequential reasoning steps:
STEP 0: Consistency matching
STEP 1: Node identification
STEP 2: Relation prediction
You MUST output ONLY the final JSON and end with <END_JSON>.

STEP 0 — CONSISTENCY MATCHING

In this step, you must decide ID consistency BEFORE any node creation.

For each detected object O in this window, even if it is partially visible, you must:

  - Determine O's object class. The class MUST belong to <ALLOWED_CLASSES>

  - Treat clustered small objects of same type (e.g., multiple shoes, books, bottles) as one single node, but treat large objects (e.g., bed, sofa, table, chairs or larger furniture) separately.

  - Decide whether O corresponds to an OLD node from <PREV_GRAPH> or is NEW:

    You must reuse an old id if there is evidence that O is the SAME physical object of a node in <PREV_GRAPH>.

    Evidence types you MUST consider:
      1) similar relative position (left/right/front/behind/closeby to prominent objects), based on current images and edges in <PREV_GRAPH>,
      2) similar visual cues (color, material, shape, text/logo, subparts) based on current images with candidates' descriptions in <PREV_GRAPH>,
      3) if O is a floor, ceiling or door, they are unique per scene, if there is already one in <PREV_GRAPH>, reuse it, otherwise create a new one,
 
  - Record the decision internally:
      1) All objects in this window that match an existing node id and are visible MUST be included in seen_old_nodes.

STEP 1 — NODE IDENTIFICATION

In this step, you must do:

  - Follow the consistency decisions made in STEP 0.
  - For each object marked as OLD in STEP 0:
      * Reuse its existing id from <PREV_GRAPH>.
      * Add ONLY its id to "seen_old_nodes".
      * DO NOT output it as a node.

  - For each object marked as NEW in STEP 0:
      * Create a new node entry in "new_nodes".
      * Assign a new global id strictly larger than MAX_EXISTING_ID in <PREV_GRAPH>.
      * Assign new ids in increasing order.

Rules:
  - "new_nodes" MUST contain ONLY NEW nodes.
  - "seen_old_nodes" MUST contain ONLY ids of reused old nodes.
  - Structural elements (floor, wall, ceiling, window, door, curtain) must be
    included if they appear in the window and were marked as NEW or OLD in STEP 0.
  - Component parts (e.g., chair leg, table leg, cabinet door, bicycle wheel)
    must NOT be separate nodes.

Node fields for NEW nodes:
  - "id": global integer ID
  - "name": short noun phrase (must match <ALLOWED_CLASSES>)
  - "description": visual cues (color, material, shape, parts, text/logo, spatial location, relation to surrounding objects) that help distinguish it from other same-class objects.


STEP 2 — RELATION PREDICTION

In this step, you must do:
    - First, carefully read the <REF_EDGES>, it includes all edges from a visually similar scan, you need to consider which src-tgt pairs have potential edges and what predicates they have.
    - Second, remember all existing edges from <PREV_GRAPH> and newly detected nodes from Step 1.
    - Third, based on the images and the nodes generated from Step 1, decide which (src, tgt) pairs have visually or logically supported relations, considering the experience you learned from <REF_EDGES>. However, DO NOT copy edges blindly from <REF_EDGES>.
    - Fourth, choose possible predicates for each (src, tgt) pair, strictly from the <ALLOWED_PREDICATES>, also consider edges between nodes from <PREV_GRAPH> and newly detected nodes.
    - Generate all possible edges, there are also potential edges between nodes distant from each other.
Rules:
  - Use ONLY predicates from <ALLOWED_PREDICATES>.
  - NEW edges can connect new nodes and old nodes in ANY combination:
      * old -> old (only if newly observed/confirmed in this window)
      * old -> new
      * new -> old
      * new -> new
  - If you output an edge (src,tgt), you MUST output EXACTLY 3 predicates:
      "c_edge": ["p1","p2","p3"]
    If only 1 predicate is obvious, choose 2 additional compatible predicates from the <ALLOWED_PREDICATES>
    (e.g., relative position left/right/front/behind/higher/lower when consistent).
  - "close by" alone is NOT enough; add extra spatial predicates (left/right/front/behind) when possible.
  - Each node must have at least three edges with its cloest three other nodes, for this reason, the edges number should be at least 3 times the nodes number.
  - For nodes on floor, there is must be at "standing on" or "supported by" edge to the floor node
  - For floor and wall, wall nodes must have at least one "attached to" edge to the floor node.

FINAL OUTPUT FORMAT 

Output ONLY one JSON object with exactly these keys:

{{
  "seen_old_nodes":[<INT>,<INT>,...]
  "new_nodes": [
    {{"id": <INT>, "name": "<short noun phrase>", "description": "<text>"}},
    ...
  ],
  "new_edges": [
    {{"src": <INT>, "tgt": <INT>, "c_edge": ["p1","p2","p3"]}},
    ...
  ]
}}

Return ONLY the JSON object. After the finish the json, print a newline and then:
<END_JSON>

""".strip()

# --- append guide first (stable prefix) ---
    content.append({"type": "text", "text": guide})

    # --- append tagged blocks (variable content) ---
    content.append({"type": "text", "text": f"<PREV_GRAPH>\n{prev_str}\n</PREV_GRAPH>"})
    content.append({"type": "text", "text": f"<REF_EDGES>\n{refs_str}\n</REF_EDGES>"})
    content.append({"type": "text", "text": f"<ALLOWED_PREDICATES>\n{relations_txt}\n</ALLOWED_PREDICATES>"})
    content.append({"type": "text", "text": f"<ALLOWED_CLASSES>\n{classes_txt}\n</ALLOWED_CLASSES>"})
    # (Optional) keep classes list if you want it visible to model later; currently not required for edges

    messages = [{"role": "user", "content": content}]
    return messages

from qwen_vl_utils import process_vision_info

@torch.no_grad()

def run_qwen_window(messages, qwen_model, qwen_proc, max_new_tokens: int = 8192):
    import re as _re, json as _json

    text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        text_token_ids = qwen_proc.tokenizer.encode(text, add_special_tokens=False)
        text_tokens = len(text_token_ids)
    except Exception:
        text_tokens = -1

    image_inputs, video_inputs, _ = process_vision_info(
        messages, image_patch_size=16,
        return_video_kwargs=True, return_video_metadata=True
    )
    inputs = qwen_proc(
        text=[text], images=image_inputs, videos=video_inputs,
        do_resize=False, return_tensors="pt"
    ).to(qwen_model.device)

    input_ids_len = inputs.input_ids.shape[1]
    max_ctx = getattr(qwen_model.config, "max_position_embeddings", None)
    if not isinstance(max_ctx, int) or max_ctx <= 0:
        max_ctx = getattr(qwen_proc.tokenizer, "model_max_length", 32768)

    print(f"[Diag] tokens_in(text_only)={text_tokens}, tokens_in(actual)={input_ids_len}, "
          f"max_ctx={max_ctx}, images={len(image_inputs)}")

    out_ids = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = out_ids.shape[1] - inputs.input_ids.shape[1]
    total_seq = out_ids.shape[1]
    print(f"[Diag] tokens_out(new)={new_tokens}, total_seq_len={total_seq}")

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out_text = qwen_proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if isinstance(out_text, list):
        out_text = "\n".join(out_text)

    m = _re.search(r"\{[\s\S]*\}", out_text)
    if not m:
        return None, out_text
    try:
        return _json.loads(m.group(0)), out_text
    except Exception:
        return None, out_text

def run_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8192):

    request, in_tokens = prepare_qwen_vllm(messages, qwen_proc, max_new_tokens)

    llm = get_qwen_vllm()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        top_k=-1,
        stop=["<END_JSON>"],
    )

    outputs = llm.generate([request], sampling_params)

    out = outputs[0].outputs[0]
    out_text = out.text
    out_tokens = len(out.token_ids)

    m = _re.search(r"\{[\s\S]*\}", out_text)
    if not m:
        return None, out_text  # <- preserve raw output
    try:
        return _json.loads(m.group(0)), out_text
    except Exception:
            return None, out_text
        
from http import HTTPStatus


# =========================
# Main per-scan pipeline
# =========================
def run_scan_with_rag_merge(
    scan_id: str,
    sequence_root: str,
    out_root: str,
    siglip_bundle,
    qwen_model,
    qwen_proc,
    colqwen_model,
    colqwen_proc,
    kb_mat,
    kb_scan_ids,
    kb_json_paths,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
):
    sequence_dir = os.path.join(sequence_root, scan_id, "sequence")
    out_dir = os.path.join(out_root, scan_id)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, f"{scan_id}.log")
    logger = setup_logger(log_path, logger_name=f"rag_sg.{scan_id}")
    logger.info(f"[START] scan={scan_id}")
    logger.info(f"[CFG] IF_vLLM={IF_vLLM} window_size={window_size} stride={stride}")


    # ---- 1) entire scan already completed -> skip directly ----
    final_path = os.path.join(out_dir, f"{scan_id}_global_graph_final.json")
    if os.path.exists(final_path):
        print(f"[Skip] {scan_id} already has final global graph.")
        return

    all_rgbs = list_sequence_rgbs(sequence_dir)
    if not all_rgbs:
        raise FileNotFoundError(f"No RGBs found under: {sequence_dir}")
    print(f"[Info] {scan_id}: {len(all_rgbs)} frames")

    win_plan = plan_windows(len(all_rgbs), window_size, stride)
    plan_path = os.path.join(out_dir, f"{scan_id}_windows.json")
    if not os.path.exists(plan_path):
        atomic_write_json(plan_path, [{"win_idx": i, "start": s, "end": e} for i, (s, e) in enumerate(win_plan)])

    classes_list = load_vocab_list(CLASSES_TXT)
    relations_list = load_vocab_list(RELATIONS_TXT)

    # ---- 2) window-level resume: check which window was last completed ----
    global_graph: Dict[str, Any] = {"nodes": [], "edges": []}
    start_win_idx = 0

    existing_global_files = glob.glob(os.path.join(out_dir, f"{scan_id}_global_win*.json"))
    if existing_global_files:
        last_idx = -1
        last_path = None
        for p in existing_global_files:
            base = os.path.basename(p)
            m = re.search(r"_global_win(\d+)\.json$", base)
            if not m:
                continue
            idx = int(m.group(1))
            if idx > last_idx:
                last_idx = idx
                last_path = p
        if last_path is not None:
            print(f"[Resume] {scan_id}: found existing global up to window #{last_idx}, loading {last_path}")
            gg = safe_read_json(last_path)
            if isinstance(gg, dict):
                global_graph = gg
                start_win_idx = last_idx + 1
            else:
                print(f"[WARN] Failed to load {last_path}, will restart from window 0.")

    # ---- 3) from start_win_idx to circulate window ----

    image_buffer: List[str] = [] # save absolute paths of finished query images

    for win_idx, (s, e) in enumerate(win_plan):

        if win_idx < start_win_idx:
            # already done
            continue

        t_win0 = time.perf_counter()
        logger.info(f"[WIN_START] win={win_idx:03d} range=[{s},{e})")

        win_rgbs = all_rgbs[s:e]
        print(f"\n=== {scan_id} | Window #{win_idx} [{s}..{e}) | {len(win_rgbs)} images ===")

        # --- build query set for RAG ---
        query_indices = list(range(s, e))
        query_rgbs = [all_rgbs[i] for i in query_indices if 0 <= i < len(all_rgbs)]


        if win_idx > 1:
            with timed(logger, f"win{win_idx:03d} repeat_judge"):
                query_rgbs = img_if_new_check_colqwen(
                    query_rgbs,
                    scan_id,
                    image_buffer,
                    colqwen_model,
                    colqwen_proc,
                    device=device
                )

                if not query_rgbs:
                    logger.warning(f"[WIN_SKIP] win={win_idx:03d} no new images")
                    continue
            logger.info(f"[JUDGE] kept_query_imgs={query_rgbs}")
        else:
            logger.info(f"[JUDGE] skipped (win_idx=0). query_imgs={query_rgbs}")

        if not query_rgbs:
            print(f"[WARN] All images in window #{win_idx} judged as repeats; skipping RAG and Qwen.")
            continue


        rag_json_path = os.path.join(out_dir, f"{scan_id}_rag_win{win_idx:03d}.json")

        for img_path in query_rgbs:
            if img_path not in image_buffer:
                image_buffer.append(img_path)

        reference_edges = run_rag_for_window(
            sequence_dir=sequence_dir,
            kb_mat=kb_mat,
            kb_scan_ids=kb_scan_ids,
            kb_json_paths=kb_json_paths,
            query_rgbs=query_rgbs,
            model_bundle=siglip_bundle,
            out_json_path=rag_json_path,
            topk=5,
)


        # Delete intermediate files after RAG results are consumed to save space
        try:
            if os.path.exists(rag_json_path):
                os.remove(rag_json_path)
        except Exception as e:
            print(f"[WARN] failed to remove RAG json {rag_json_path}: {e}")

        messages = build_prompt_window_rag(
            window_rgbs=win_rgbs,
            prev_global=global_graph if win_idx > 0 or start_win_idx > 0 else None,
            reference_edges=reference_edges,
            classes_list=classes_list,
            relations_list=relations_list,
        )

        local_json_path  = os.path.join(out_dir, f"{scan_id}_local_win{win_idx:03d}.json")
        raw_txt_path     = os.path.join(out_dir, f"{scan_id}_local_win{win_idx:03d}.raw.txt")
        global_json_path = os.path.join(out_dir, f"{scan_id}_global_win{win_idx:03d}.json")

        if IF_vLLM:
            parsed, raw_text = run_qwen_vllm(
                messages, qwen_proc, max_new_tokens=8192
            )

        else:
            parsed, raw_text = run_qwen_window(
                messages, qwen_model, qwen_proc, max_new_tokens=8192
            )

        if parsed is None:
            print("[WARN] Model did not return valid JSON. Saving raw output and attempting salvage...")
            try:
                with open(raw_txt_path, "w", encoding="utf-8") as f:
                    f.write(raw_text or "")
            except Exception as _e:
                print(f"[WARN] failed to save raw output: {_e}")
            recovered = salvage_local_from_raw(raw_text, max_nodes=20, max_edges=80)
            if not recovered:
                continue

            local_graph = recovered
        else:
            local_graph = parsed
            with open(raw_txt_path, "w", encoding="utf-8") as f:
                f.write(raw_text or "")

        # >>> delta(+context) -> legacy local graph for merge_graphs
        # Model outputs ONLY new_nodes/new_edges.
        # We auto-inject the referenced old nodes from previous global_graph to avoid:
        #  - long decode time (model listing old_nodes)
        #  - missing old_nodes causing KeyError / edge drop
        if isinstance(local_graph, dict) and ("new_nodes" in local_graph or "new_edges" in local_graph):
            new_nodes = local_graph.get("new_nodes") or []
            new_edges = local_graph.get("new_edges") or []

            # --- normalize ids as str early ---
            new_node_ids = {str(n.get("id")).strip() for n in new_nodes if n.get("id") is not None}

            # Collect all endpoint ids used in new_edges
            used_edge_ids = set()
            for e in new_edges:
                s_id = str(e.get("src", "")).strip()
                t_id = str(e.get("tgt", "")).strip()
                if s_id:
                    used_edge_ids.add(s_id)
                if t_id:
                    used_edge_ids.add(t_id)

            # old ids are those referenced by edges but not newly created in this window
            needed_old_ids = sorted(list(used_edge_ids - new_node_ids), key=lambda x: int(x) if x.isdigit() else x)

            # Build stub old nodes from previous global_graph
            global_nodes_by_id = {str(n.get("id")): n for n in (global_graph.get("nodes") or [])}
            old_nodes_stub = []
            for oid in needed_old_ids:
                gn = global_nodes_by_id.get(oid)
                if gn is None:
                    # If the model hallucinated an id, skip this id (edge may be filtered later)
                    continue
                old_nodes_stub.append({
                    "id": str(gn.get("id")),
                    "name": gn.get("name", ""),
                    "description": ""  # keep empty to avoid overwriting richer global desc
                })

            local_graph = {
                "nodes": old_nodes_stub + new_nodes,
                "edges": new_edges,
            }

        atomic_write_json(local_json_path, local_graph)

        global_graph = merge_graphs(global_graph, local_graph)
        atomic_write_json(global_json_path, global_graph)

        print(f"[OK] Window #{win_idx} merged. Global nodes={len(global_graph.get('nodes',[]))}, edges={len(global_graph.get('edges',[]))}")
        t_win = time.perf_counter() - t_win0
        logger.info(f"[TIMER] {scan_id} | win={win_idx:03d} [{s}..{e}) | total_window_time={t_win:.3f}s")



    # All windows processed, write the final global graph
    atomic_write_json(final_path, global_graph)
    print(f"\n✅ Done. Final global graph saved to: {final_path}")

# =========================
# Batch driver
# =========================
def get_scan_ids(test_list_path: str) -> List[str]:
    with open(test_list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    # These paths can be changed as needed
    test_list_path = "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
    sequence_root  = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan"
    out_root       = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/RAG_Qwen3/baseline_note_guided"
    os.makedirs(out_root, exist_ok=True)

    print("[Load] SigLIP encoder ...")
    siglip_bundle = load_siglip()


    print("[Load] Qwen model ...")
    qwen_model = None
    qwen_proc  = None
    if not IF_API:
        qwen_proc  = AutoProcessor.from_pretrained(QWEN_MODEL_NAME_2ND)
        if not IF_vLLM:
            print("[Load] Qwen model ...")
            qwen_model = AutoModelForImageTextToText.from_pretrained(QWEN_MODEL_NAME_2ND, dtype=torch.bfloat16, device_map="auto")

    colqwen_proc, colqwen_model = load_colqwen(COLQWEN_NAME, device=device)
    kb_mat, kb_scan_ids, kb_json_paths = load_kb_frame_text_embs(KB_ROOT, device="cpu")

    scan_ids = sorted(get_scan_ids(test_list_path))
    for scan_id in scan_ids:
        run_scan_with_rag_merge(
            scan_id=scan_id,
            sequence_root=sequence_root,
            out_root=out_root,
            siglip_bundle=siglip_bundle,
            qwen_model=qwen_model,
            qwen_proc=qwen_proc,
            colqwen_model=colqwen_model,
            colqwen_proc=colqwen_proc, 
            kb_mat=kb_mat,
            kb_scan_ids=kb_scan_ids,
            kb_json_paths=kb_json_paths,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )
            

