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
import dashscope
import pickle
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from copy import deepcopy
from vllm import LLM, SamplingParams
import re as _re, json as _json
import numpy as np
from PIL import Image
import faiss
import torch
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers import AutoModelForImageTextToText, AutoProcessor


# =========================
# Config
# =========================
CLASSES_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/data_GT/3RScan_3RScan160/classes.txt"
RELATIONS_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/data_GT/3RScan_3RScan160/relationships.txt"
QWEN_MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

WINDOW_SIZE = 6
STRIDE      = 4  # overlap = WINDOW_SIZE - STRIDE = 2

NAME_WEIGHT = 0.6
DESC_WEIGHT = 0.4
SIM_THRESHOLD = 0.62

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

def must_read_json(path: str):
    obj = safe_read_json(path)
    if obj is None:
        raise FileNotFoundError(f"JSON not found: {path}")
    return obj

def load_master_list(sequence_root: str, scan_id: str) -> List[str]:
    # Your DUSt3R filter output location
    master_list_path = os.path.join(sequence_root, scan_id, "dust3r_block30_filter", "master_list.json")
    master = must_read_json(master_list_path)
    if not isinstance(master, list) or len(master) == 0:
        raise FileNotFoundError(f"master_list.json missing/empty: {master_list_path}")
    # Unify to str
    master = [str(p) for p in master]
    return master

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

def mixed_similarity(name_a: str, desc_a: str, name_b: str, desc_b: str) -> float:
    return NAME_WEIGHT * jaccard_token_sim(name_a, name_b) + DESC_WEIGHT * jaccard_token_sim(desc_a, desc_b)

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

def cap_graph(g, max_nodes: int = 20, max_edges: int = 80):
    """
    Keep at most `max_nodes` nodes and `max_edges` edges.
    - Only keep edges whose src/tgt are in the kept node IDs.
    - Normalize edge key to 'c_edge'.
    """
    if isinstance(g, dict) and ("new_nodes" in g or "new_edges" in g):
        g = {"nodes": g.get("new_nodes") or [], "edges": g.get("new_edges") or []}

    if not isinstance(g, dict):
        return {"nodes": [], "edges": []}

    # 1) Truncate nodes
    nodes = (g.get("nodes") or [])[:max_nodes]

    # Allow id to be int/str, unify to str
    keep_ids = {str(n.get("id")) for n in nodes if n.get("id") is not None}

    # 2) Filter & truncate edges
    edges = []
    for e in (g.get("edges") or []):
        src = str(e.get("src", "")).strip()
        tgt = str(e.get("tgt", "")).strip()
        if not src or not tgt:
            continue
        if src not in keep_ids or tgt not in keep_ids:
            continue

        rel = (
            e.get("c_edge")
            or e.get("relationship")
            or e.get("predicate")
            or e.get("relation")
            or ""
        )

        if isinstance(rel, list):
            preds = [str(x).strip().lower() for x in rel if str(x).strip()]
        else:
            s = str(rel).strip()
            preds = [s.lower()] if s else []

        edges.append({"src": src, "tgt": tgt, "c_edge": preds})
        if len(edges) >= max_edges:
            break

    return {"nodes": nodes, "edges": edges}

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
    - No longer perform similarity matching based on name/description
    - Fully trust that the model-generated node["id"] is already the "global id as the model sees it"
    - If the id already exists in global_nodes_by_id, treat it as the same object and reuse directly
    - If the id does not exist:
        * If the id is an integer and greater than the current maximum used id, use it directly as the new global id
        * Otherwise, to avoid conflicts, assign a new incrementing id (next_id) and map the local id to it
    """
    id_map = {}
    new_global_nodes = []

    # Count the integer ids currently occupied in the global graph
    existing_int_ids = []
    for gid in global_nodes_by_id.keys():
        v = _to_int_safe(gid)
        if v is not None:
            existing_int_ids.append(v)
    max_existing = max(existing_int_ids) if existing_int_ids else 0
    next_id = max_existing + 1

    used_ids = set(global_nodes_by_id.keys())  # in string form

    for node in local_nodes:
        lid_raw = node.get("id")
        lid = str(lid_raw).strip()
        name = node.get("name", "")
        desc = node.get("description", "")

        # Case 1: The id from the model already exists in the global graph -> treat as the same object directly
        if lid in global_nodes_by_id:
            id_map[lid] = lid
            # If you want to update the global description, you can also merge description here
            # For example: append the new desc (optional)
            continue

        # Case 2: This id has not been used yet
        lid_int = _to_int_safe(lid)
        if lid_int is not None and lid_int > max_existing and lid not in used_ids:
            # The model provided an integer larger than all current ids, considered a valid new global id
            g_id = lid
            used_ids.add(g_id)
            # Also update max_existing / next_id to avoid future conflicts
            if lid_int > max_existing:
                max_existing = lid_int
                next_id = max_existing + 1
        else:
            # The model gave a small integer already in use / a non-integer id / some other unusual form,
            # For safety, we assign a truly unique new_id
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
        # Keep only required fields
        n["name"] = n.get("name", "")
        n["description"] = n.get("description", "")

    for e in local_edges:
        # Unify relationship field
        if "c_edge" not in e:
            e["c_edge"] = e.get("predicate") or e.get("relation") or ""

    # Build local->global mapping + add new nodes
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
                # Fallback: if the historical entry is abnormal/None, overwrite with the full node
                cur = {“id”: gid, “name”: ln_name, “description”: ln_desc}
                global_nodes_by_id[gid] = cur

            # Name completion (force rename)
            if ln_name:
                cur[“name”] = ln_name
            # Update description only when the new one is “richer”
            if len(normalize_text(ln_desc)) > len(normalize_text(cur.get(“description”, “”))):
                cur[“description”] = ln_desc
        else:
            # New global node: only three fields
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
            merged = list(dict.fromkeys(old_preds + preds))  # union (preserve order), dict keys must be unique
            global_graph["edges"][i]["c_edge"] = merged
        else:
            global_graph["edges"].append({
                "src": src_g,
                "tgt": tgt_g,
                "c_edge": preds
            })
            pair2idx[key] = len(global_graph["edges"]) - 1


    # Sort by numeric ID (global IDs are ensured to be numbers)
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
            model=QWEN_MODEL_NAME,
            mm_encoder_tp_mode="data",
            tensor_parallel_size=max(torch.cuda.device_count(), 1),
            enable_expert_parallel=False,    # Disable EP on single GPU
            gpu_memory_utilization=0.75,     # Avoid OOM
            enforce_eager=False,              # Temporarily True to avoid compile timeout
            # executor="uniproc",            # Keep this if you are currently using it
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
# RAG: SigLIP index -> reference edges
# =========================
def load_siglip(model_name: str = "google/siglip-base-patch16-224"):
    processor = SiglipImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name)
    model.eval()
    return processor, model

def embed_image(model_bundle, image: Image.Image):
    processor, model = model_bundle
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output[0].numpy()

def parse_kb_key_to_scan_frame(key: str) -> Tuple[str, int]:
    s = key.decode("utf-8", "ignore") if isinstance(key, (bytes, bytearray)) else str(key)
    s = s.replace("\\", "/")
    m_uuid = UUID_RE.search(s)
    m_fr = FRAME_RE_KB.search(s) or re.search(r"frame-(\d+)", s, re.IGNORECASE)
    if not m_uuid or not m_fr:
        raise ValueError(f"Cannot parse scan/frame from key: {key}")
    return m_uuid.group(0).lower(), int(m_fr.group(1))

def _vote_top_scan(cands: List[Dict[str, Any]]) -> str:
    by_scan = defaultdict(list)
    for c in cands:
        by_scan[c["kb_scan_id"]].append(c["distance"])
    counts = {sid: len(dists) for sid, dists in by_scan.items()}
    max_cnt = max(counts.values())
    tied = [sid for sid, cnt in counts.items() if cnt == max_cnt]
    if len(tied) == 1:
        return tied[0]
    means = {sid: (sum(by_scan[sid]) / len(by_scan[sid])) for sid in tied}
    best_mean = min(means.values())
    tied2 = [sid for sid, m in means.items() if abs(m - best_mean) < 1e-9]
    if len(tied2) == 1:
        return tied2[0]
    mins = {sid: min(by_scan[sid]) for sid in tied2}
    return min(mins, key=mins.get)

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

def run_rag_for_window(
    sequence_dir: str,
    embedding_index_path: str,
    rgb_keys_path: str,
    kb_scan_root: str,
    query_rgbs: List[str],
    model_bundle,
    out_json_path: str,
    faiss_topk: int = 5,
) -> List[Dict[str, Any]]:
    if not query_rgbs:
        raise FileNotFoundError(f"No query RGBs provided for {sequence_dir}")

    processor, model = model_bundle

    faiss_index = faiss.read_index(embedding_index_path)
    with open(rgb_keys_path, "rb") as f:
        rgb_keys = pickle.load(f)
    rgb_keys = [(k.decode("utf-8", "ignore") if isinstance(k, (bytes, bytearray)) else str(k)) for k in rgb_keys]

    embs = []
    for p in query_rgbs:
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            embs.append(outputs.pooler_output[0].numpy().astype(np.float32))
    embs = np.stack(embs, axis=0)

    cands = []
    per_query_top5 = []
    for qi, e in enumerate(embs):

        D, I = faiss_index.search(e.reshape(1, -1), faiss_topk)
        qlist = []
        for rank in range(faiss_topk):
            idx = int(I[0, rank]); dist = float(D[0, rank])
            key = rgb_keys[idx]
            try:
                kb_sid, kb_fid = parse_kb_key_to_scan_frame(key)
            except Exception as ex:
                print(f"[WARN] parse key failed, skip. key={key} err={ex}")
                continue
            entry = {
                "query_index": qi,
                "kb_scan_id": kb_sid,
                "kb_frame_id": int(kb_fid),
                "distance": dist,
                "rank": rank,
            }
            cands.append(entry)
            qlist.append(entry)
        per_query_top5.append(qlist)

    if not cands:
        raise RuntimeError("No valid FAISS candidates; check index/keys/embeddings.")

    winning_scan = _vote_top_scan(cands)
    print(f"[RAG] winning_kb_scan_id = {winning_scan}")

    chosen_refs = []
    for qlist in per_query_top5:
        w = [c for c in qlist if c["kb_scan_id"] == winning_scan]
        if not w:
            continue
        best = min(w, key=lambda x: x["distance"])
        json_path = os.path.join(kb_scan_root, winning_scan, "frame_sg", f"frame-{best['kb_frame_id']:06d}.json")
        if not os.path.isfile(json_path):
            print(f"[INFO] missing frame json -> skip: {json_path}")
            continue
        try:
            with open(json_path, "r") as f:
                sg = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load json ({json_path}): {e}")
            continue
        chosen_refs.append({
            "kb_scan_id": winning_scan,
            "kb_frame_id": best["kb_frame_id"],
            "json_path": json_path,
            "json": sg,
            "distance": float(best["distance"])
        })

    merged_reference_sg = _merge_reference_sgs(chosen_refs, winning_scan)
    merged_edges = merged_reference_sg.get("edges", [])

    vote_counts = Counter([c["kb_scan_id"] for c in cands])
    vote_stats = {"counts": dict(vote_counts), "winning_scan": winning_scan}

    out = {
        "vote_stats": vote_stats,
        "reference_edges": merged_edges,
    }
    atomic_write_json(out_json_path, out)
    print(f"[RAG] saved -> {out_json_path}")
    return merged_edges

# =========================
# Qwen inference (window-level, with previous_global + ref_edges)
# =========================
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
        # Determine type by file extension, default is jpg here
        return f"data:image/jpeg;base64,{base64_data}"
    
def build_prompt_window_rag(
    window_rgbs: List[str],
    prev_global: Optional[Dict[str, Any]],
    reference_edges: List[Dict[str, Any]],
    classes_list: List[str],
    relations_list: List[str],
    IF_API: bool = False,
) -> List[Dict[str, Any]]:
    
    import json as _json

    content: List[Dict[str, Any]] = []

    # --- visual inputs ---
    if IF_API:
        # DashScope / API style
        content.append({"video": [get_base64_image(path) for path in window_rgbs], "fps": 2})
    else:
        # local / vLLM style
        for p in window_rgbs:
            content.append({"type": "image", "image": p})

    # --- stringify context blocks (keep exactly your variable names) ---
    prev_str = _json.dumps(prev_global or {"nodes": [], "edges": []},
                           ensure_ascii=False, separators=(",", ":"))
    refs_str = _json.dumps({"edges": reference_edges or []},
                           ensure_ascii=False, separators=(",", ":"))


    # NOTE: classes_txt kept for compatibility (even if not used in guide)
    classes_txt   = ", ".join(classes_list)
    relations_txt = ", ".join(relations_list)

    # --------- main guide (uses tags, no huge inline text) ----------
    guide = f"""
You are a reliable 3D scene graph annotator for indoor scenes.

You are given:
- multiple images of ONE scene window, which contains several images,
- a PREVIOUS global scene graph, it includes some nodes and edges of the scene same as images of this window (provided in <PREV_GRAPH>),
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

For each detected object O in this window:

  - Determine O's object class. The class MUST belong to <ALLOWED_CLASSES>
    (or be a clear semantic synonym). Otherwise, IGNORE O completely.

  - Build a candidate set C from <PREV_GRAPH>:
      C = all existing nodes whose node names match O's object class or are semantically similar.

  - Decide whether O corresponds to an OLD node or is NEW:

    You must reuse an old id ONLY if there is evidence that O is the SAME physical object as a candidate in C.

    Evidence types you MUST consider:
      1) Structural attachment (e.g., attached to / supported by wall, floor, ceiling), based on current images and edges in <PREV_GRAPH>,
      2) Relative position (left/right/front/behind/closeby to prominent objects), based on current images and edges in <PREV_GRAPH>,
      3) Compare visual cues (color, material, shape, text/logo, subparts) based on current images with candidates' descriptions in <PREV_GRAPH>.

    If multiple candidates exist and the evidence is ambiguous or insufficient,
    treat O as one of them.

  - Record the decision internally:
      * If OLD: reuse the existing id and mark it as SEEN.
      * If NEW: mark it as NEW and assign a new id later in STEP 1.

IMPORTANT:
  - STEP 0 is the ONLY step where OLD vs NEW decisions are made.
  - STEP 1 MUST NOT re-evaluate consistency.


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
  - STEP 1 MUST NOT decide whether an object is OLD or NEW.
  - STEP 1 MUST NOT compare names/descriptions against <PREV_GRAPH>.
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
Rules:
  - Use ONLY predicates from <ALLOWED_PREDICATES>.
  - NEW edges can connect:
      * old -> old (only if newly observed/confirmed in this window)
      * old -> new
      * new -> old
      * new -> new
  - If you output an edge (src,tgt), you MUST output EXACTLY 3 predicates:
      "c_edge": ["p1","p2","p3"]
    If only 1 predicate is obvious, choose 2 additional compatible predicates from the <ALLOWED_PREDICATES>
    (e.g., relative position left/right/front/behind/higher/lower when consistent).
  - For each (src, tgt) pair:
    - Imagine you stand at the center of tgt, looking toward to the interior center, normally facing opposite to the wall behind tgt, determine the relations of src with respect to tgt from YOUR perspective. 
    - If there is at least one clear relation, you MUST output **at least 1** predicate in "c_edge".
    - If multiple relations are visually supported (e.g., "standing on" and "supported by"), you MUST include **all of them** in "c_edge", up to 3 predicates in total.
  - "close by" alone is NOT enough; add extra spatial predicates when possible.

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

        # * No longer instantiate a new LLM, use the global singleton instead
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
            return None, out_text  # <- Preserve raw output
        try:
            return _json.loads(m.group(0)), out_text
        except Exception:
            return None, out_text

from http import HTTPStatus

def run_qwen_window_api(
    messages,
    qwen_model_name: str = "qwen3-vl-32b-instruct",
    max_new_tokens: int = 8192,
):
    """
    Use DashScope MultiModalConversation for window-level inference.
    - Input: messages (consistent with the output of build_prompt_window_rag)
    - Output: (parsed_graph_dict or None, raw_text or "")
    """
    import json as _json

    # It is recommended to configure the API Key via environment variable rather than hardcoding it
    # For example: in the shell run `export DASHSCOPE_API_KEY=sk-xxxx`
    # Then here: dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    # If you insist on passing it manually, you can also set dashscope.api_key = "sk-xxxx" outside
    try:
        response = dashscope.MultiModalConversation.call(
            api_key="sk-09a3812f7fc34d8abe65aae9d4132a9f",
            model=qwen_model_name,
            messages=messages,
            result_format="message",            # Return OpenAI-style message
            response_format={"type": "json_object"},  # Enable JSON Mode
            max_tokens=max_new_tokens,         # Control output length
        )
    except Exception as e:
        print(f"[DashScope] API call failed: {e}")
        return None, ""

    # According to DashScope docs, check status_code to determine success
    if getattr(response, "status_code", None) != HTTPStatus.OK:
        print(f"[DashScope] Error code={getattr(response, 'code', None)}, "
              f"message={getattr(response, 'message', None)}")
        return None, ""

    # Normal case: response.output.choices[0].message.content
    try:
        content = response.output.choices[0].message.content
    except Exception as e:
        print(f"[DashScope] Unexpected response format: {e}")
        return None, ""

    raw_text = ""
    parsed = None

    # Case 1: content is a string, directly a JSON string
    if isinstance(content, str):
        raw_text = content.strip()
        try:
            parsed = _json.loads(raw_text)
        except Exception:
            parsed = None

    # Case 2: content is a list, each element is {"type": "...", "text": "..."} or {"type":"json_object","json":{...}}
    elif isinstance(content, list):
        # Prefer to find the json_object segment first
        for seg in content:
            if not isinstance(seg, dict):
                continue
            # In some modes it directly provides {"type": "json_object", "json": {...}}
            if seg.get("type") == "json_object" and "json" in seg:
                try:
                    parsed = seg["json"]
                    raw_text = _json.dumps(parsed, ensure_ascii=False)
                    break
                except Exception:
                    pass
            # More often it is the {"type":"text","text":"<json string>"} form
            if "text" in seg and isinstance(seg["text"], str):
                raw_text_candidate = seg["text"].strip()
                # Strip possible ```json wrapping
                tmp = raw_text_candidate.strip()
                tmp = tmp.removeprefix("```json").removesuffix("```").strip()
                try:
                    parsed_candidate = _json.loads(tmp)
                    parsed = parsed_candidate
                    raw_text = raw_text_candidate
                    break
                except Exception:
                    # On parse failure, record raw_text first for outer salvage
                    if not raw_text:
                        raw_text = raw_text_candidate

        # If parsing has not succeeded yet but raw_text exists, try one more full loads
        if parsed is None and raw_text:
            try:
                tmp = raw_text.strip()
                tmp = tmp.removeprefix("```json").removesuffix("```").strip()
                parsed = _json.loads(tmp)
            except Exception:
                parsed = None

    else:
        print(f"[DashScope] Unsupported content type: {type(content)}")
        return None, ""

    return parsed, (raw_text or "")


# =========================
# Main per-scan pipeline
# =========================
def run_scan_with_rag_merge(
    scan_id: str,
    sequence_root: str,
    out_root: str,
    embedding_index_path: str,
    rgb_keys_path: str,
    kb_scan_root: str,
    siglip_bundle,
    qwen_model,
    qwen_proc,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
):
    sequence_dir = os.path.join(sequence_root, scan_id, "sequence")
    out_dir = os.path.join(out_root, scan_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) Entire scan already completed -> skip directly ----
    final_path = os.path.join(out_dir, f"{scan_id}_global_graph_final.json")
    if os.path.exists(final_path):
        print(f"[Skip] {scan_id} already has final global graph.")
        return

    # ---------- NEW: use master_list.json as frame source ----------
    all_rgbs = load_master_list(sequence_root, scan_id)
    print(f"[Info] {scan_id}: {len(all_rgbs)} frames (from master_list.json)")

    if all_rgbs is None or not all_rgbs:
        return

    # windows still by (window_size, stride) but indices refer to master_list
    win_plan = plan_windows(len(all_rgbs), window_size, stride)

    plan_path = os.path.join(out_dir, f"{scan_id}_windows.json")
    if not os.path.exists(plan_path):
        windows = []
        for win_idx, (s, e) in enumerate(win_plan):
            img_idx = list(range(s, e))
            img_path = [all_rgbs[k] for k in img_idx]
            windows.append({
                "win_idx": win_idx,
                "img_idx": img_idx,
                "img_path": img_path,
            })
        atomic_write_json(plan_path, windows)


    classes_list = load_vocab_list(CLASSES_TXT)
    relations_list = load_vocab_list(RELATIONS_TXT)

    # ---- 2) Window-level resume: check which window was reached ----
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

    # ---- 3) Loop through each window starting from start_win_idx ----
    windows_json = must_read_json(plan_path)
    if not isinstance(windows_json, list) or not windows_json:
        raise RuntimeError(f"bad windows json: {plan_path}")

    for win_idx, (s, e) in enumerate(win_plan):
        if win_idx < start_win_idx:
            continue

        w = windows_json[win_idx]
        win_img_idx = w["img_idx"]
        win_rgbs = w["img_path"]

        # Still display interval when printing (for alignment with previous logs)
        s_print = win_img_idx[0] if win_img_idx else -1
        e_print = (win_img_idx[-1] + 1) if win_img_idx else -1
        print(f"\n=== {scan_id} | Window #{win_idx} [{s_print}..{e_print}) | {len(win_rgbs)} images ===")

        # query = union(prev_window, curr_window) by img_idx (keep original logic unchanged)
        if win_idx == 0:
            query_indices = list(win_img_idx)
        else:
            prev_idx = windows_json[win_idx - 1]["img_idx"]
            query_indices = sorted(set(prev_idx) | set(win_img_idx))

        query_rgbs = [all_rgbs[i] for i in query_indices if 0 <= i < len(all_rgbs)]


        rag_json_path = os.path.join(out_dir, f"{scan_id}_rag_win{win_idx:03d}.json")

        reference_edges = run_rag_for_window(
            sequence_dir=sequence_dir,
            embedding_index_path=embedding_index_path,
            rgb_keys_path=rgb_keys_path,
            kb_scan_root=kb_scan_root,
            query_rgbs=query_rgbs,
            model_bundle=siglip_bundle,
            out_json_path=rag_json_path,
            faiss_topk=5,
        )

        # Delete intermediate files after RAG results are used, to save space
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
            IF_API=IF_API,
        )

        local_json_path  = os.path.join(out_dir, f"{scan_id}_local_win{win_idx:03d}.json")
        raw_txt_path     = os.path.join(out_dir, f"{scan_id}_local_win{win_idx:03d}.raw.txt")
        global_json_path = os.path.join(out_dir, f"{scan_id}_global_win{win_idx:03d}.json")

        if IF_API:
            parsed, raw_text = run_qwen_window_api(
                messages,
                qwen_model_name="qwen3-vl-32b-instruct",
                max_new_tokens=8192,
            )
        elif IF_vLLM:
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

    # All windows finished, write final global graph
    atomic_write_json(final_path, global_graph)
    print(f"\n✅ Done. Final global graph saved to: {final_path}")

# =========================
# Batch driver
# =========================
def get_scan_ids(test_list_path: str) -> List[str]:
    with open(test_list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    # You can modify these paths as needed
    test_list_path = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
    sequence_root  = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan"
    out_root       = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/RAG_baseline_Qwen3/RAG_end2end_dust3r"
    embedding_index_path = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset/rgb_index.ivfflat"
    rgb_keys_path        = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset/all_keys.pkl"
    kb_scan_root         = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan"
    os.makedirs(out_root, exist_ok=True)

    print("[Load] SigLIP encoder ...")
    siglip_bundle = load_siglip()


    print("[Load] Qwen model ...")
    qwen_model = None
    qwen_proc  = None
    if not IF_API:
        qwen_proc  = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
        if not IF_vLLM:
            print("[Load] Qwen model ...")
            qwen_model = AutoModelForImageTextToText.from_pretrained(QWEN_MODEL_NAME, dtype=torch.bfloat16, device_map="auto")

    scan_ids = sorted(get_scan_ids(test_list_path))
    for scan_id in scan_ids[:8]:
        run_scan_with_rag_merge(
            scan_id=scan_id,
            sequence_root=sequence_root,
            out_root=out_root,
            embedding_index_path=embedding_index_path,
            rgb_keys_path=rgb_keys_path,
            kb_scan_root=kb_scan_root,
            siglip_bundle=siglip_bundle,
            qwen_model=qwen_model,
            qwen_proc=qwen_proc,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )

