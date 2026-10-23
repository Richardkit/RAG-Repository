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
from transformers import SiglipProcessor, SiglipModel
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

WIN_RAG_ALPHA = 12.0

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

def salvage_local_from_raw(raw_text, max_nodes=15, max_edges=30):
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

    # 1) Truncate nodes.
    nodes = (g.get("nodes") or [])[:max_nodes]

    # Accept integer or string IDs and normalize them to strings.
    keep_ids = {str(n.get("id")) for n in nodes if n.get("id") is not None}

    # 2) Filter and truncate edges.
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
    Build a mapping from local node IDs to global IDs.
    - Do not match objects by name or description similarity.
    - Treat each model-generated node["id"] as its proposed global ID.
    - Reuse an ID already present in global_nodes_by_id as the same object.
    - For a new ID:
        * Accept an integer above the current maximum as a new global ID.
        * Otherwise allocate the next unused increasing ID and map the local ID to it.
    """
    id_map = {}
    new_global_nodes = []

    # Collect integer IDs already used in the global graph.
    existing_int_ids = []
    for gid in global_nodes_by_id.keys():
        v = _to_int_safe(gid)
        if v is not None:
            existing_int_ids.append(v)
    max_existing = max(existing_int_ids) if existing_int_ids else 0
    next_id = max_existing + 1

    used_ids = set(global_nodes_by_id.keys())  # String representation.

    for node in local_nodes:
        lid_raw = node.get("id")
        lid = str(lid_raw).strip()
        name = node.get("name", "")
        desc = node.get("description", "")

        # Case 1: An existing global ID refers to the same object.
        if lid in global_nodes_by_id:
            id_map[lid] = lid
            # Optionally merge the new description into the global description.
            # For example, append the new description when needed.
            continue

        # Case 2: This ID has not been used.
        lid_int = _to_int_safe(lid)
        if lid_int is not None and lid_int > max_existing and lid not in used_ids:
            # An integer above the current maximum is accepted as a new global ID.
            g_id = lid
            used_ids.add(g_id)
            # Update max_existing and next_id to prevent later collisions.
            if lid_int > max_existing:
                max_existing = lid_int
                next_id = max_existing + 1
        else:
            # For a reused smaller integer, a noninteger ID, or another unexpected format,
            # allocate a genuinely unique new_id to avoid collisions.
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
        # Retain only the required fields.
        n["name"] = n.get("name", "")
        n["description"] = n.get("description", "")

    for e in local_edges:
        # Normalize relation fields.
        if "c_edge" not in e:
            e["c_edge"] = e.get("predicate") or e.get("relation") or ""

    # Build the local-to-global mapping and add new nodes.
    id_map, new_global_nodes = build_id_map(local_nodes, global_nodes_by_id)

    # ---- Merge nodes without seen_in_frames. ----
    for ln in local_nodes:
        ln_id  = str(ln.get("id"))
        ln_name = ln.get("name", "")
        ln_desc = ln.get("description", "")
        gid = id_map[ln_id]

        if gid in global_nodes_by_id:
            cur = global_nodes_by_id[gid]
            if not isinstance(cur, dict):
                # Fallback: replace an invalid or None historical entry with the full node.
                cur = {"id": gid, "name": ln_name, "description": ln_desc}
                global_nodes_by_id[gid] = cur

            # Fill in the name, replacing the previous name.
            if ln_name:
                cur["name"] = ln_name
            # Update the description only when the new one is richer.
            if len(normalize_text(ln_desc)) > len(normalize_text(cur.get("description", ""))):
                cur["description"] = ln_desc
        else:
            # A new global node contains only three fields.
            global_nodes_by_id[gid] = {"id": gid, "name": ln_name, "description": ln_desc}

    # ---- Merge and deduplicate edges. ----
    # prepare existing edges: map (src,tgt) -> edge index
    pair2idx = {}
    for idx, e in enumerate(global_graph["edges"]):
        src = str(e.get("src")).strip()
        tgt = str(e.get("tgt")).strip()
        pair2idx[(src, tgt)] = idx

    # merge
    for e in local_edges:
        src_g = str(e["src"])
        tgt_g = str(e["tgt"])

        if src_g not in id_map or tgt_g not in id_map:
            continue

        if src_g == "" or tgt_g == "":
            continue

        src_g = id_map[src_g]
        tgt_g = id_map[tgt_g]

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


    # Sort numerically; global IDs are expected to be numeric.
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
            enable_expert_parallel=False,    # Disable expert parallelism on a single GPU.
            gpu_memory_utilization=0.75,     # Avoid out-of-memory errors.
            enforce_eager=False,              # Temporarily True to avoid compile timeout
            # executor="uniproc",            # Keep this option if it is already in use.
        )
    return _GLOBAL_VLLM

def prepare_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8192):
    from qwen_vl_utils import process_vision_info
    import re as _re, json as _json

    # === Build the input. ===
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
# RAG (NEW): window-level SigLIP pooled embedding -> reference guides
# =========================

def load_siglip_window_encoder(model_name: str = "google/siglip-base-patch16-224"):
    """
    Use the SAME family as indexing script: SiglipProcessor + SiglipModel.get_image_features()
    """
    processor = SiglipProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name)
    model.eval()
    return processor, model

def _l2norm_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.maximum(np.sum(ex), 1e-12)

def _uniform_subsample(paths: List[str], max_n: int) -> List[str]:
    n = len(paths)
    if n <= max_n:
        return paths
    idx = np.linspace(0, n - 1, num=max_n)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, n - 1)
    seen, out = set(), []
    for i in idx.tolist():
        if i not in seen:
            out.append(paths[i])
            seen.add(i)
    return out

@torch.inference_mode()
def embed_window_softmax_pool(
    model_bundle,
    rgb_paths: List[str],
    alpha: float = 12.0,
    max_frames: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    """
    SAME pooling as your indexing script:
    - per-frame get_image_features
    - L2 normalize feats
    - mu = mean(feats)
    - score_i = cos(feat_i, mu)
    - w_i = softmax(alpha * score_i)
    - win = sum(w_i * feat_i), L2 normalize
    """
    processor, model = model_bundle
    rgb_paths = _uniform_subsample(rgb_paths, max_frames)

    imgs = [Image.open(p).convert("RGB") for p in rgb_paths]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    feats = model.get_image_features(**inputs)          # (N,D)
    feats = feats.to(torch.float32).cpu().numpy()       # float32
    feats = _l2norm_np(feats)                           # (N,D)

    mu = feats.mean(axis=0, keepdims=True).astype(np.float32)
    mu = _l2norm_np(mu)

    scores = (feats @ mu.T).reshape(-1)                 # cosine since normalized
    w = _softmax_np(alpha * scores).astype(np.float32)  # (N,)

    win = (w[:, None] * feats).sum(axis=0, keepdims=True)
    win = _l2norm_np(win).astype(np.float32)
    return win[0]  # (D,)

def run_rag_for_window_guides(
    query_rgbs: List[str],
    window_index_path: str,
    window_keys_path: str,
    faiss_topk: int = 8,
    model_bundle=None,
    alpha: float = 12.0,
    max_frames: int = 8,
    min_score: float = None,   # optional gating for IP/cos
) -> List[Dict[str, Any]]:
    """
    Returns a list of retrieved guide items:
    [
      {"score": float, "scan_id": str, "win_idx": int, "exp_json_path": str, "guide": str},
      ...
    ]
    """
    if not query_rgbs:
        return []

    # load index + keys
    index = faiss.read_index(window_index_path)
    with open(window_keys_path, "rb") as f:
        keys = pickle.load(f)  # list of dicts per vector

    # device for encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = model_bundle
    model = model.to(device)
    model.eval()

    q = embed_window_softmax_pool(
        model_bundle=(processor, model),
        rgb_paths=query_rgbs,
        alpha=alpha,
        max_frames=max_frames,
        device=device,
    ).astype(np.float32)

    # IndexFlatIP expects normalized vectors for cosine
    D, I = index.search(q.reshape(1, -1), faiss_topk)

    out = []
    for rank in range(int(I.shape[1])):
        idx = int(I[0, rank])
        score = float(D[0, rank])
        if idx < 0 or idx >= len(keys):
            continue
        if (min_score is not None) and (score < float(min_score)):
            continue

        meta = keys[idx]
        exp_path = meta.get("exp_json_path")
        scan_id = meta.get("scan_id")
        win_idx = meta.get("win_idx")

        guide = ""
        try:
            with open(exp_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            guide = (j.get("guide") or "").strip()
        except Exception as e:
            print(f"[WARN] failed to load exp json: {exp_path} err={e}")
            continue

        if not guide:
            continue

        out.append({
            "rank": rank,
            "score": score,
            "scan_id": scan_id,
            "win_idx": int(win_idx),
            "exp_json_path": exp_path,
            "guide": guide,
        })

    with open("/hkfs/work/workspace/scratch/rg5174-ragspace2/data/tmp_window_index_304/temp_exp_rag.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# =========================
# Qwen inference (window-level, with previous_global + ref_edges)
# =========================
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
        # Infer the image type from the extension; JPEG is the default here.
        return f"data:image/jpeg;base64,{base64_data}"

def build_prompt_window_rag(
    window_rgbs: List[str],
    prev_global: Optional[Dict[str, Any]],
    reference_guides: List[str],
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

    guides_str = _json.dumps({"guides": reference_guides or []}, ensure_ascii=False, separators=(",", ":"))
    # print(f"[DEBUG] reference_guides json: {guides_str}")


    # NOTE: classes_txt kept for compatibility (even if not used in guide)
    classes_txt   = ", ".join(classes_list)
    relations_txt = ", ".join(relations_list)

    # --------- main guide (uses tags, no huge inline text) ----------
    guide = f"""
You are a reliable 3D scene graph annotator for indoor scenes.

You are given:
- multiple images of ONE scene window, which contains several images,
- a PREVIOUS global scene graph, it includes some nodes and edges of the scene same as images of this window (provided in <PREV_GRAPH>),
- experience hints from visually similar windows (provided in <REF_GUIDES>),
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

  - Decide whether O corresponds to an OLD node from <PREV_GRAPH> or is NEW:

    You must reuse an old id if there is evidence that O is the SAME physical object of a node in <PREV_GRAPH>.

    Evidence types you MUST consider:
      1) similar relative position (left/right/front/behind/closeby to prominent objects), based on current images and edges in <PREV_GRAPH>,
      2) similar visual cues (color, material, shape, text/logo, subparts) based on current images with candidates' descriptions in <PREV_GRAPH>,
      3) if O is a floor, ceiling or door, they are unique per scene, if there is already one in <PREV_GRAPH>, reuse it, otherwise create a new one,
      4) if O is large and prominent (e.g., bed, sofa, table, cabinet), it is less likely to be new if there is a similar one in <PREV_GRAPH>.


  - Record the decision internally:
      1) All objects in this window that match an existing node id and are visible MUST be included in seen_old_nodes.
      2) For new detected objects, you are only allowed to assign less than 15 new nodes in each window, so choose these clear ones with high confidence.

IMPORTANT:
  - STEP 0 is the ONLY step where OLD vs NEW decisions are made.
  - STEP 1 MUST NOT re-evaluate consistency.


STEP 1 — NODE IDENTIFICATION

In this step, you must do:

  - For each object marked as NEW in STEP 0:
      * Create a new node entry in "new_nodes".
      * Assign a new global id strictly larger than MAX_EXISTING_ID in <PREV_GRAPH>.
      * Assign new ids in increasing order.

Rules:
  - If the object is visually ambiguous or blurry, do NOT create a new node for it
  - "new_nodes" MUST contain ONLY NEW nodes.
  - "seen_old_nodes" MUST contain ONLY ids of reused old nodes.
  - Only less than 15 new nodes can be created in each window, so chosse these clear ones with high confidence.
  - Floor, ceiling, door are unique per scene, only create them if they do not exist in <PREV_GRAPH>.
  - treat component parts (e.g., chair leg, table leg, cabinet door, bicycle wheel) as one single object.
  - treat small cabinets of a large appliance as one cabinet.

Node fields for NEW nodes:
  - "id": global integer ID
  - "name": short noun phrase (must match <ALLOWED_CLASSES>)
  - "description": visual cues (color, material, shape, parts, text/logo, spatial location, relation to surrounding objects, relative size) that help distinguish it from other same-class objects.

Check:
  - do not output floor or ceiling multiple times if they are already in <PREV_GRAPH>, they are unique per scene.


STEP 2 — RELATION PREDICTION

In this step, you must do:
  - First, carefully read <REF_GUIDES_JSON>. It contains experience/strategies extracted from visually similar windows.
  - Second, remember all existing edges from <PREV_GRAPH> and newly detected nodes from STEP 1.
  - Third, based on the images, decide which (src, tgt) pairs have visually or logically supported relations.
  - Fourth, use <REF_GUIDES_JSON> as soft constraints and instructions (DO NOT copy blindly). Choose EXACTLY 3 predicates per edge,
    strictly from <ALLOWED_PREDICATES>.

Rules:
  - Use ONLY predicates from <ALLOWED_PREDICATES>.
  - It is impossible that a node has NO relations with any other nodes, output all visually and logically possible edges.
  - NEW edges can connect:
      * old -> old (only if newly observed/confirmed in this window)
      * old -> new
      * new -> old
      * new -> new
  - If you output an edge (src,tgt), you MUST output EXACTLY 3 predicates:
      "c_edge": ["p1","p2","p3"]
    If only 1 predicate is obvious, choose 2 additional compatible predicates from the <ALLOWED_PREDICATES>
    (e.g., relative position left/right/front/behind/higher/lower when consistent).
  - "close by" alone is NOT enough; add extra spatial predicates when possible.
  - Edges are directed: src -> tgt.

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
    content.append({"type": "text", "text": f"<ALLOWED_PREDICATES>\n{relations_txt}\n</ALLOWED_PREDICATES>"})
    content.append({"type": "text", "text": f"<ALLOWED_CLASSES>\n{classes_txt}\n</ALLOWED_CLASSES>"})
    content.append({"type": "text", "text": f"<REF_GUIDES>\n{guides_str}\n</REF_GUIDES>"})
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

        # Reuse the global LLM singleton instead of creating a new instance.
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
            return None, out_text  # Preserve the raw output.
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
    Run window-level inference through DashScope MultiModalConversation.
    - Input: messages produced by build_prompt_window_rag.
    - Output: (parsed_graph_dict or None, raw_text or "").
    """
    import json as _json

    # Configure the API key through the environment; do not hardcode it.
    # For example, set DASHSCOPE_API_KEY in the shell using your local secret.
    # Then use dashscope.api_key = os.getenv("DASHSCOPE_API_KEY").
    # Alternatively, pass os.environ["DASHSCOPE_API_KEY"] explicitly.
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            model=qwen_model_name,
            messages=messages,
            result_format="message",            # Return an OpenAI-style message.
            response_format={"type": "json_object"},  # Enable JSON mode.
            max_tokens=max_new_tokens,         # Limit the output length.
        )
    except Exception as e:
        print(f"[DashScope] API call failed: {e}")
        return None, ""

    # Check status_code for success as specified by DashScope.
    if getattr(response, "status_code", None) != HTTPStatus.OK:
        print(f"[DashScope] Error code={getattr(response, 'code', None)}, "
              f"message={getattr(response, 'message', None)}")
        return None, ""

    # Normal response path: response.output.choices[0].message.content.
    try:
        content = response.output.choices[0].message.content
    except Exception as e:
        print(f"[DashScope] Unexpected response format: {e}")
        return None, ""

    raw_text = ""
    parsed = None

    # Case 1: content is already a JSON string.
    if isinstance(content, str):
        raw_text = content.strip()
        try:
            parsed = _json.loads(raw_text)
        except Exception:
            parsed = None

    # Case 2: content is a list of text or json_object segments.
    elif isinstance(content, list):
        # Prefer a json_object segment.
        for seg in content:
            if not isinstance(seg, dict):
                continue
            # Some modes return {"type": "json_object", "json": {...}} directly.
            if seg.get("type") == "json_object" and "json" in seg:
                try:
                    parsed = seg["json"]
                    raw_text = _json.dumps(parsed, ensure_ascii=False)
                    break
                except Exception:
                    pass
            # More commonly, the payload is {"type": "text", "text": "<json string>"}.
            if "text" in seg and isinstance(seg["text"], str):
                raw_text_candidate = seg["text"].strip()
                # Remove an optional Markdown JSON code fence.
                tmp = raw_text_candidate.strip()
                tmp = tmp.removeprefix("```json").removesuffix("```").strip()
                try:
                    parsed_candidate = _json.loads(tmp)
                    parsed = parsed_candidate
                    raw_text = raw_text_candidate
                    break
                except Exception:
                    # Retain raw_text on parse failure for outer-layer recovery.
                    if not raw_text:
                        raw_text = raw_text_candidate

        # If parsing failed but raw_text exists, retry parsing the complete text.
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
    siglip_bundle,
    qwen_model,
    qwen_proc,
    window_index_path: str,
    window_keys_path: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
):

    sequence_dir = os.path.join(sequence_root, scan_id, "sequence")
    out_dir = os.path.join(out_root, scan_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) Skip a scan that has already completed. ----
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

    # ---- 2) Resume at the window level by locating the next unfinished window. ----
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

    # ---- 3) Iterate over windows starting at start_win_idx. ----
    for win_idx, (s, e) in enumerate(win_plan):
        if win_idx < start_win_idx:
            # Skip a window completed by an earlier run.
            continue

        win_rgbs = all_rgbs[s:e]
        print(f"\n=== {scan_id} | Window #{win_idx} [{s}..{e}) | {len(win_rgbs)} images ===")

        # --- build query set for RAG (union of prev+curr for win>0; only curr for win0) ---
        if win_idx == 0:
            query_indices = list(range(s, e))
        else:
            ps, pe = win_plan[win_idx - 1]
            cs, ce = s, e
            query_indices = sorted(set(range(ps, pe)) | set(range(cs, ce)))
        query_rgbs = [all_rgbs[i] for i in query_indices if 0 <= i < len(all_rgbs)]

        retrieved = run_rag_for_window_guides(
            query_rgbs=query_rgbs,
            window_index_path=window_index_path,
            window_keys_path=window_keys_path,
            faiss_topk=8,
            model_bundle=siglip_bundle,
            alpha=WIN_RAG_ALPHA,
            max_frames=8,
            min_score=None,
        )

        # Limit token usage by retaining the top three guides.
        reference_guides = [r["guide"] for r in retrieved[:1]]

        messages = build_prompt_window_rag(
            window_rgbs=win_rgbs,
            prev_global=global_graph if win_idx > 0 or start_win_idx > 0 else None,
            reference_guides=reference_guides,
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

    # Write the final global graph after all windows complete.
    atomic_write_json(final_path, global_graph)
    print(f"\n✅ Done. Final global graph saved to: {final_path}")

# =========================
# Batch driver
# =========================
def get_scan_ids(test_list_path: str) -> List[str]:
    with open(test_list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    # Adjust these paths for the local environment.
    test_list_path = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
    sequence_root  = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan"
    out_root       = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/RAG_baseline_Qwen3/RAG_end2end_CoT_exp"
    WINDOW_INDEX_PATH = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/tmp_window_index_304/window_index.faiss"
    WINDOW_KEYS_PATH  = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/tmp_window_index_304/window_keys.pkl"

    os.makedirs(out_root, exist_ok=True)

    print("[Load] SigLIP window encoder ...")
    siglip_bundle = load_siglip_window_encoder("google/siglip-base-patch16-224")

    print("[Load] Qwen model ...")
    qwen_model = None
    qwen_proc  = None
    if not IF_API:
        qwen_proc  = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
        if not IF_vLLM:
            print("[Load] Qwen model ...")
            qwen_model = AutoModelForImageTextToText.from_pretrained(QWEN_MODEL_NAME, dtype=torch.bfloat16, device_map="auto")

    scan_ids = sorted(get_scan_ids(test_list_path))
    for scan_id in scan_ids:
        run_scan_with_rag_merge(
            scan_id=scan_id,
            sequence_root=sequence_root,
            out_root=out_root,
            siglip_bundle=siglip_bundle,
            qwen_model=qwen_model,
            qwen_proc=qwen_proc,
            window_index_path=WINDOW_INDEX_PATH,
            window_keys_path=WINDOW_KEYS_PATH,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )


