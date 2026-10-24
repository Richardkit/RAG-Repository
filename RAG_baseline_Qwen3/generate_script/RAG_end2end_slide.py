import os
import re
import json
import glob
import base64
import dashscope

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
import pickle

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
IF_vLLM      = False
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

def salvage_local_from_raw(raw_text, max_nodes=10, max_edges=30):
    if not raw_text:
        return None
    node_objs = _extract_array_objects(raw_text, "nodes")
    edge_objs = _extract_array_objects(raw_text, "edges")

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
            rel = (e.get("c_edge") or e.get("predicate") or e.get("relationship") or e.get("relation") or "").strip().lower()
            if src in keep_ids and tgt in keep_ids and rel:
                edges.append({"src": src, "tgt": tgt, "c_edge": rel})
        except Exception:
            continue

    if nodes:
        return {"nodes": nodes, "edges": edges}
    return None

def cap_graph(g, max_nodes: int = 10, max_edges: int = 80):
    """
    Keep at most `max_nodes` nodes and `max_edges` edges.
    - Only keep edges whose src/tgt are in the kept node IDs.
    - Normalize edge key to 'c_edge'.
    """
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
                cur = {"id": gid, "name": ln_name, "description": ln_desc}
                global_nodes_by_id[gid] = cur

            # Name completion (force rename)
            if ln_name:
                cur["name"] = ln_name
            # Update description only when the new one is “richer”
            if len(normalize_text(ln_desc)) > len(normalize_text(cur.get("description", ""))):
                cur["description"] = ln_desc
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
        src_g = id_map[str(e["src"])]
        tgt_g = id_map[str(e["tgt"])]
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
            enforce_eager=True,              # As discussed, disable torch.compile/cudagraph
            # executor="uniproc",            # Keep this if you are currently using it
        )
    return _GLOBAL_VLLM

def prepare_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8196):
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
            print(f"[INFO] no candidates for winning scan in query {qlist[0]['query_index']} -> skip")
            continue
        best = min(w, key=lambda x: x["distance"])
        json_path = os.path.join(kb_scan_root, winning_scan, "frame_sg", f"frame-{best['kb_frame_id']:06d}.json")
        if not os.path.isfile(json_path):
            print(f"[INFO] missing frame json -> skip: {json_path}")
            continue
        try:
            with open(json_path, "r") as f:
                sg = json.load(f)
                # print(f"[Debug] sg: {sg}")

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
    # print(f"[Debug] merged_reference_sg: {merged_reference_sg}")
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
) -> Tuple[List[Dict[str, Any]], str]:

    import json as _json
    content: List[Dict[str, Any]] = []

    if IF_API:
        content.append({"video" : [get_base64_image(path) for path in window_rgbs], "fps":2})

    else:
        for p in window_rgbs:
            content.append({"type": "image", "image": p})

    prev_str = _json.dumps(prev_global or {"nodes": [], "edges": []}, ensure_ascii=False, separators=(",", ":"))
    refs_str = _json.dumps({"edges": reference_edges or []}, ensure_ascii=False, separators=(",", ":"))

    classes_txt   = ", ".join(classes_list)
    relations_txt = ", ".join(relations_list)

    guide = f"""You will see a contiguous sequence of RGB images from the same scan (no geometry), a previous built scene graph for same scene is also provided (may be empty).
Return **ONLY** one JSON object (no extra text).
Critically, **nodes and edges are independent top-level arrays**.

OUTPUT SCHEMA (nodes/edges are independent):

{{
  "nodes": [
    {{
      "id": "<GLOBAL_ID>",
      "name": "<short noun phrase>",
      "description": "<rich disambiguating description>"
    }},
    ...
  ],
  "edges": [
    {{
      "src": "<GLOBAL_ID>",
      "tgt": "<GLOBAL_ID>",
      "c_edge": ["<predicate_1>", "<predicate_2>", "<predicate_3>"]
    }},
    ...
  ]
}}

OBJECT ID RULES:
- Each node MUST contain an "id" field that is the GLOBAL object ID.
- Floors, walls, ceilings and curtains are also objects; include them if visible, if there are large furniture, then there are usually walls where they are leaning against, and the wall is not always white.
- The first two given images are also included in PREVIOUS GRAPH, which means every visible object in the first two images has already appeared in PREVIOUS GRAPH.
- Views for previous graph might also appear later in the current window, so it is important to judge if an object is already present in PREVIOUS GRAPH.
- Output all objects that are VISIBLE even partially in ANY of the images in this window.
- See the PREVIOUS GRAPH carefully to find all previously detected objects, based on their "name" and "description" to help you identify them, and reuse their IDs if they are still present in the current window.
- Only when you are VERY SURE that the detected object is a DIFFERENT physical object from all objects in PREVIOUS GRAPH,
  ASSIGN A NEW id to the object, otherwise just reuse the previous id.
- If PREVIOUS GRAPH is empty, you should start ids from 1, 2, 3, ...

- ids MUST:
  * be integers,
  * for a new id, the integer must be strictly larger than ANY id in PREVIOUS GRAPH,
  * be unique within this window,
  * be assigned in increasing order (max_id+1, max_id+2, ...).

NODE WRITING GUIDE:
- "name": short noun phrase (e.g., "white sofa", "wooden dining table").
- "description": rich cues (color, material, shape, parts, text/logo, location in the room, relation to surrounding objects) that help distinguish between multiple similar objects.


INTERPRETATION OF c_edge:
- "c_edge" is a **list of 3 predicates** for the same (src, tgt) pair.
- You MUST output 3 predicates from the RELATIONS list below for each "src"-"tgt" pair.
- Output all edges which you think are visually or logically supported.
- The Number of edges should be at least larger than 3x the number of nodes.

EDGE RULES:
The relation predicate MUST be chosen **exactly** from the following RELATIONS list:
RELATIONS = [{relations_txt}]
- Each edge represents one ordered pair of objects (src -> tgt).
- For each (src, tgt) pair, you MUST use a **single edge object** with "c_edge" as a list of predicates,
  instead of creating multiple separate edges with the same src/tgt.
- "edges" MUST reference only IDs existing in "nodes" for this window (or reused from PREVIOUS GRAPH).
- Edges are **directed**: src -> tgt. For symmetric relations, you may output edges in both directions,
  each with its own "c_edge" list.
- The predicates in "c_edge" MUST be chosen exactly from RELATIONS (case-sensitive string match).
- For each (src, tgt) pair:
  - If there is at least one clear relation, you MUST output **at least 1** predicate in "c_edge".
  - If multiple relations are visually supported (e.g., "standing on" and "supported by"), you MUST include **all of them**
    in "c_edge", up to 3 predicates in total.
- If you see two objects are "close by", a single predicate "close by" is DEFINITELY NOT enough:
  you MUST also add at least one more spatial relation between them in the same "c_edge" list
  (e.g., "left" / "right" / "front" / "behind" / "higher than" / "lower than").
- Some spatial relations need multiple predicates to represent clearly, for example:
  if A is left-front to B, you should output both "front" and "left" in A's "c_edge" toward B.
- objects on the floor should have "standing on" relation to the floor object.
- objects whose surface is touching the wall or other surface should have "attached to" or "hanging on" relation to the wall or other surface, wall should have "attached to" relation with ceiling or floor.
- Do NOT invent new predicates and do NOT use single prepositional words (e.g., "on", "in", "at") as predicates,
  because they are NOT included in RELATIONS.

PREVIOUS GRAPH (may be empty; reuse IDs if you are sure it is the same physical object):
{prev_str}

REF_EDGES (You can take it as extra hints, it indicates some relations that are likely to exist in this scene and between objects):
{refs_str}

Before you output the final JSON, carefully check that:
- All predicate strings in every "c_edge" list are exactly from RELATIONS.
- Each "c_edge" list has 3 predicates.
"""


    content.append({"type": "text", "text": guide})
    return [{"role": "user", "content": content}], refs_str, guide

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

def run_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8196):

        request, in_tokens = prepare_qwen_vllm(messages, qwen_proc, max_new_tokens)

        # * No longer instantiate a new LLM, use the global singleton instead
        llm = get_qwen_vllm()

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            top_k=-1,
            stop_token_ids=[],
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
    # If you insist on passing it manually, you can also set dashscope.api_key=os.environ["DASHSCOPE_API_KEY"] outside
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=os.environ["DASHSCOPE_API_KEY"],
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
    final_path = os.path.join(out_dir, f"{scan_id}_global_graph_ragmerge.json")
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
    for win_idx, (s, e) in enumerate(win_plan):
        if win_idx < start_win_idx:
            # This window was already completed in a previous run, skip directly
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

        print(f"[Debug]reference_edges: {reference_edges}")
        # Delete intermediate files after RAG results are used, to save space
        try:
            if os.path.exists(rag_json_path):
                os.remove(rag_json_path)
        except Exception as e:
            print(f"[WARN] failed to remove RAG json {rag_json_path}: {e}")

        messages, refs_str, guide = build_prompt_window_rag(
            window_rgbs=win_rgbs,
            prev_global=global_graph if win_idx > 0 or start_win_idx > 0 else None,
            reference_edges=reference_edges,
            classes_list=classes_list,
            relations_list=relations_list,
            IF_API=IF_API,
        )

        # Debug: token counts for reference edges and overall input
               # Debug: token counts for reference edges, text and images
        try:
            # 1) token count of the entire text prompt (including guide + prev_graph + refs, etc.)
            full_prompt = qwen_proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            total_text_tokens = len(
                qwen_proc.tokenizer.encode(full_prompt, add_special_tokens=False)
            )

            # 2) count only the tokens for the refs_str JSON segment, as a lower bound for the RAG prompt
            refs_tokens = len(
                qwen_proc.tokenizer.encode(refs_str, add_special_tokens=False)
            )

            print(f"[Debug] refs_str: {refs_str}")

            # 3) run process_vision_info again with Qwen official API to get image / video inputs
            img_inputs, vid_inputs, _ = process_vision_info(
                messages, image_patch_size=16,
                return_video_kwargs=True, return_video_metadata=True
            )

            def estimate_mm_tokens(mm_inputs, patch_size: int = 16) -> int:
                """
                Rough estimate of multimodal token count:
                - for each image, use (H // patch_size) * (W // patch_size) as the patch count
                - compatible with PIL.Image / numpy / torch.Tensor
                """
                if mm_inputs is None:
                    return 0
                total = 0
                for x in mm_inputs:
                    h = w = None

                    # PIL.Image
                    if isinstance(x, Image.Image):
                        w, h = x.size

                    else:
                        arr = None
                        try:
                            import torch
                            if isinstance(x, torch.Tensor):
                                arr = x
                            else:
                                arr = np.array(x)
                        except Exception:
                            arr = None

                        if arr is not None:
                            if arr.ndim == 4:
                                # (B, C, H, W)
                                h, w = arr.shape[-2], arr.shape[-1]
                            elif arr.ndim == 3:
                                # (C, H, W) or (H, W, C)
                                h, w = arr.shape[-2], arr.shape[-1]

                    if h is None or w is None:
                        continue

                    total += int((h // patch_size) * (w // patch_size))

                return total

            image_tokens = estimate_mm_tokens(img_inputs, patch_size=16)
            video_tokens = estimate_mm_tokens(vid_inputs, patch_size=16)

            print(
                f"[Debug] total_text_tokens={total_text_tokens}, "
                f"refs_tokens≈{refs_tokens}, "
                f"image_tokens≈{image_tokens}, "
                f"video_tokens≈{video_tokens}"
            )
        except Exception as dbg_err:
            print(f"[Debug] token count failed: {dbg_err}")



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

        local_graph = cap_graph(local_graph, max_nodes=20, max_edges=80)
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
    # out_root       = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/RAG_end2end_slide_new"
    out_root       = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/RAG_sg_split_new/aggregate"
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
    for scan_id in scan_ids:
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
