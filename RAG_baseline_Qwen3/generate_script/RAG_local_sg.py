# RAG_whole_scan_sg.py
# End-to-end incremental local Scene Graph (no geometry) with resumable checkpoints.
# - Nodes: id (global), name, description, seen_in_frames
# - Edges: src, tgt, c_edge
# - Sliding window over RGB frames; resume from last completed window if interrupted.

import os
import re
import json, re
import glob
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# =========================
# Config
# =========================
# === Vocab files (hard constraints) ===
CLASSES_TXT = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/classes.txt"
RELATIONS_TXT = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/relationships.txt"
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


WINDOW_SIZE = 6
STRIDE      = 4  # overlap = WINDOW_SIZE - STRIDE = 2

NAME_WEIGHT = 0.6
DESC_WEIGHT = 0.4
SIM_THRESHOLD = 0.62

FRAME_RE = re.compile(r"frame-(\d+)\.color\.(?:jpg|jpeg|png)$", re.IGNORECASE)

# =========================
# IO utils (atomic + resume)
# =========================
def atomic_write_json(path: str, obj: Any):
    """Write JSON atomically to avoid partial files."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic on POSIX

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
        m = FRAME_RE.search(os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(paths, key=_idx)

def plan_windows(n: int, win: int, stride: int) -> List[Tuple[int, int]]:
    """Return list of [start_idx, end_idx) on frame indices (0-based)."""
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

def build_vocab_sets() -> Tuple[List[str], List[str], set, set]:
    cls_list = load_vocab_list(CLASSES_TXT)         # Original case preserved (displayed in prompts)
    rel_list = load_vocab_list(RELATIONS_TXT)
    cls_set  = {normalize_text(x) for x in cls_list}  # Used for code filtering (lowercased)
    rel_set  = {normalize_text(x) for x in rel_list}
    return cls_list, rel_list, cls_set, rel_set

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
    # Find the "key": [ ... ] fragment in raw, return each complete {...} text (as many as possible)
    m = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if not m: return []
    i = m.end()  # points to after '['
    objs, depth, start = [], 0, None
    while i < len(raw):
        ch = raw[i]
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(raw[start:i+1])
                start = None
        elif ch == ']':
            if depth == 0:
                break  # end of array
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
            nid = str(n.get("id","")).strip()
            if not nid: continue
            nodes.append({"id": nid, "name": n.get("name",""), "description": n.get("description","")})
        except Exception:
            continue

    keep_ids = {str(n["id"]) for n in nodes}
    for t in edge_objs:
        if len(edges) >= max_edges: break
        try:
            e = json.loads(t)
            src = str(e.get("src","")).strip()
            tgt = str(e.get("tgt","")).strip()
            rel = (e.get("c_edge") or e.get("predicate") or e.get("relationship") or e.get("relation") or "").strip().lower()
            if src in keep_ids and tgt in keep_ids and rel:
                edges.append({"src": src, "tgt": tgt, "c_edge": rel})
        except Exception:
            continue

    if nodes:
        return {"nodes": nodes, "edges": edges}
    return None

def cap_graph(g, max_nodes=10, max_edges=30):
    if not isinstance(g, dict):
        return {"nodes": [], "edges": []}
    nodes = (g.get("nodes") or [])[:max_nodes]
    # Keep only edges that reference existing nodes
    keep_ids = {str(n.get("id")) for n in nodes}
    edges = []
    for e in (g.get("edges") or []):
        src = str(e.get("src", ""))
        tgt = str(e.get("tgt", ""))
        if src in keep_ids and tgt in keep_ids:
            edges.append({"src": src, "tgt": tgt, "c_edge": (e.get("c_edge") or e.get("predicate") or e.get("relationship") or e.get("relation") or "").strip().lower()})
        if len(edges) >= max_edges:
            break
    return {"nodes": nodes, "edges": edges}


# =========================
# Model Inference
# =========================

def build_prompt_for_window(window_rgbs, classes_list, relations_list):
    content = [{"type": "image", "image": p} for p in window_rgbs]

    classes_txt   = ", ".join(classes_list)
    relations_txt = ", ".join(relations_list)

    guide = f"""You will see a contiguous sequence of RGB images from the same scan (no geometry).
Return ONLY one JSON object (no extra text).
Critically, nodes and edges are independent top-level arrays.

OUTPUT SCHEMA:
{{
  "nodes": [
    {{"id":"<LOCAL_ID_INTEGER>","name":"<class from CLASSES>","description":"<rich disambiguating description>"}},
    ...
  ],
  "edges": [
    {{"src":"<LOCAL_ID_INTEGER>","tgt":"<LOCAL_ID_INTEGER>","c_edge":"<predicate from RELATIONS>"}},
    ...
  ]
}}

HARD CONSTRAINTS:
- IDs MUST be **integers starting from 1** within this window (e.g., 1,2,3...). **Do NOT use any prefix like 'node_1'**.
- Object 'name' MUST be chosen from CLASSES only. If no suitable class exists, omit that object.
- You should describe each object in 'description' to help distinguish it from others of the same class, using attributes, colors, relative positions, etc. Avoid giving exactly the same description to two different objects when you can see any difference (position, size, color, context). Only if they look truly identical and indistinguishable in this window, similar descriptions are acceptable.
- Edge 'c_edge' MUST be chosen from RELATIONS only. Do not invent predicates.
- Output at most 20 nodes and 40 edges for this window.
- No geometry fields; directed edges (src->tgt); for symmetric relations (e.g., "next to"), output both directions if confident.

CLASSES (allowed):
{classes_txt}

RELATIONS (allowed, lowercase):
{relations_txt}
"""
    content.append({"type": "text", "text": guide})
    return [{"role": "user", "content": content}]



def objects_to_nodes_edges(j):
    """Convert object-centric JSON to {'nodes':[], 'edges':[]} (no geometry)."""
    if not isinstance(j, dict):
        return {"nodes": [], "edges": []}
    objs = j.get("objects") or []
    nodes, edges = [], []
    for o in objs:
        if not isinstance(o, dict): 
            continue
        oid = str(o.get("id","")).strip()
        if not oid: 
            continue
        name = o.get("name","")
        desc = o.get("description","")
        nodes.append({"id": oid, "name": name, "description": desc})
        for e in (o.get("edges") or []):
            if not isinstance(e, dict): 
                continue
            src = str(e.get("src") or oid).strip()   # If src is omitted, default to the current object
            tgt = str(e.get("tgt","")).strip()
            rel = (e.get("relationship") or e.get("c_edge") or e.get("predicate") or "").strip().lower()
            if src and tgt and rel:
                edges.append({"src": src, "tgt": tgt, "c_edge": rel})
    return {"nodes": nodes, "edges": edges}


@torch.no_grad()
def run_qwen(messages, qwen_model, qwen_proc, max_new_tokens: int = 8196):
    from qwen_vl_utils import process_vision_info
    import re as _re, json as _json

    # === Build input ===
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
        do_resize=False, padding=True, return_tensors="pt"
    ).to(qwen_model.device)

    input_ids_len = inputs.input_ids.shape[1]
    max_ctx = getattr(qwen_model.config, "max_position_embeddings", None)
    if not isinstance(max_ctx, int) or max_ctx <= 0:
        max_ctx = getattr(qwen_proc.tokenizer, "model_max_length", 32768)

    print(f"[Diag] tokens_in(text_only)={text_tokens}, tokens_in(actual)={input_ids_len}, "
          f"max_ctx={max_ctx}, images={len(image_inputs)}")

    # === Generation ===
    out_ids = qwen_model.generate(**inputs, max_new_tokens= max_new_tokens)
    new_tokens = out_ids.shape[1] - inputs.input_ids.shape[1]
    total_seq = out_ids.shape[1]
    print(f"[Diag] tokens_out(new)={new_tokens}, total_seq_len={total_seq}")

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out_text = qwen_proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if isinstance(out_text, list):
        out_text = "\n".join(out_text)

    m = _re.search(r"\{[\s\S]*\}", out_text)
    if not m:
        return None, out_text  # ← keep the original output
    try:
        return _json.loads(m.group(0)), out_text
    except Exception:
        return None, out_text


# =========================
# Graph merge
def build_id_map(current_nodes, global_nodes_by_id):
    """
    Map local nodes in this window to global nodes:
    - First try text similarity matching with existing global nodes (if >=SIM_THRESHOLD, reuse their ID)
    - Otherwise assign an **incrementing numeric ID** (never repeating “1”) to “new nodes in this window”
    """
    from collections import defaultdict

    id_map = {}
    new_global_nodes = []

    # 1) Count the current maximum global numeric ID, prepare for incremental allocation
    def _to_int(s):
        s = str(s)
        return int(s) if s.isdigit() else None
    existing_int_ids = [_to_int(k) for k in global_nodes_by_id.keys()]
    existing_int_ids = [x for x in existing_int_ids if x is not None]
    next_id = (max(existing_int_ids) + 1) if existing_int_ids else 1  # starting numeric ID

    # 2) Build a coarse index by name (to speed up similarity matching)
    index_by_name = defaultdict(list)
    for gid, n in global_nodes_by_id.items():
        index_by_name[normalize_text(n.get("name", ""))].append((gid, n))

    # 3) For each local node: reuse by similarity or assign a new numeric ID
    for n in (current_nodes or []):
        lid  = str(n.get("id", "")).strip()
        name = n.get("name", "")
        desc = n.get("description", "")

        # 3.1 Similarity candidates: same-name first + supplement with full list
        candidates = []
        same_name_key = normalize_text(name)
        if same_name_key in index_by_name:
            candidates.extend(index_by_name[same_name_key])
        exist = set(g for g, _ in candidates)
        for g, obj in global_nodes_by_id.items():
            if g not in exist:
                candidates.append((g, obj))

        # 3.2 Select the best match (reuse if score is sufficient)
        best_gid, best_score = None, -1.0
        for gid, obj in candidates:
            s = mixed_similarity(name, desc, obj.get("name", ""), obj.get("description", ""))
            if s > best_score:
                best_gid, best_score = gid, s

        if best_gid is not None and best_score >= SIM_THRESHOLD:
            # Reuse existing global ID
            id_map[lid] = best_gid
        else:
            # 3.3 Assign a **new numeric ID** (incrementing)
            new_id = str(next_id)
            next_id += 1
            id_map[lid] = new_id
            new_global_nodes.append({
                "id": new_id,
                "name": name,
                "description": desc
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

            # Name completion (do not force rename)
            if not cur.get("name") and ln_name:
                cur["name"] = ln_name
            # Update description only when the new one is “richer”
            if len(normalize_text(ln_desc)) > len(normalize_text(cur.get("description", ""))):
                cur["description"] = ln_desc
        else:
            # New global node: only three fields
            global_nodes_by_id[gid] = {"id": gid, "name": ln_name, "description": ln_desc}

    # ---- Merge edges and deduplicate ----
    existing = set((e["src"], e["tgt"], e.get("c_edge", "")) for e in global_graph["edges"])
    for e in local_edges:
        src = id_map.get(str(e.get("src")))
        tgt = id_map.get(str(e.get("tgt")))
        pred = (e.get("c_edge") or "").strip().lower()
        if not src or not tgt or not pred:
            continue
        key = (src, tgt, pred)
        if key not in existing:
            existing.add(key)
            global_graph["edges"].append({"src": src, "tgt": tgt, "c_edge": pred})

    # Sort by numeric ID (global IDs are ensured to be numbers)
    global_graph["nodes"] = sorted(global_nodes_by_id.values(), key=lambda x: int(x["id"]))
    return global_graph

def renumber_local_ids(local_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize node ids in the local window to pure numeric strings 1..N,
    and rewrite src/tgt of edges using the same mapping.
    Numbers are assigned in the current order of nodes.
    """
    nodes = list(local_graph.get("nodes") or [])
    edges = list(local_graph.get("edges") or [])

    # 1) Build a mapping from “original id string” -> “1..N” for nodes
    id_map = {}
    new_nodes = []
    next_id = 1
    for n in nodes:
        old = str(n.get("id","")).strip()
        if not old:
            continue
        if old not in id_map:
            id_map[old] = str(next_id)
            next_id += 1
        new_nodes.append({
            "id": id_map[old],
            "name": n.get("name",""),
            "description": n.get("description",""),
        })

    keep_ids = set(id_map.values())

    # 2) Rewrite edge endpoints; keep only edges where both endpoints exist and are not self-loops
    new_edges = []
    for e in edges:
        s_old = str(e.get("src","")).strip()
        t_old = str(e.get("tgt","")).strip()
        rel   = (e.get("c_edge") or e.get("predicate") or e.get("relationship") or e.get("relation") or "").strip().lower()
        s = id_map.get(s_old)
        t = id_map.get(t_old)
        if s and t and s != t and rel:
            new_edges.append({"src": s, "tgt": t, "c_edge": rel})

    return {"nodes": new_nodes, "edges": new_edges}

# =========================
# Pipeline (resumable)
# =========================
def prepare_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

def window_paths(out_dir: str, scan_id: str, win_idx: int) -> Dict[str, str]:
    base = os.path.join(out_dir, f"{scan_id}")
    return {
        "lock":   f"{base}_win{win_idx:03d}.lock",
        "localW": f"{base}_local_sg_win{win_idx:03d}.work",
        "local":  f"{base}_local_sg_win{win_idx:03d}.json",
        "raw":    f"{base}_local_sg_win{win_idx:03d}.raw.txt",
    }

def find_last_completed_local_window(out_dir: str, scan_id: str) -> int:
    pattern = os.path.join(out_dir, f"{scan_id}_local_sg_win*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return -1
    last = -1
    for p in files:
        m = re.search(r"_win(\d{3})\.json$", p)
        if m:
            last = max(last, int(m.group(1)))
    return last


def main_stream(sequence_dir: str,
                scan_id: str,
                out_dir: str,
                window_size: int = WINDOW_SIZE,
                stride: int = STRIDE,
                qwen_model=None,
                qwen_proc=None):

    prepare_out_dir(out_dir)

    # load frames
    all_rgbs = list_sequence_rgbs(sequence_dir)
    if not all_rgbs:
        raise FileNotFoundError(f"No RGBs found under: {sequence_dir}")
    print(f"[Info] {scan_id}: {len(all_rgbs)} frames")

    # plan windows and persist the plan
    win_plan = plan_windows(len(all_rgbs), window_size, stride)
    plan_path = os.path.join(out_dir, f"{scan_id}_windows.json")
    if not os.path.exists(plan_path):
        atomic_write_json(plan_path, [{"win_idx": i, "start": s, "end": e} for i, (s, e) in enumerate(win_plan)])
    else:
        saved = safe_read_json(plan_path)
        if saved and len(saved) == len(win_plan):
            print(f"[Resume] Using existing window plan for {scan_id} ({len(win_plan)} windows)")
        else:
            print(f"[Warn] Plan exists but mismatched; using existing file as source of truth.")
            win_plan = [(d["start"], d["end"]) for d in saved]

    # load vocab once
    classes_list = load_vocab_list(CLASSES_TXT)
    relations_list = load_vocab_list(RELATIONS_TXT)

    # resume from last completed local window
    last_done = find_last_completed_local_window(out_dir, scan_id)
    start_win = last_done + 1

    for win_idx in range(start_win, len(win_plan)):
        s, e = win_plan[win_idx]
        win_rgbs = all_rgbs[s:e]
        paths = window_paths(out_dir, scan_id, win_idx)

        # if local json exists -> already done
        if os.path.exists(paths["local"]):
            print(f"[Skip] Window #{win_idx} already done.")
            continue

        # mark in-progress
        with open(paths["lock"], "w") as f:
            f.write("in-progress")

        print(f"\n=== {scan_id} | Window #{win_idx} [{s}..{e}) | {len(win_rgbs)} images ===")

        # build prompt (no previous global graph)
        messages = build_prompt_for_window(win_rgbs, classes_list=classes_list, relations_list=relations_list)

        # inference
        parsed, raw_text = run_qwen(messages, qwen_model, qwen_proc, max_new_tokens=8192)

        if parsed is None:
            print("[WARN] Model did not return valid JSON. Saving raw output and attempting salvage...")
            try:
                with open(paths["raw"], "w", encoding="utf-8") as f:
                    f.write(raw_text or "")
            except Exception as _e:
                print(f"[WARN] failed to save raw output: {_e}")
            recovered = salvage_local_from_raw(raw_text, max_nodes=10, max_edges=30)
            if not recovered:
                continue
            local_graph = recovered
        else:
            local_graph = parsed

        # cap 10/30
        local_graph = cap_graph(local_graph, max_nodes=10, max_edges=30)

        # Compatible with object-centric format
        if "objects" in (local_graph or {}):
            local_graph = objects_to_nodes_edges(local_graph)
            local_graph = cap_graph(local_graph, max_nodes=10, max_edges=30)

        # **Key**: Renumber all ids uniformly to 1..N; also update edges accordingly
        local_graph = renumber_local_ids(local_graph)

        # Write to disk (local only)
        atomic_write_json(paths["localW"], local_graph)
        os.replace(paths["localW"], paths["local"])

        # Remove the lock file
        if os.path.exists(paths["lock"]):
            os.remove(paths["lock"])

        print(f"[OK] Window #{win_idx} finished -> {os.path.basename(paths['local'])}")


# =========================
# Batch driver
# =========================
if __name__ == "__main__":
    def get_scan_ids(test_list_path):
        with open(test_list_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    test_list_path = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_validation_subset_20pct.txt"
    out_root = "/hkfs/work/workspace/scratch/rg5174-myspace/data/RAG_sg_split_new/split"

    # load model
    print("[Load] Qwen model ...")
    qwen_model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, dtype="auto", device_map="auto")
    qwen_proc  = AutoProcessor.from_pretrained(MODEL_NAME)
    os.makedirs(out_root, exist_ok=True)

    scan_ids = sorted(get_scan_ids(test_list_path))
    for scan_id in scan_ids:
        out_dir = os.path.join(out_root, scan_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        sequence_dir = os.path.join("/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan", scan_id, "sequence")
        main_stream(sequence_dir=sequence_dir,
                    scan_id=scan_id,
                    out_dir=out_dir,
                    window_size=WINDOW_SIZE,
                    stride=STRIDE,
                    qwen_model=qwen_model,
                    qwen_proc=qwen_proc)
