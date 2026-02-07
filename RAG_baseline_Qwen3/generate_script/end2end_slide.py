# RAG_whole_scan_sg.py
# End-to-end incremental local Scene Graph (no geometry) with resumable checkpoints.
# - Nodes: id (global), name, description, seen_in_frames
# - Edges: src, tgt, c_edge
# - Sliding window over RGB frames; resume from last completed window if interrupted.

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
import json, re
import glob
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from vllm import LLM, SamplingParams
from dataclasses import asdict
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import re as _re, json as _json

# =========================
# Config
# =========================
CLASSES_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/data_GT/3RScan_3RScan160/classes.txt"
RELATIONS_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/data_GT/3RScan_3RScan160/relationships.txt"
MODEL_NAME = "/hkfs/work/workspace/scratch/rg5174-ragspace/models/Qwen3-VL-32B-Instruct"
# DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 6
STRIDE      = 4  # overlap = WINDOW_SIZE - STRIDE = 2

IF_vLLM = False  # whether to use vllm for inference

# ---- Global LLM cache for vLLM ----
_GLOBAL_VLLM = None

FRAME_RE = re.compile(r"frame-(\d+)\.color\.(?:jpg|jpeg|png)$", re.IGNORECASE)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
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


def _extract_array_objects(raw, key):
    # 在 raw 里找到 "key": [ ... ] 的片段，返回每个完整的 {...} 文本（尽量多）
    m = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if not m: return []
    i = m.end()  # 指向 '[' 后
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
                break  # 数组结束
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
            rel = (e.get("c_edge") or e.get("predicate") or e.get("relationship") or e.get("relation") or "")
            if isinstance(rel, list):
                preds = [str(x).strip().lower() for x in rel if str(x).strip()]
            else:
                s = str(rel).strip()
                preds = [s.lower()] if s else []

            if src in keep_ids and tgt in keep_ids and preds:
                edges.append({"src":src,"tgt":tgt,"c_edge":preds})

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

    # 1) nodes 截断
    nodes = (g.get("nodes") or [])[:max_nodes]

    # 允许 id 是 int/str，统一转成 str
    keep_ids = {str(n.get("id")) for n in nodes if n.get("id") is not None}

    # 2) edges 过滤 & 截断
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

def get_qwen_vllm():
    """
    Lazily create a single vLLM LLM instance and reuse it
    across all windows / scans within this python process.
    """
    global _GLOBAL_VLLM
    if _GLOBAL_VLLM is None:
        print("[vLLM] Initializing Qwen3-VL-32B engine once ...")
        _GLOBAL_VLLM = LLM(
            model=MODEL_NAME,
            mm_encoder_tp_mode="data",
            tensor_parallel_size=max(torch.cuda.device_count(), 1),
            enable_expert_parallel=False,    # 单卡禁止 EP
            gpu_memory_utilization=0.9,     # 避免 OOM
            max_model_len=100000,              # 需要处理长上下文
            enforce_eager=True,              # 之前讨论过，关掉 torch.compile/cudagraph
            # executor="uniproc",            # 如果你现在用这个，也可以保留
        )
    return _GLOBAL_VLLM

# =========================
# Model Inference
# =========================

def build_prompt_for_window(window_rgbs, classes_list, relations_list, prev_graph):
    import json as _json
    content = [{"type": "image", "image": p} for p in window_rgbs]
    classes_txt   = ", ".join(classes_list)
    relations_txt = ", ".join(relations_list)

    # 最小上下文：仅注入已有 nodes/edges，用于 ID 复用
    prev_graph_txt = ""
    if prev_graph:
        prev_min = {
            "nodes": [
                {"id": n["id"], "name": n.get("name",""), "description": n.get("description","")}
                for n in (prev_graph.get("nodes") or [])
            ],
            "edges": prev_graph.get("edges", []),
        }
        prev_graph_txt = _json.dumps(prev_min, ensure_ascii=False, separators=(",", ":"))

    # --------- main guide (uses tags, no huge inline text) ----------
    guide = f"""
You are a reliable 3D scene graph annotator for indoor scenes.

You are given:
- multiple images of ONE scene window, which contains several images,
- a PREVIOUS global scene graph (provided in <PREV_GRAPH>),
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
      C = C = all existing nodes whose node names match O's object class or are semantically similar.

  - Decide whether O corresponds to an OLD node or is NEW:

    You must reuse an old id ONLY if there is evidence that O is the
    SAME physical object as a candidate in C.

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
    - First, remember all existing edges from <PREV_GRAPH> and newly detected nodes from Step 1.
    - Third, based on the images and the nodes generated from Step 1, decide which (src, tgt) pairs have visually or logically supported relations, 
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
    content.append({"type": "text", "text": guide})
    content.append({"type": "text", "text": f"<PREV_GRAPH>\n{prev_graph_txt}\n</PREV_GRAPH>"})
    content.append({"type": "text", "text": f"<ALLOWED_PREDICATES>\n{relations_txt}\n</ALLOWED_PREDICATES>"})
    content.append({"type": "text", "text": f"<ALLOWED_CLASSES>\n{classes_txt}\n</ALLOWED_CLASSES>"})

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
            src = str(e.get("src") or oid).strip()   # 若省略src，默认就是本对象
            tgt = str(e.get("tgt","")).strip()
            
            rel = (e.get("relationship") or e.get("c_edge") or e.get("predicate") or "")
            if isinstance(rel,list):
                preds = [str(x).strip().lower() for x in rel if str(x).strip()]
            else:
                s = str(rel).strip()
                preds = [s.lower()] if s else []
            if src and tgt and preds:
                edges.append({"src": src, "tgt": tgt, "c_edge": preds})
    return {"nodes": nodes, "edges": edges}


@torch.no_grad()
def run_qwen(messages, qwen_model, qwen_proc, max_new_tokens: int = 8196):

    # === 构建输入 ===
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

    # === 生成 ===
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
        return None, out_text  # ← 保留原始输出
    try:
        return _json.loads(m.group(0)), out_text
    except Exception:
        return None, out_text

def prepare_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8196):
    from qwen_vl_utils import process_vision_info
    import re as _re, json as _json

    # === 构建输入 ===
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
    

def run_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8196):

        request, in_tokens = prepare_qwen_vllm(messages, qwen_proc, max_new_tokens)

        # ★ 不再 new LLM，而是拿全局单例
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
            return None, out_text  # ← 保留原始输出
        try:
            return _json.loads(m.group(0)), out_text
        except Exception:
            return None, out_text

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
    - 不再根据 name/description 做相似度匹配
    - 完全相信模型生成的 node["id"] 已经是「它认为的 global id」
    - 若该 id 已存在于 global_nodes_by_id，则视为同一物体，直接复用
    - 若该 id 不存在：
        * 如果 id 是整数且大于当前已用最大 id，则直接用这个 id 作为新 global id
        * 否则，为避免冲突，分配一个新的递增 id（next_id），并将 local id 映射到它
    """
    id_map = {}
    new_global_nodes = []

    # 统计目前全局已经占用的整数 id
    existing_int_ids = []
    for gid in global_nodes_by_id.keys():
        v = _to_int_safe(gid)
        if v is not None:
            existing_int_ids.append(v)
    max_existing = max(existing_int_ids) if existing_int_ids else 0
    next_id = max_existing + 1

    used_ids = set(global_nodes_by_id.keys())  # 字符串形式

    for node in local_nodes:
        lid_raw = node.get("id")
        lid = str(lid_raw).strip()
        name = node.get("name", "")
        desc = node.get("description", "")

        # 情况 1：模型给的 id 已经存在于 global graph → 直接当成同一个物体
        if lid in global_nodes_by_id:
            id_map[lid] = lid
            # 如果你希望更新一下全局描述，这里也可以 merge 一下 description
            # 例如：把新的 desc append 进去（可选）
            continue

        # 情况 2：这个 id 还没用过
        lid_int = _to_int_safe(lid)
        if lid_int is not None and lid_int > max_existing and lid not in used_ids:
            # 模型给的是一个「比当前所有 id 都大的整数」，认为是一个合理的新 global id
            g_id = lid
            used_ids.add(g_id)
            # 同时更新 max_existing / next_id，避免之后冲突
            if lid_int > max_existing:
                max_existing = lid_int
                next_id = max_existing + 1
        else:
            # 模型给了一个已经用过的小整数 / 非整数 id / 其他奇怪形式，
            # 为了安全起见，我们分配一个真正唯一的 new_id
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
        # 只保留必须字段
        n["name"] = n.get("name", "")
        n["description"] = n.get("description", "")

    for e in local_edges:
        # 统一关系字段
        if "c_edge" not in e:
            e["c_edge"] = e.get("predicate") or e.get("relation") or ""

    # 建立 local->global 映射 + 新增节点
    id_map, new_global_nodes = build_id_map(local_nodes, global_nodes_by_id)

    # ---- 合并节点（无 seen_in_frames）----
    for ln in local_nodes:
        ln_id  = str(ln.get("id"))
        ln_name = ln.get("name", "")
        ln_desc = ln.get("description", "")
        gid = id_map[ln_id]

        if gid in global_nodes_by_id:
            cur = global_nodes_by_id[gid]
            if not isinstance(cur, dict):
                # 兜底：若历史是异常/None，直接用完整节点覆盖
                cur = {"id": gid, "name": ln_name, "description": ln_desc}
                global_nodes_by_id[gid] = cur

            # 名称补全（强制改名）
            if ln_name:
                cur["name"] = ln_name
            # 描述更“丰富”时才更新
            if len(normalize_text(ln_desc)) > len(normalize_text(cur.get("description", ""))):
                cur["description"] = ln_desc
        else:
            # 新全局节点：只三字段
            global_nodes_by_id[gid] = {"id": gid, "name": ln_name, "description": ln_desc}

    # ---- 合并边并去重 ----
    # prepare existing edges: map (src,tgt) -> edge index
    pair2idx = {}
    for idx, e in enumerate(global_graph["edges"]):
        src = str(e.get("src")).strip()
        tgt = str(e.get("tgt")).strip()
        pair2idx[(src, tgt)] = idx

    # merge
    for e in local_edges:
        
        src_key = str(e.get("src", "")).strip()
        tgt_key = str(e.get("tgt", "")).strip()

        src_g = id_map.get(src_key, "")
        tgt_g = id_map.get(tgt_key, "")

        # 如果模型把 id 写成了 "wall"/"floor"/"sofa" 这类字符串，直接跳过，避免 KeyError
        if not src_g or not tgt_g:
            # 可选：只打印前几条，便于你定位是哪条输出有问题
            # print(f"[WARN] skip edge with unknown endpoint: src={src_key}, tgt={tgt_key}")
            continue


        if src_g == "" or tgt_g == "":
            continue
        preds = e["c_edge"]

        key = (src_g, tgt_g)
        if key in pair2idx:
            i = pair2idx[key]
            old_preds = global_graph["edges"][i]["c_edge"]
            merged = list(dict.fromkeys(old_preds + preds))  # union (preserve order)
            global_graph["edges"][i]["c_edge"] = merged
        else:
            global_graph["edges"].append({
                "src": src_g,
                "tgt": tgt_g,
                "c_edge": preds
            })
            pair2idx[key] = len(global_graph["edges"]) - 1


    # 数字ID排序（你已确保全局ID为数字）
    global_graph["nodes"] = sorted(global_nodes_by_id.values(), key=lambda x: int(x["id"]))
    return global_graph

# =========================
# Pipeline (resumable)
# =========================
def prepare_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

def window_paths(out_dir: str, scan_id: str, win_idx: int) -> Dict[str, str]:
    base = os.path.join(out_dir, f"{scan_id}")
    return {
        "lock":   f"{base}_win{win_idx:03d}.lock",
        "localW": f"{base}_local_win{win_idx:03d}.work",
        "globalW":f"{base}_global_win{win_idx:03d}.work",
        "local":  f"{base}_local_win{win_idx:03d}.json",
        "global": f"{base}_global_win{win_idx:03d}.json",
        "raw":    f"{base}_local_win{win_idx:03d}.raw.txt",
    }

def find_last_completed_window(out_dir: str, scan_id: str) -> int:
    """Return last completed window index, or -1 if none."""
    pattern = os.path.join(out_dir, f"{scan_id}_global_sg_win*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return -1
    # parse largest idx
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
                stride: int = STRIDE):

    prepare_out_dir(out_dir)

    final_path = os.path.join(out_dir, f"{scan_id}_global_graph_final.json")
    if os.path.exists(final_path):
        print(f"[Skip] {scan_id} already completed -> {final_path}")
        return

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
        # trust existing plan to keep determinism
        saved = safe_read_json(plan_path)
        if saved and len(saved) == len(win_plan):
            print(f"[Resume] Using existing window plan for {scan_id} ({len(win_plan)} windows)")
        else:
            print(f"[Warn] Plan exists but mismatched; using existing file as source of truth.")
            win_plan = [(d["start"], d["end"]) for d in saved]

    classes_list = load_vocab_list(CLASSES_TXT)
    relations_list = load_vocab_list(RELATIONS_TXT)

    # load model
    qwen_proc = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    if not IF_vLLM:
        print("[Load] Qwen model ...")
        qwen_model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto", local_files_only=True)


    # initialize / resume global graph
    global_graph = {"nodes": [], "edges": []}
    last_done = find_last_completed_window(out_dir, scan_id)
    if last_done >= 0:
        # load the last completed global graph snapshot
        last_global = os.path.join(out_dir, f"{scan_id}_global_sg_win{last_done:03d}.json")
        gg = safe_read_json(last_global)
        if gg:
            global_graph = gg
            print(f"[Resume] Loaded global graph from window #{last_done}")

    # process windows
    start_win = last_done + 1
    for win_idx in range(start_win, len(win_plan)):
        s, e = win_plan[win_idx]
        win_rgbs = all_rgbs[s:e]
        paths = window_paths(out_dir, scan_id, win_idx)

        # if final json exists -> already done
        if os.path.exists(paths["global"]) and os.path.exists(paths["local"]):
            print(f"[Skip] Window #{win_idx} already done.")
            # update global_graph from file to stay in sync (optional)
            gg = safe_read_json(paths["global"])
            if gg:
                global_graph = gg
            continue

        # if lock exists but json not exists -> previous crash; we just rerun
        # create lock to mark "in-progress"
        with open(paths["lock"], "w") as f:
            f.write("in-progress")

        print(f"\n=== {scan_id} | Window #{win_idx} [{s}..{e}) | {len(win_rgbs)} images ===")
        # build prompt with previous global_graph
        messages = build_prompt_for_window(win_rgbs, classes_list=classes_list, relations_list=relations_list, prev_graph=global_graph if win_idx > 0 else None)



        # === inference ===
        if IF_vLLM:
            parsed, raw_text = run_qwen_vllm(messages, qwen_proc, max_new_tokens=8196)
        else:
            parsed, raw_text = run_qwen(messages, qwen_model, qwen_proc, max_new_tokens=8196)

        if parsed is None:
            print("[WARN] Model did not return valid JSON. Saving raw output and attempting salvage...")
            # 先保存原文
            try:
                with open(paths["raw"], "w", encoding="utf-8") as f:
                    f.write(raw_text or "")
            except Exception as _e:
                print(f"[WARN] failed to save raw output: {_e}")
            # 抢救：从原文截取前10/30
            recovered = salvage_local_from_raw(raw_text, max_nodes=20, max_edges=80)
            if not recovered:
                # 保持原有逻辑：跳过，但有.raw可排查
                continue
            local_graph = recovered
        else:
            local_graph = parsed
            with open(paths["raw"], "w", encoding="utf-8") as f:
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

        atomic_write_json(paths['local'], local_graph)

        global_graph = merge_graphs(global_graph, local_graph)
        atomic_write_json(paths['global'], global_graph)

        print(f"[OK] Window #{win_idx} merged. Global nodes={len(global_graph.get('nodes',[]))}, edges={len(global_graph.get('edges',[]))}")

    # 所有 window 跑完，写最终 global
    atomic_write_json(final_path, global_graph)
    print(f"\n✅ Done. Final global graph saved to: {final_path}")

# =========================
# Batch driver
# =========================
if __name__ == "__main__":
    def get_scan_ids(test_list_path):
        with open(test_list_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    test_list_path = "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
    out_root = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/RAG_Qwen3/end2end_slide_new"
    os.makedirs(out_root, exist_ok=True)

    scan_ids = sorted(get_scan_ids(test_list_path))
    for scan_id in scan_ids:
        out_dir = os.path.join(out_root, scan_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        sequence_dir = os.path.join("/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan", scan_id, "sequence")
        main_stream(sequence_dir=sequence_dir,
                    scan_id=scan_id,
                    out_dir=out_dir,
                    window_size=WINDOW_SIZE,
                    stride=STRIDE)
