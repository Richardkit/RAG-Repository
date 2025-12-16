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
from vllm import LLM, SamplingParams
from dataclasses import asdict
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import re as _re, json as _json

# =========================
# Config
# =========================
CLASSES_TXT = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/classes.txt"
RELATIONS_TXT = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/relationships.txt"
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 6
STRIDE      = 4  # overlap = WINDOW_SIZE - STRIDE = 2

NAME_WEIGHT = 0.6
DESC_WEIGHT = 0.4
SIM_THRESHOLD = 0.62

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

def mixed_similarity(name_a: str, desc_a: str, name_b: str, desc_b: str) -> float:
    return NAME_WEIGHT * jaccard_token_sim(name_a, name_b) + DESC_WEIGHT * jaccard_token_sim(desc_a, desc_b)

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
            gpu_memory_utilization=0.75,     # 避免 OOM
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


INTERPRETATION OF c_edge:
- "c_edge" is a **list of 3 predicates** for the same (src, tgt) pair.
- You MUST output 3 predicates from the RELATIONS list below for each "src"-"tgt" pair.

STRONG RULES:
- Output **as many nodes and relations as visible**.
- Floors, walls and curtains are also objects; include them if visible.
- The Number of edges should be larger than 1.5x the number of nodes.

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
- Do NOT invent new predicates and do NOT use single prepositional words (e.g., "on", "in", "at") as predicates,
  because they are NOT included in RELATIONS.

NODE WRITING GUIDE:
- "name": short noun phrase (e.g., "white sofa", "wooden dining table").
- "description": rich cues (color, material, shape, parts, text/logo, location in the room,
  relation to surrounding objects) that help distinguish between multiple similar objects.
- Output all objects as nodes, even if you are not sure about their class or they are partially visible.


PREVIOUS GRAPH (may be empty; reuse IDs if you are sure it is the same physical object):
{prev_graph_txt}

Before you output the final JSON, carefully check that:
- All predicate strings in every "c_edge" list are exactly from RELATIONS.
- Each "c_edge" list has between 1 and 3 predicates, multiple predicates help to describe the relation more clearly.
- You can check if all objects are included as nodes by considering the following questions:
    1) Are all pixels in the image covered by at least one object?
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
# Graph merge
def build_id_map(current_nodes, global_nodes_by_id):
    """
    将本窗口局部节点映射到全局节点：
    - 先尝试与已有全局节点做文本相似度匹配（>=SIM_THRESHOLD则复用其ID）
    - 否则为“本窗口新节点”分配**递增的数字ID**（不会重复为"1"）
    """
    from collections import defaultdict

    id_map = {}
    new_global_nodes = []

    # 1) 统计现有全局最大数字ID，准备递增分配
    def _to_int(s):
        s = str(s)
        return int(s) if s.isdigit() else None
    existing_int_ids = [_to_int(k) for k in global_nodes_by_id.keys()]
    existing_int_ids = [x for x in existing_int_ids if x is not None]
    next_id = (max(existing_int_ids) + 1) if existing_int_ids else 1  # 起始数字ID

    # 2) 建立按名称的粗索引（加速相似度匹配）
    index_by_name = defaultdict(list)
    for gid, n in global_nodes_by_id.items():
        index_by_name[normalize_text(n.get("name", ""))].append((gid, n))

    # 3) 对每个局部节点：相似度复用 or 分配新数字ID
    for n in (current_nodes or []):
        lid  = str(n.get("id", "")).strip()
        name = n.get("name", "")
        desc = n.get("description", "")

        # 3.1 相似度候选：同名优先 + 全表补充
        candidates = []
        same_name_key = normalize_text(name)
        if same_name_key in index_by_name:
            candidates.extend(index_by_name[same_name_key])
        exist = set(g for g, _ in candidates)
        for g, obj in global_nodes_by_id.items():
            if g not in exist:
                candidates.append((g, obj))

        # 3.2 选最佳匹配（若分数足够就复用）
        best_gid, best_score = None, -1.0
        for gid, obj in candidates:
            s = mixed_similarity(name, desc, obj.get("name", ""), obj.get("description", ""))
            if s > best_score:
                best_gid, best_score = gid, s

        if best_gid is not None and best_score >= SIM_THRESHOLD:
            # 复用已有全局ID
            id_map[lid] = best_gid
        else:
            # 3.3 分配**新的数字ID**（递增）
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
        src_g = id_map[str(e["src"])]
        tgt_g = id_map[str(e["tgt"])]
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
        "localW": f"{base}_local_sg_win{win_idx:03d}.work",
        "globalW":f"{base}_global_sg_win{win_idx:03d}.work",
        "local":  f"{base}_local_sg_win{win_idx:03d}.json",
        "global": f"{base}_global_sg_win{win_idx:03d}.json",
        "raw":    f"{base}_local_sg_win{win_idx:03d}.raw.txt",
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
    qwen_proc = AutoProcessor.from_pretrained(MODEL_NAME)
    if not IF_vLLM:
        print("[Load] Qwen model ...")
        qwen_model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto")


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
            recovered = salvage_local_from_raw(raw_text, max_nodes=20, max_edges=60)
            if not recovered:
                # 保持原有逻辑：跳过，但有.raw可排查
                continue
            local_graph = recovered
        else:
            local_graph = parsed

        # 无论成功/抢救，统一做一次“上限截断”（防止超量）
        local_graph = cap_graph(local_graph, max_nodes=10, max_edges=30)

        # （如果你的模型偶尔输出 object-centric）
        if "objects" in (local_graph or {}):
            local_graph = objects_to_nodes_edges(local_graph)
            local_graph = cap_graph(local_graph, max_nodes=10, max_edges=30)

        # write local work, then promote
        atomic_write_json(paths["localW"], local_graph)

        # merge and snapshot global
        global_graph = merge_graphs(global_graph, local_graph)
        atomic_write_json(paths["globalW"], global_graph)

        # promote work -> final jsons
        os.replace(paths["localW"],  paths["local"])
        os.replace(paths["globalW"], paths["global"])

        # remove lock to mark completion
        if os.path.exists(paths["lock"]):
            os.remove(paths["lock"])

        print(f"[OK] Window #{win_idx} finished -> {os.path.basename(paths['global'])}")

    # final
    atomic_write_json(final_path, global_graph)
    print(f"\n✅ Done. Final graph saved to: {final_path}")

# =========================
# Batch driver
# =========================
if __name__ == "__main__":
    def get_scan_ids(test_list_path):
        with open(test_list_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    test_list_path = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_validation_subset_20pct.txt"
    out_root = "/hkfs/work/workspace/scratch/rg5174-myspace/data/end2end_slide_new"
    os.makedirs(out_root, exist_ok=True)

    scan_ids = sorted(get_scan_ids(test_list_path))
    for scan_id in scan_ids[20:]:
        out_dir = os.path.join(out_root, scan_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        sequence_dir = os.path.join("/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan", scan_id, "sequence")
        main_stream(sequence_dir=sequence_dir,
                    scan_id=scan_id,
                    out_dir=out_dir,
                    window_size=WINDOW_SIZE,
                    stride=STRIDE)
