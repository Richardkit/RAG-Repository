"""
The scripts executes the retrospective experience abstraction process with several steps:
"""
import os
import re
import json
import glob
import base64
import dashscope
import pickle
import shutil
import colorsys
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from copy import deepcopy
# from vllm import LLM, SamplingParams
import re as _re, json as _json
import numpy as np
from PIL import Image
import faiss
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams

# =========================
# Config
# =========================

CLASSES_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/data_GT/3RScan_3RScan160/classes.txt"
RELATIONS_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/data_GT/3RScan_3RScan160/relationships.txt"
QWEN_MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
ORINAL_DATASET_PATH = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan"
TEMP_IMAGE_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_Rexplore_Qwen3/temp_images"
VALID_SCANS_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
OUT_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/RAG_Rexplore_Qwen3"

WIN_SIZE = 6
STRIDE = 4
FRAME_RE_SEQ = re.compile(r"frame-(\d+)\.color\.(?:jpg|jpeg|png)$", re.IGNORECASE)


# ---- Global LLM cache for vLLM ----
_GLOBAL_VLLM = None
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# =========================
# Windows Tools
# =========================

def atomic_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

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

def list_sequence_rgbs(sequence_dir: str) -> List[str]:
    paths = glob.glob(os.path.join(sequence_dir, "frame-*.color.jpg")) \
          + glob.glob(os.path.join(sequence_dir, "frame-*.color.png")) \
          + glob.glob(os.path.join(sequence_dir, "frame-*.color.jpeg"))
    def _idx(p):
        m = FRAME_RE_SEQ.search(os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(paths, key=_idx)

def load_window_level_gt_sg(scan_id: str, window: List[int], win_idx: Optional[int]=None):

    nodes_ids = set()
    window_level_edges = set()
    node_bboxes = []

    with open(os.path.join(ORINAL_DATASET_PATH, scan_id,f"{scan_id}_gt_bboxes.json"), "r") as f:
        gt_bboxes = json.load(f)

    for i in range(window[0], window[1]):

        path = os.path.join(ORINAL_DATASET_PATH, scan_id, "frame_sg", f"frame-{i:06d}.json")

        if not os.path.exists(path):
            print(f"Warning: {path} not exists!")
            continue

        with open(path, "r") as f:
            frame_sg = json.load(f)
        for node in frame_sg["objects"]:
            nodes_ids.add((node["id"], node["name"]))
            for edge in node["edges"]:
                window_level_edges.add((edge["src"], edge["tgt"], edge["src_label"], edge["tgt_label"], edge["relationship"]))

    for node_id in nodes_ids:

        single_node_info = gt_bboxes.get(scan_id, {}).get(str(node_id[0]), {})
        seen_frames = single_node_info.get("seen_frames", {})
        # print(f"node_id: {node_id}, seen_frames: {seen_frames}")
        window_seen_frames_idx = [(seen_frames.index(f"frame-{i:06d}"),i) for i in range(window[0], window[1]) if f"frame-{i:06d}" in seen_frames]
        window_node= {"id": node_id[0], "name": node_id[1], "bboxes": [ [0,0,0,0] for _ in range(window[1] - window[0]) ]}

        for seen_frame_idx in window_seen_frames_idx:

            bbox = single_node_info.get("bounding_boxes",[])[seen_frame_idx[0]]  # get the bbox using the index of seen frame in seen_frames
            # print(f"node_id: {node_id}, seen_frame_idx: {seen_frame_idx[0]}, frame_idx: {seen_frame_idx[1]}, length_bboxes: {len(window_node['bboxes'])},window_seen_frame_idx: {window_seen_frames_idx}")
            window_node["bboxes"][seen_frame_idx[1] - window[0]] = bbox
            # print(f"window_node: {window_node}")

        node_bboxes.append(window_node)


    # Convert edge tuples back to dictionaries for JSON serialization
    edges_list = [
        {"src": e[0], "tgt": e[1], "src_label": e[2], "tgt_label": e[3], "predicate": e[4]}
        for e in window_level_edges
    ]

    with open(os.path.join(OUT_DIR, scan_id, "gt_sg", f"window_{win_idx}_gt_sg.json"), "w") as f: 
        json.dump({"nodes":node_bboxes, "edges": edges_list}, f, indent=2)    


def prepare_images_with_selection(scan_id, window_level_gt_sg_path, window,
                                  temp_img_dir, min_area=1000):
    
    with open(window_level_gt_sg_path, "r") as f:
        window_level_gt_sg = json.load(f)
    nodes = window_level_gt_sg["nodes"]
    edges = window_level_gt_sg["edges"]

    ori_images_paths = [os.path.join(ORINAL_DATASET_PATH, scan_id, "sequence", f"frame-{i:06d}.color.jpg") for i in range(window[0], window[1])]
            
    def idx_to_color(idx: int) -> tuple:
        # Golden-ratio hue cycling to generate many distinct colors
        h = (idx * 0.61803398875) % 1.0
        s = 0.75
        v = 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    for ori_image_path, frame_id in zip(ori_images_paths, range(window[0], window[1])):

        if not os.path.exists(ori_image_path):
            print(f"Warning: {ori_image_path} not exists!")
            continue
            
        img = Image.open(ori_image_path).convert("RGB")

        draw = ImageDraw.Draw(img)
        label_boxes = []  # track placed label boxes to avoid overlap

        def find_non_overlapping_pos(x, y, text, font, margin=2):
            # Shift label downward until it no longer overlaps existing labels
            tx1, ty1, tx2, ty2 = draw.textbbox((x, y), text, font=font)
            height = ty2 - ty1

            def overlaps(b1, b2):
                x1, y1, x2, y2 = b1
                a1, b1_, a2, b2_ = b2
                return not (x2 < a1 or a2 < x1 or y2 < b1_ or b2_ < y1)

            while any(overlaps((tx1, ty1, tx2, ty2), b) for b in label_boxes):
                ty1 += height + margin
                ty2 += height + margin
            return tx1, ty1, tx2, ty2

        for color_idx, node in enumerate(nodes):
            
            # print(f"frame_id - window[0]: {frame_id - window[0]}")

            bbox = node["bboxes"][frame_id - window[0]]
            node_id = node["id"]
            node_name = node["name"]

            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < min_area:
                continue

            color = idx_to_color(color_idx)

            # Draw bounding box
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=3)

            # Draw node ID and name on the upper edge of the bounding box; avoid overlaps
            text = f"{node_id}: {node_name}"
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=20)
            tx1, ty1, tx2, ty2 = find_non_overlapping_pos(bbox[0], bbox[1], text, font)
            label_boxes.append((tx1, ty1, tx2, ty2))
            draw.text((tx1 + 2, ty1), text, fill=color, font=font)

        temp_image_path = os.path.join(temp_img_dir, f"{scan_id}_frame_{frame_id:06d}.jpg")
        
        # Ensure parent directory exists before saving
        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
        
        print(f"Saving annotated image to {temp_image_path}")
        img.save(temp_image_path)

    return [os.path.join(temp_img_dir, f"{scan_id}_frame_{frame_id:06d}.jpg") for frame_id in range(window[0], window[1])]

# =========================
# vllm Prepare & Run
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
            enforce_eager=True,              # Disable torch.compile and CUDA graphs as previously configured.
            # executor="uniproc",            # Keep this option if it is already in use.
        )
    return _GLOBAL_VLLM

def prepare_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8196):
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


def run_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8196):

        request, in_tokens = prepare_qwen_vllm(messages, qwen_proc, max_new_tokens)

        # Reuse the global LLM singleton instead of creating a new instance.
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
            return None, out_text  # Preserve the raw output.
        try:
            return _json.loads(m.group(0)), out_text
        except Exception:
            return None, out_text
        
# =========================
# Local Scene Graph Generation
# =========================  

def generate_local_scene_graph(window: List[int],
                               scan_id: str,
                               qwen_proc,
                               win_idx: int) -> Dict[str, Any]:
    """
    First inference pass: predict only edges from annotated window images
    containing bounding boxes and object IDs. Reuse window-level GT nodes.

    Requirements:
      - OUT_DIR/scan_id/gt_sg/window_{win_idx}_gt_sg.json exists.
      - TEMP_IMAGE_DIR contains the annotated JPEGs for this window.
      - Global definitions are available for:
            - run_qwen_vllm(messages, qwen_proc, max_new_tokens)
            - qwen_proc = AutoProcessor.from_pretrained(...)
      - RELATIONS_TXT lists one relation class per line.
    Outputs:
      - Write under OUT_DIR/scan_id:
          * {scan_id}_window_{win_idx}_local_pred_sg.json (GT nodes, predicted edges)
          * {scan_id}_window_{win_idx}_local_raw.txt (raw model output)
      - Return the local_graph dictionary.
    """
    global OUT_DIR, TEMP_IMAGE_DIR, RELATIONS_TXT

    # ---------- 1) Prepare paths. ----------
    out_dir = os.path.join(OUT_DIR, scan_id)
    os.makedirs(out_dir, exist_ok=True)

    win_gt_sg_path = os.path.join(
        OUT_DIR, scan_id, "gt_sg", f"window_{win_idx}_gt_sg.json"
    )
    if not os.path.exists(win_gt_sg_path):
        raise FileNotFoundError(f"Window GT SG not found: {win_gt_sg_path}")

    # ---------- 2) Load window-level GT nodes as the fixed object set. ----------
    with open(win_gt_sg_path, "r") as f:
        win_gt_sg = json.load(f)
    gt_nodes: List[Dict[str, Any]] = win_gt_sg.get("nodes", [])

    if not gt_nodes:
        print(f"[WARN] {scan_id} window {win_idx}: empty GT nodes.")

    # Prepare a compact object list containing IDs and names for the prompt.
    obj_lines = [f"- id {n['id']}: {n.get('name', '')}" for n in gt_nodes]
    objects_txt = "\n".join(obj_lines) if obj_lines else "(no objects?)"

    # ---------- 3) Load relation classes. ----------
    relations_list: List[str] = []
    with open(RELATIONS_TXT, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                relations_list.append(line)
    relations_txt = ", ".join(relations_list)

    # ---------- 4) Collect annotated images for this window. ----------
    annotated_rgbs: List[str] = []
    for frame_id in range(window[0], window[1]):
        img_path = os.path.join(
            TEMP_IMAGE_DIR, f"{scan_id}_frame_{frame_id:06d}.jpg"
        )
        if os.path.exists(img_path):
            annotated_rgbs.append(img_path)
        else:
            print(f"[WARN] annotated image not found: {img_path}")

    if not annotated_rgbs:
        raise RuntimeError(
            f"No annotated images found for {scan_id} window {win_idx}"
        )

    # ---------- 5) Build the Qwen multi-image and text prompt. ----------
    content: List[Dict[str, Any]] = []
    for p in annotated_rgbs:
        content.append({"type": "image", "image": p})

    guide = f"""
You are a reliable 3D scene graph annotator for indoor RGB panoramas.

In each image, all visible objects are ALREADY DETECTED and marked with:
  - a bounding box, and
  - a text label of the form "id: name".
These ids are the FINAL object ids for this window. You MUST NOT create new ids.

Your task in this step:
  - ONLY infer spatial relations (edges) between these existing objects.
  - DO NOT change the set of objects and DO NOT invent new objects.

AVAILABLE OBJECTS IN THIS WINDOW (id -> name):
{objects_txt}

ALLOWED RELATIONS (you MUST choose from this list; exact string match):
[{relations_txt}]

OUTPUT REQUIREMENT:
Return ONLY one JSON object, with the following structure:

{{
  "nodes": [],
  "edges": [
    {{
      "src": <INT>,          // object id from the list above
      "tgt": <INT>,          // object id from the list above
      "c_edge": ["predicate1", "predicate2", "predicate3"]
    }},
    ...
  ]
}}

STRICT RULES:
  - "nodes" MUST be an empty list [] because object detection is already fixed.
  - "src" and "tgt" MUST be chosen ONLY from the ids listed above.
  - For each (src, tgt) pair, you MAY output at most one edge object.
  - Every "c_edge" MUST be a list of EXACTLY 3 predicates.
  - Each predicate MUST come from the ALLOWED RELATIONS list and match its spelling exactly.
  - If there is no clear relation between two objects, you MAY omit that (src, tgt) pair.
  - However, for typical indoor scenes, the number of edges is often several times the number of nodes.
    You should cover all clearly visible and logically supported relations (left/right, front/behind, attached to, etc.).
  - Objects on the floor (e.g., chair, table) often have relation A "standing on" floor.

Before you output the final JSON, carefully check that:
  - "nodes" is exactly [].
  - Every edge has integer "src" and "tgt" ids from the AVAILABLE OBJECTS list.
  - Every "c_edge" list has exactly 3 predicates, all from the ALLOWED RELATIONS list, a predicate out of the list is not allowed.
"""

    content.append({"type": "text", "text": guide})
    messages = [{"role": "user", "content": content}]

    # ---------- 6) Call Qwen through vLLM. ----------
    # run_qwen_vllm(...) and qwen_proc must already be defined elsewhere.
    local_json_path = os.path.join(
        out_dir, "pred_sg", f"window_{win_idx}_local_pred_sg.json"
    )
    raw_txt_path = os.path.join(
        out_dir, "pred_sg", f"window_{win_idx}_local_raw.txt"
    )

    try:
        parsed, raw_text = run_qwen_vllm(
            messages, qwen_proc, max_new_tokens=4096
        )
    except Exception as e:
        print(f"[ERROR] Qwen vLLM inference failed on {scan_id} win {win_idx}: {e}")
        parsed, raw_text = None, ""

    # Save raw output for later recovery and debugging.
    try:
        with open(raw_txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text or "")
    except Exception as _e:
        print(f"[WARN] failed to save raw output: {_e}")

    # ---------- 7) Parse and assemble the local graph. ----------
    if not isinstance(parsed, dict):
        print(f"[WARN] Parsed result is None or not dict for {scan_id} win {win_idx}, use empty edges.")
        pred_edges: List[Dict[str, Any]] = []
    else:
        pred_edges = parsed.get("edges", [])
        if not isinstance(pred_edges, list):
            pred_edges = []

    # local_graph uses GT nodes and model-predicted edges.
    local_graph = {
        "nodes": gt_nodes,
        "edges": pred_edges,
    }

    atomic_write_json(local_json_path, local_graph)
    print(
        f"[OK] Local SG generated for {scan_id} window {win_idx}: "
        f"{len(gt_nodes)} nodes, {len(pred_edges)} pred edges"
    )

    return local_graph

# =========================
# Retrospective Experience Abstraction
# =========================  
def extract_guide_from_raw(raw_text: str) -> str:
    """
    Extract the JSON "guide" string from the raw model output.
    - Return an empty string when no guide can be found.
    - Locate the first string literal following "guide": using manual parsing.
    """
    if not raw_text:
        return ""

    key = '"guide"'
    key_idx = raw_text.find(key)
    if key_idx == -1:
        return ""

    colon_idx = raw_text.find(":", key_idx)
    if colon_idx == -1:
        return ""

    # Locate the opening quote of the value.
    first_quote = raw_text.find('"', colon_idx + 1)
    if first_quote == -1:
        return ""

    # Scan manually, handling escaped quotes.
    i = first_quote + 1
    escaped = False
    buf = []

    while i < len(raw_text):
        ch = raw_text[i]
        if escaped:
            buf.append(ch)
            escaped = False
        else:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                # An unescaped quote terminates the string.
                break
            else:
                buf.append(ch)
        i += 1

    return "".join(buf).strip()


def generate_experience_abstraction(window: List[int],
                                    scan_id: str,
                                    qwen_proc,
                                    win_idx: int) -> Optional[Dict[str, Any]]:
    """
    Second inference pass: reflect and abstract experience using the window-level
    GT scene graph and the local scene graph predicted by the first pass.

    Inputs:
      - Annotated images with boxes and IDs: TEMP_IMAGE_DIR/{scan_id}_frame_*.jpg
      - GT graph: OUT_DIR/scan_id/gt_sg/window_{win_idx}_gt_sg.json
      - Predicted graph: OUT_DIR/scan_id/pred_sg/window_{win_idx}_local_pred_sg.json

    Outputs:
      - Save JSON to OUT_DIR/scan_id/window_{win_idx}_abstraction.json:
        {
          "scan_id": "...",
          "window_index": int,
          "guide": "<4-8 English sentences>"
        }
      - Save raw LLM output to window_{win_idx}_abstraction_raw.txt.
      - Return the parsed dictionary or None.
    """
    print("start generate_experience_abstraction")
    out_dir = os.path.join(OUT_DIR, scan_id)

    # ---------- 1) Load the GT and predicted scene graphs. ----------
    gt_sg_path = os.path.join(out_dir, "gt_sg", f"window_{win_idx}_gt_sg.json")
    pred_sg_path = os.path.join(out_dir, "pred_sg", f"window_{win_idx}_local_pred_sg.json")

    if not os.path.exists(gt_sg_path):
        print(f"[ERROR] GT SG not found for {scan_id} window {win_idx}: {gt_sg_path}")
        return None
    if not os.path.exists(pred_sg_path):
        print(f"[ERROR] Pred SG not found for {scan_id} window {win_idx}: {pred_sg_path}")
        return None

    with open(gt_sg_path, "r") as f:
        gt_sg_raw = json.load(f)
    with open(pred_sg_path, "r") as f:
        pred_sg_raw = json.load(f)

    gt_nodes = []
    for n in gt_sg_raw.get("nodes", []):
        gt_nodes.append({
            "id": n.get("id"),
            "name": n.get("name", "")
            # Do not retain bboxes.
        })

    gt_edges = gt_sg_raw.get("edges", [])

    pred_nodes = []
    for n in pred_sg_raw.get("nodes", []):
        pred_nodes.append({
            "id": n.get("id"),
            "name": n.get("name", "")
        })

    pred_edges = pred_sg_raw.get("edges", [])

    # ---------- 2) Prepare object and edge lists for LLM reflection. ----------
    # Object list containing IDs and names.
    node_map = {}
    for n in gt_nodes:
        nid = n.get("id")
        if nid is not None:
            node_map[nid] = n.get("name", "")

    objects_txt_lines = [
        f"- id {nid}: {name}" for nid, name in sorted(node_map.items(), key=lambda x: x[0])
    ]
    objects_txt = "\n".join(objects_txt_lines) if objects_txt_lines else "(no objects)"

    # GT edge list.
    gt_edge_lines = []
    for e in gt_edges:
        src = e.get("src")
        tgt = e.get("tgt")
        rel = e.get("predicate")
        if isinstance(rel, list):
            rel_str = ", ".join(rel)
        else:
            rel_str = str(rel)
        gt_edge_lines.append(
            f"- ({src} -> {tgt})  [{node_map.get(src, '?')} -> {node_map.get(tgt, '?')}]  predicates: {rel_str}"
        )
    gt_edges_txt = "\n".join(gt_edge_lines) if gt_edge_lines else "(no GT edges)"

    # Predicted edge list.
    pred_edge_lines = []
    for e in pred_edges:
        src = e.get("src")
        tgt = e.get("tgt")
        rel_list = e.get("c_edge", [])
        if isinstance(rel_list, list):
            rel_str = ", ".join(rel_list)
        else:
            rel_str = str(rel_list)
        pred_edge_lines.append(
            f"- ({src} -> {tgt})  [{node_map.get(src, '?')} -> {node_map.get(tgt, '?')}]  predicates: {rel_str}"
        )
    pred_edges_txt = "\n".join(pred_edge_lines) if pred_edge_lines else "(no predicted edges)"

    # ---------- 3) Collect annotated window images as additional visual context. ----------
    annotated_rgbs: List[str] = []
    for frame_id in range(window[0], window[1]):
        img_path = os.path.join(
            TEMP_IMAGE_DIR, f"{scan_id}_frame_{frame_id:06d}.jpg"
        )
        if os.path.exists(img_path):
            annotated_rgbs.append(img_path)

    # ---------- 4) Build the CoT-style reflection and abstraction prompt. ----------
    content: List[Dict[str, Any]] = []
    for p in annotated_rgbs:
        content.append({"type": "image", "image": p})

    guide = f"""
You are an expert analyst for 3D scene graph generation.

Your goal in this step is to perform a retrospective reflection and experience abstraction
for a SINGLE window of an indoor scan, based on:

1) The list of objects (with global ids and names).
2) The GROUND-TRUTH edges.
3) The PREDICTED edges produced by the current model.

----------------------------------------
OBJECTS (id -> name)
----------------------------------------
{objects_txt}

----------------------------------------
GROUND-TRUTH EDGES
----------------------------------------
Each line is: (src -> tgt)  [src_name -> tgt_name]  predicates: ...
{gt_edges_txt}

----------------------------------------
PREDICTED EDGES
----------------------------------------
Each line is: (src -> tgt)  [src_name -> tgt_name]  predicates: ...
{pred_edges_txt}

----------------------------------------
INTERNAL REASONING STEPS (DO NOT OUTPUT)
----------------------------------------

You MUST internally follow these steps:

Step 1 — Reflection:
  - Carefully compare the predicted edges with the ground-truth edges.
  - Identify:
      * which kinds of edges and object pairs are predicted correctly and WHY,
      * which ground-truth edges are missing and WHY they may have been missed,
      * which predicted edges are incorrect or hallucinated and possible reasons.
  - When you mention any object or edge in your reasoning, you MAY refer to it as:
      "object <id> (<name>)", e.g. "object 3 (bathtub)".

Step 2 — Structured Reflection + General Guide (Focus on improving edge prediction):
  - Based on your reflection, you MUST produce TWO different kinds of information:

    (A) A structured REFLECTION that is SPECIFIC to this window:
        - "success_cases":
            2–4 sentences, describing which types of edges and object pairs
            were predicted correctly and WHY. You may use concrete examples like
            "object 17 (shower curtain) -> object 6 (ceiling)".
        - "failure_cases":
            2–4 sentences, describing which kinds of edges were missed or wrong,
            again using 1–2 representative examples with ids and names.
        - "error_causes":
            2–4 sentences, analyzing WHY these failures happened
            (e.g., occlusion, ambiguous depth, small objects, cluttered layout).
        - "cues_to_check":
            2–4 sentences, summarizing which visual or geometric cues
            the model should pay more attention to (e.g., contact points, overlap,
            relative height, clear separation vs. occlusion).

    (B) A GENERAL GUIDE that can be re-used for FUTURE scenes:
        - "guide": a concise cohesive paragraph (about 4–8 sentences)
          that describes general, reusable advice for edge prediction.
        - This guide MUST:
            * NOT mention any specific object ids.
            * Use only category-level phrases such as
              "objects such as shower curtains", "large fixtures such as bathtubs".
            * Focus on what cues to check and how to avoid repeating similar errors,
              in scenes that may be similar but not identical to this one.
            * Should start with "To improve spatial relation prediction in similar indoor scenes, the model should ..."
          For example:
            - "When an object such as a curtain partially occludes a large fixture such as a bathtub,
               the model should check depth ordering and contact with walls or floors
               before choosing predicates like 'attached to', 'supported by' or 'standing on."

  - The REFLECTION part may contain specific ids and names.
  - The GUIDE part MUST be fully general, without any id numbers.

You MUST perform all reasoning internally. You MUST NOT output your intermediate analysis.

----------------------------------------
FINAL OUTPUT FORMAT (MANDATORY)
----------------------------------------

You MUST output ONLY ONE JSON object with the following keys:

{{
  "reflection": {{
    "success_cases": "<2-4 sentences, may mention ids+names>",
    "failure_cases": "<2-4 sentences, may mention ids+names>",
    "error_causes": "<2-4 sentences>",
    "cues_to_check": "<2-4 sentences>"
  }},
  "guide": "<4-8 sentences, general reusable advice WITHOUT any ids>"
}}

STRICT REQUIREMENTS:
  - Do NOT output any explanations outside the JSON.
  - Do NOT add extra keys.
  - The JSON MUST be the only content in your response.
"""



    content.append({"type": "text", "text": guide})
    messages = [{"role": "user", "content": content}]

    abs_json_path = os.path.join(out_dir, "exp", f"window_{win_idx}_abstraction.json")
    raw_txt_path = os.path.join(out_dir, "exp", f"window_{win_idx}_abstraction_raw.txt")

    # ---------- 5) Call Qwen through vLLM. ----------
    try:
        parsed, raw_text = run_qwen_vllm(messages, qwen_proc, max_new_tokens=4096)
    except Exception as e:
        print(f"[ERROR] Qwen vLLM abstraction failed on {scan_id} win {win_idx}: {e}")
        parsed, raw_text = None, ""

    # Save the raw text for debugging.
    try:
        with open(raw_txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text or "")
    except Exception as _e:
        print(f"[WARN] failed to save abstraction raw output: {_e}")

    # ---------- 6) Parse the output and retain only the guide. ----------
    guide_text = ""

    if isinstance(parsed, dict) and isinstance(parsed.get("guide"), str):
        # 1) Prefer the guide field from the parsed JSON.
        guide_text = parsed["guide"].strip()
    else:
        # 2) Try extracting the "guide": "..." value from raw_text manually.
        guide_text = extract_guide_from_raw(raw_text or "")
        if guide_text:
            print(f"[INFO] Extracted 'guide' from raw_text for {scan_id} win {win_idx}.")
        else:
            # 3) As a final fallback, use all of raw_text as the guide.
            guide_text = (raw_text or "").strip()
            print(
                f"[WARN] Failed to get 'guide' from parsed/raw for {scan_id} win {win_idx}, "
                f"using full raw_text as guide."
            )

    # Store only scan_id, window_index, and guide in the final object.
    abstraction_obj = {
        "scan_id": scan_id,
        "window_index": win_idx,
        "guide": guide_text,
    }

    atomic_write_json(abs_json_path, abstraction_obj)
    print(f"[OK] Abstraction saved for {scan_id} window {win_idx}")

    return abstraction_obj


# =========================
# Main Function
# =========================  

def main():

    # load valid scans
    with open(VALID_SCANS_TXT, "r") as f:
        valid_scans = [line.strip() for line in f if line.strip()]

    # load all knowledge base scans
    all_scans = os.listdir(ORINAL_DATASET_PATH)
    base_scans = [scan_id for scan_id in all_scans if scan_id not in valid_scans]
    qwen_proc = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
    
    for scan_id in sorted(base_scans):

        print(f"Processing scan:{scan_id}")

        sequence_dir = os.path.join(ORINAL_DATASET_PATH, scan_id, "sequence")
        out_dir = os.path.join(OUT_DIR, scan_id)
        os.makedirs(out_dir, exist_ok=True)

        # 1) generate windows
        all_rgbs = list_sequence_rgbs(sequence_dir)
        if not all_rgbs:
            raise ValueError(f"No RGB images found in {sequence_dir}")
        print(f"[Info] {scan_id}: {len(all_rgbs)} frames")

        win_plan = plan_windows(len(all_rgbs), WIN_SIZE, STRIDE)
        plan_path = os.path.join(out_dir, f"{scan_id}_windows.json")
        if not os.path.exists(plan_path):
            atomic_write_json(plan_path, [{"win_idx": i, "start": s, "end": e} for i, (s, e) in enumerate(win_plan)])

        # skip existing windows and scans
        if os.path.exists(os.path.join(out_dir, "exp", f"window_{len(win_plan)-1}_abstraction.json")):
            print(f"[Info] {scan_id}: all windows exist, skip!")
            continue

        if not os.path.exists(TEMP_IMAGE_DIR):
            os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

        if not os.path.exists(os.path.join(out_dir, "gt_sg")):
            os.makedirs(os.path.join(out_dir, "gt_sg"), exist_ok=True)

        if not os.path.exists(os.path.join(out_dir, "pred_sg")):
            os.makedirs(os.path.join(out_dir, "pred_sg"), exist_ok=True)
        
        if not os.path.exists(os.path.join(out_dir, "exp")):
            os.makedirs(os.path.join(out_dir, "exp"), exist_ok=True)

        for win_idx, window in enumerate(win_plan,start=0):

            if os.path.exists(os.path.join(out_dir, "exp", f"window_{win_idx}_abstraction.json")):
                print(f"[Info] {scan_id} window {win_idx}: exists, skip!")
                continue

            load_window_level_gt_sg(scan_id, window, win_idx=win_idx)
            annotated_paths = prepare_images_with_selection(scan_id,
                                          os.path.join(out_dir, "gt_sg", f"window_{win_idx}_gt_sg.json"),
                                          window,
                                          TEMP_IMAGE_DIR,
                                          min_area=1000)

            generate_local_scene_graph(window,
                                       scan_id,
                                       qwen_proc,
                                       win_idx=win_idx)
            
            generate_experience_abstraction(window, scan_id, qwen_proc, win_idx)

            for p in annotated_paths:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception as e:
                    print(f"[Warn] failed to remove temp image: {p} err={e}")
if __name__ == "__main__":
    main()





        
