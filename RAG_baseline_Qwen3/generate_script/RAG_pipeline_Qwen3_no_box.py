import os
import re
import json
import glob
import sys
import random
import cv2
import numpy as np
from PIL import Image
import faiss
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from transformers import SiglipImageProcessor, SiglipVisionModel
import torch
import pickle


from collections import Counter, defaultdict
from copy import deepcopy

def list_random_k_sequence_depths(sequence_dir: str, rgb_paths: List[str]) -> List[str]:
    """
    For each RGB image path, return the corresponding depth image path.
    If depth image is missing, append None.
    """
    depth_paths = []
    for rgb_path in rgb_paths:
        frame_name = os.path.basename(rgb_path).replace(".color.jpg", ".depth_m.pgm")
        depth_path = os.path.join(sequence_dir, frame_name)
        if os.path.isfile(depth_path):
            depth_paths.append(depth_path)
        else:
            depth_paths.append(None)
    return depth_paths

def _vote_top_scan(cands60: list[dict]) -> str:
    """
    Vote a winning kb_scan_id.
    - Primary: frequency count across the 60 candidates.
    - Tie-break: smaller mean distance wins; then smaller min distance wins.
    """
    by_scan = defaultdict(list)
    for c in cands60:
        by_scan[c["kb_scan_id"]].append(c["distance"])
    # frequency
    counts = {sid: len(dists) for sid, dists in by_scan.items()}
    max_cnt = max(counts.values())
    tied = [sid for sid, cnt in counts.items() if cnt == max_cnt]
    if len(tied) == 1:
        return tied[0]
    # tie-break 1: mean distance
    means = {sid: (sum(by_scan[sid]) / len(by_scan[sid])) for sid in tied}
    best_mean = min(means.values())
    tied2 = [sid for sid, m in means.items() if abs(m - best_mean) < 1e-9]
    if len(tied2) == 1:
        return tied2[0]
    # tie-break 2: min distance
    mins = {sid: min(by_scan[sid]) for sid in tied2}
    return min(mins, key=mins.get)


from collections import defaultdict
from copy import deepcopy

def _merge_reference_sgs(ref_jsons: list[dict],
                         winning_scan: str,
                         keep_policy: str = "first",
                         max_edges_per_object: int | None = None) -> dict:
    """
    Merge multiple frame-level SGs (same scan) into one SG.

    Updated behavior:
      - KEEP 'floor' object nodes.
      - INCLUDE all edges involving floor (both floor↔object and floor↔floor).

    Assumptions:
      - object 'id' is scan-level consistent (from proposal.h5).

    Steps:
      1) Group objects by 'id' across frames; pick one representative per id.
      2) Identify floor ids (for bookkeeping), but DO NOT drop them.
      3) For each kept object (including floor), keep edges iff BOTH endpoints exist in kept set.
         Deduplicate edges by (src, relationship/pred_label, tgt).
      4) Optionally cap per-object edge count with `max_edges_per_object`.
    """
    by_id: dict[int, list[tuple[dict, int]]] = defaultdict(list)
    frames_used = set()

    # 1) group by id across frames
    for r in ref_jsons:
        sg = r.get("json") or {}
        f_id = int(r.get("kb_frame_id"))
        frames_used.add(f_id)
        for obj in sg.get("objects", []) or []:
            oid = obj.get("id")
            if oid is None:
                continue
            by_id[oid].append((obj, f_id))

    # choose representative per id
    kept_pre: dict[int, dict] = {}
    for oid, lst in by_id.items():
        if keep_policy == "most_edges":
            rep = max(lst, key=lambda x: len((x[0].get("edges") or [])))
        elif keep_policy == "min_frame":
            rep = min(lst, key=lambda x: x[1])
        else:  # "first"
            rep = lst[0]
        kept_pre[oid] = deepcopy(rep[0])

    # helper: detect floor (kept for info / future filters)
    def _is_floor(o: dict) -> bool:
        return (o.get("name") or "").strip().lower() == "floor"

    # 2) KEEP floor objects — do NOT drop them
    floor_ids = {oid for oid, o in kept_pre.items() if _is_floor(o)}
    kept = {oid: o for oid, o in kept_pre.items()}  # keep all (including floors)
    kept_ids = set(kept.keys())

    # 3) filter & dedup edges for every kept object (including floor)
    for oid, obj in kept.items():
        edges = obj.get("edges", []) or []
        new_edges, seen = [], set()
        for e in edges:
            src = e.get("src")
            tgt = e.get("tgt")
            rel = (e.get("relationship") or e.get("pred_label") or "").strip()
            if not rel or src is None or tgt is None:
                continue

            # keep only edges whose both endpoints exist in the merged graph
            if (src in kept_ids) and (tgt in kept_ids):
                key = (src, rel, tgt)
                if key not in seen:
                    new_edges.append(e)
                    seen.add(key)

        if isinstance(max_edges_per_object, int) and max_edges_per_object > 0:
            new_edges = new_edges[:max_edges_per_object]
        obj["edges"] = new_edges

    merged = {
        "scan": winning_scan,
        "frames_used": sorted(frames_used),
        "objects": sorted(kept.values(), key=lambda x: x.get("id", 1 << 30))
    }
    return merged


# ---------------------------
# 1. Load SigLIP model
# ---------------------------
def load_model(model_name: str = "google/siglip-base-patch16-224"):
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

# ---------------------------
# Helpers for 3RScan sequence
# ---------------------------

def _frame_idx_from_name(path: str) -> int:
    m = FRAME_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1))
    # fallback: any number
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0

def list_random_k_sequence_rgbs(sequence_dir: str, k: int = 12) -> List[str]:
    """
    Return a random *contiguous* window of up to k RGBs from a 3RScan 'sequence/'.
    - Collect all '*.color.jpg' under `sequence_dir`
    - Sort by frame index (via _frame_idx_from_name)
    - Randomly choose a start index and slice a contiguous block
    - If total frames < k, return all frames
    """
    if k <= 0:
        return []

    # Find all color frames (e.g., frame-000000.color.jpg, etc.)
    cands = glob.glob(os.path.join(sequence_dir, "*.color.jpg"))
    rgb_paths = sorted({p for p in cands if os.path.isfile(p)}, key=_frame_idx_from_name)

    n = len(rgb_paths)
    if n == 0:
        return []

    window = min(k, n)
    max_start = n - window
    # option 1: randomly choose a start index
    # start = random.randint(0, max_start) if max_start > 0 else 0

    # option 2: always choose the first k frames (for reproducibility)
    start = 0

    return rgb_paths[start:start + window]

# def list_random_k_clear_sequence_rgbs(sequence_dir: str, k: int = 12) -> List[str]:
#     """
#     Return a random *contiguous* window of up to k RGBs from a 3RScan 'sequence/'.
#     - Collect all '*.color.jpg' under `sequence_dir`
#     - Sort by frame index (via _frame_idx_from_name)
#     - Randomly choose a start index and slice a contiguous block
#     - If total frames < k, return all frames
#     """
#     if k <= 0:
#         return []

#     # Find all color frames (e.g., frame-000000.color.jpg, etc.)
#     cands = glob.glob(os.path.join(sequence_dir, "*.color.jpg")) + glob.glob(os.path.join(sequence_dir, "*.color.png"))
#     rgb_paths = sorted({p for p in cands if os.path.isfile(p)}, key=_frame_idx_from_name)

#     n = len(rgb_paths)
#     if n == 0:
#         return []

#     window = min(k, n)
#     max_start = n - window
#     start = random.randint(0, max_start) if max_start > 0 else 0
    
#     rgb_paths_clear = []
#     # Check for blurry images and skip them
#     for p in rgb_paths[start:]:

#         gray = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         variance = laplacian.var()
#         if variance < 15.0:  # Threshold for blurriness
#             continue
#         rgb_paths_clear.append(p)
#         if len(rgb_paths_clear) >= window:
#             break

#     return rgb_paths_clear

# Regex accepting "frame-000140" or "frame-000140.color.jpg/png/jpeg"
FRAME_RE = re.compile(r"frame-(\d+)(?:\.color)?(?:\.(?:jpg|jpeg|png))?$", re.IGNORECASE)
UUID_RE  = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

def parse_kb_key_to_scan_frame(key: str) -> tuple[str, int]:
    """Return (scan_uuid, frame_id) from a rgb_keys.pkl entry."""
    s = key.decode("utf-8", "ignore") if isinstance(key, (bytes, bytearray)) else str(key)
    s = s.replace("\\", "/")
    m_uuid = UUID_RE.search(s)
    m_fr = FRAME_RE.search(s) or re.search(r"frame-(\d+)", s, re.IGNORECASE)
    if not m_uuid or not m_fr:
        raise ValueError(f"Cannot parse scan/frame from key: {key}")
    return m_uuid.group(0).lower(), int(m_fr.group(1))


# ---------------------------
# 2. Main RAG pipeline for sequence queries
# ---------------------------
def run_rag(
    sequence_dir: str,
    embedding_index_path: str,
    embedding_data_path: str,   # kept for compatibility; not used here
    rgb_keys_path: str,
    kb_scan_root: str,          # where <scan_id>/frame_sg/frame-XXXXXX.json lies
    output_json_path: str,
    model_bundle,
    max_query_frames: int = 6,
    faiss_topk: int = 5         # ← Top-5 per query (was 1)
):
    """
    New retrieval:
      - For the first <=6 RGBs in 'sequence/', retrieve top-5 each via FAISS.
      - Aggregate up to 30 candidates and VOTE a winning kb_scan_id.
      - For each query, keep only the nearest candidate from the winning scan (if any).
      - Load those frame-level SG JSONs and MERGE them into a single reference SG.
    """
    processor, model = model_bundle

    # 1) Discover queries
    query_rgbs = list_random_k_sequence_rgbs(sequence_dir, k=max_query_frames)
    if not query_rgbs:
        raise FileNotFoundError(f"No RGBs found under: {sequence_dir}")
    current_scan_id = os.path.basename(os.path.dirname(sequence_dir.rstrip("/")))
    print(f"Query scan: {current_scan_id} | {len(query_rgbs)} query images (<= {max_query_frames})")

    # 2) Load FAISS & keys
    faiss_index = faiss.read_index(embedding_index_path)
    with open(rgb_keys_path, "rb") as f:
        rgb_keys = pickle.load(f)
    rgb_keys = [(k.decode("utf-8", "ignore") if isinstance(k, (bytes, bytearray)) else str(k)) for k in rgb_keys]


    # 3) Embed queries
    embs = []
    for p in tqdm(query_rgbs, desc="Embedding queries"):
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            embs.append(outputs.pooler_output[0].numpy().astype(np.float32))
    embs = np.stack(embs, axis=0)

    # 4) Search top-5 per query and collect 60 candidates
    cands60 = []          # all 12*5 candidates
    per_query_top5 = []   # for traceability in output
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
                "rank": rank
            }
            cands60.append(entry)
            qlist.append(entry)
        per_query_top5.append(qlist)

    if not cands60:
        raise RuntimeError("No valid FAISS candidates; check index/keys/embeddings.")

    # 5) Vote winning scan
    winning_scan = _vote_top_scan(cands60)
    print(f"[VOTE] winning_kb_scan_id = {winning_scan}")

    # 6) For each query, keep nearest candidate FROM the winning scan (if any)
    chosen_refs = []    # per-query chosen frame json (maybe fewer than #queries)
    refs_rbgs = []
    for qlist in per_query_top5:
        # filter to winning scan
        w = [c for c in qlist if c["kb_scan_id"] == winning_scan]
        if not w:
            continue
        best = min(w, key=lambda x: x["distance"])
        json_path = os.path.join(kb_scan_root, winning_scan, "frame_sg", f"frame-{best['kb_frame_id']:06d}.json")
        frame_rgb_path = os.path.join(kb_scan_root, winning_scan, "sequence", f"frame-{best['kb_frame_id']:06d}.color.jpg")
        refs_rbgs.append(frame_rgb_path)
        if not os.path.isfile(json_path):
            # allowed to skip if SG missing
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


    # 7) Merge chosen reference SGs into a single multi-frame reference
    merged_reference_sg = _merge_reference_sgs(chosen_refs, winning_scan)

    # 8) Save output
    # vote stats for debugging
    vote_counts = Counter([c["kb_scan_id"] for c in cands60])
    vote_stats = {
        "counts": dict(vote_counts),
        "winning_scan": winning_scan
    }

    out = {
        "current_scan_id": current_scan_id,
        "query_images": query_rgbs,
        "winning_kb_scan_id": winning_scan,
        "vote_stats": vote_stats,
        "merged_reference_sg": merged_reference_sg,
        "per_query_top5": per_query_top5,        # optional traceability
        "refs_rbgs": refs_rbgs
    }
    with open(output_json_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✅ RAG prompt JSON generated: {output_json_path}")


# ---------------------------
# 3. Build Qwen Prompt (sequence version)
# ---------------------------
def build_Qwen_prompt(json_path: str, output_prompt_path: str, output_txt_path: str):
    """
    Janus-compatible prompt with explicit placeholders.
    IMPORTANT: Qwen only accepts content items with type in {"text","image","video"}.
               We therefore feed both query images and reference images as type="image",
               and use nearby text markers to indicate their roles.
    """
    with open(json_path, "r") as f:
        rag = json.load(f)

    # 如果你不想立刻删原始RAG输出，可以注释掉下一行
    # os.remove(json_path)

    query_images = rag.get("query_images", []) or []
    merged_ref   = rag.get("merged_reference_sg", {}) or {}
    winning_sid  = rag.get("winning_kb_scan_id", "unknown")
    refs_rgbs    = rag.get("refs_rbgs", []) or []   # 参考场景图片（这次是GT，会等于query）

    # 限定可用谓词集合（保持你现有文件）
    relationship_type_path = "/hkfs/work/workspace/scratch/rg5174-myspace/RAG_baseline_Qwen2.5/relationships_n160.txt"
    with open(relationship_type_path, "r") as f:
        relationship_type = [line.strip() for line in f if line.strip()]

    # 序列化参考SG，尽量压缩以省token
    merged_ref_str = json.dumps(merged_ref, ensure_ascii=False, separators=(",", ":"))

    # 组装 content：先放查询图（每张图前插一个标注的text），再放参考图（同理），最后放总指令
    content = []

    # 查询图（标准 image）
    content = []
    for img in query_images:
        content.append({"type": "image", "image": f"{img}"})

    # 参考图（标准 image）——即便与query相同也没关系；如果你担心重复，可做去重
    # refs_rgbs = [p for p in refs_rgbs if p not in set(query_images)]  # 可选：去掉与query重复的图
    # for j, ref_img in enumerate(refs_rgbs, 1):
    #    content.append({"type": "text", "text": f"[REF #{j}] (similar scene image)"})
    #    content.append({"type": "image", "image": ref_img})

    # 大指令（使用三引号写法更安全）
    guide_text = f"""You are given a sequence of distinct RGB images from the same scan.

Now detect all objects and describe them in the video as many as possible.

Output JSON (object-centric):

{{
  "objects": [
    {{
      "id": "1", "name": "chair",
      "description": "a wooden chair with a cushion",
      "seen_in_frames": [0, 2, 3],  # list of query image indices where this object appears
    }},
    {{"id":"2", ...}}, ...
  ]
}}

"seen_in_frames" is a list of indices indicating which query images the object appears in, for example, [0,2,3] means the object appears in the first, third, and fourth query images.
[1,5] means the object appears in the second and sixth query images.
The given sequence of images are from the same scan, and objects might appear in multiple images, but the total number of distinct objects should be kept smaller than 30.
You are encouraged to detect objects as many as possible, but when the number is bigger than 30, choose these larger and more important objects.
Return ONLY the final JSON, no other text or explanation.
"""
    content.append({"type": "text", "text": guide_text})

    messages = [{"role": "user", "content": content}]

    if os.path.exists(output_prompt_path):
        os.remove(output_prompt_path)
    with open(output_prompt_path, "w") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

    with open(output_txt_path, "w") as f:
        f.writelines("\n".join(query_images))




# =========================
# 4. Qwen inference
# =========================
import os, re, json, torch
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_model_inference(
    prompt_path: str,
    qwen_model_bundle,
    out_path: Optional[str] = None,
    scan_id: Optional[str] = None, 
    # model_path: str = "Qwen/Qwen2.5-VL-32B-Instruct",  
    max_new_tokens: int = 8192,
    use_bf16: bool = True
) -> Dict[str, Any]:

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype='auto',device_map="auto")
    # processor = AutoProcessor.from_pretrained(model_path)

    processor, model = qwen_model_bundle

    with open(prompt_path, "r") as f:
        messages = json.load(f)

    os.remove(prompt_path)  # remove the prompt file after reading
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # see input tokens
    num_tokens = int(inputs.input_ids.numel()) if hasattr(inputs, "input_ids") else -1
    print(f"📝 input tokens: {num_tokens}")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("\n🧠 Qwen output:\n", output_text)

    # Try to extract a JSON object from the text
    parsed_json = None
    if isinstance(output_text, list):
        output_text_str = "\n".join(output_text)
    else:
        output_text_str = str(output_text)
    m = re.search(r"\{[\s\S]*\}", output_text_str)
    if m:
        candidate = m.group(0)
        try:
            parsed_json = json.loads(candidate)
        except Exception as e:
            print(f"[WARN] failed to parse JSON block: {e}")

    # Save results
    if out_path is None:
        stem = os.path.splitext(os.path.basename(prompt_path))[0]
        out_path = os.path.join(os.path.dirname(prompt_path), f"{scan_id}_generation.json")

    result = {
        "raw_text": output_text,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ Inference saved to: {out_path}")

    # 如果需要把解析出的 JSON 单独落盘：
    if parsed_json is not None:
        sg_path = os.path.join(os.path.dirname(out_path), f"{scan_id}_parse_sg.json")
        with open(sg_path, "w") as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        print(f"🗂️ Parsed scene graph saved to: {sg_path}")

    return result