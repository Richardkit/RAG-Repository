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
import json
import glob
import random
from collections import defaultdict
import numpy as np

from PIL import Image, ImageDraw
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from collections import defaultdict, Counter
from vllm import LLM, SamplingParams
import re as _re, json as _json

# ================== 配置区域 ==================

# 使用 8B 模型（当前脚本配置）
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

# GT bbox 信息，与 gt_description_generate.py 保持一致
GT_BBOXES_PATH = "/hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_baseline_Qwen3/evaluate/gt_bboxes.json"

# 3RScan GT 场景目录（包含 {scan_id}_scan_sg.json 和 sequence/*.color.jpg）
SCAN_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan"

# 验证集 scan 列表
VALID_SCANS_PATH = "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"

# Predicted 全局 scene graph 根目录
PRED_SG_ROOT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/RAG_Qwen3/RAG_end2end_CoT"

# 临时可视化 bbox 图像目录
TEMP_IMG_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_baseline_Qwen3/temporary_img_qwen_judge"

# 输出结果：每行一个 JSON
OUT_JSONL = "/hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_baseline_Qwen3/evaluate/CoT_rag_result/obj_match_results.jsonl"

# 单个 GT obj 最多选几帧做判断
MAX_IMAGES_PER_OBJ = 5
MIN_BBOX_AREA = 1000  # 与原脚本一致

# ---- Global LLM cache for vLLM ----
_GLOBAL_VLLM = None

# =========================================================
# 工具函数：加载 scan 列表、bbox 图像等
# =========================================================

def load_valid_scans(valid_scans_path):
    with open(valid_scans_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def prepare_images_with_selection(scan_id, object_id, gt_object_name, scan_dir,
                                  temp_img_dir, max_images=6, min_area=400):
    """
    为某个 GT obj 选出若干帧并画 bbox。
    返回临时图片路径列表。
    """
    with open(os.path.join(scan_dir, scan_id, f"{scan_id}_gt_bboxes.json"), "r") as f:
        gt_bboxes = json.load(f)

    if scan_id not in gt_bboxes:
        print(f"[BBox] Scan ID {scan_id} not found in gt_bboxes.json.")
        return []

    if object_id not in gt_bboxes[scan_id]:
        print(f"[BBox] Object ID {object_id} not found in scan {scan_id}.")
        return []

    obj_data = gt_bboxes[scan_id][object_id]
    seen_frames = obj_data.get("seen_frames", [])
    bounding_boxes = obj_data.get("bounding_boxes", [])

    os.makedirs(temp_img_dir, exist_ok=True)

    bbox_areas = [
        (i, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        for i, bbox in enumerate(bounding_boxes)
    ]
    valid_indices = [i for i, area in bbox_areas if area > min_area]

    if not valid_indices:
        print(f"[BBox] No bounding boxes with area > {min_area} for {scan_id} obj {object_id}.")
        return []

    largest_index = max(valid_indices, key=lambda i: bbox_areas[i][1])
    selected_indices = [largest_index]

    remaining_indices = [i for i in valid_indices if i != largest_index]
    if remaining_indices:
        selected_indices += random.sample(
            remaining_indices,
            min(max_images - 1, len(remaining_indices))
        )

    prepared_images = []
    for i in selected_indices:
        frame = seen_frames[i]
        bbox = bounding_boxes[i]
        frame_path = os.path.join(scan_dir, scan_id, "sequence", f"{frame}.color.jpg")
        if not os.path.exists(frame_path):
            print(f"[BBox] Frame {frame} not found in scan {scan_id}. Skipping.")
            continue

        image = Image.open(frame_path)
        draw = ImageDraw.Draw(image)

        if gt_object_name not in ["floor", "wall", "ceiling"]:
            
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

        temp_img_path = os.path.join(temp_img_dir, f"{scan_id}_{frame}_{object_id}.jpg")
        image.save(temp_img_path)
        prepared_images.append(temp_img_path)

    return prepared_images


def delete_temporary_images(temp_img_dir):
    for img_path in glob.glob(os.path.join(temp_img_dir, "*.jpg")):
        os.remove(img_path)
    print(f"[Tmp] Deleted all temporary images in {temp_img_dir}.")


def load_gt_scan_sg(scan_id):
    path = os.path.join(SCAN_DIR, scan_id, f"{scan_id}_scan_sg.json")
    if not os.path.exists(path):
        print(f"[GT SG] Not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_pred_global_sg(scan_id):
    """
    加载 predicted 全局 scene graph: {PRED_SG_ROOT}/{scan_id}_global_graph_final.json
    """
    path = os.path.join(PRED_SG_ROOT, scan_id, f"{scan_id}_global_graph_final.json")
    if not os.path.exists(path):
        print(f"[Pred SG] Not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

# ==========================================================
# vllm inference
# ==========================================================
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
            enforce_eager=False,              # Temporarily True to avoid compile timeout
            # executor="uniproc",            # 如果你现在用这个，也可以保留
        )
    return _GLOBAL_VLLM

def prepare_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8192):
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

def run_qwen_vllm(messages, qwen_proc, max_new_tokens: int = 8192):

        request, in_tokens = prepare_qwen_vllm(messages, qwen_proc, max_new_tokens)

        # ★ 不再 new LLM，而是拿全局单例
        llm = get_qwen_vllm()

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            top_k=-1,
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

# ==========================================================
# Clean function: 1. Main Process for pred-vs-gt object matching
# ==========================================================

def pred_vs_gt(scan_level_out, total_matched_pred_ids, full_candidates, qwen_proc):

    used_pred_ids = []
    used_gt_ids = []
    best_match_gt2pred = {f"{item['gt_obj_id']}": "none" for item in scan_level_out['nodes_mapping']}

    def iterate_over_matched_pred_nodes(scan_level_out, total_matched_pred_ids, full_candidates, 
                                        qwen_proc, used_pred_ids, used_gt_ids, best_match_gt2pred, iterate_time):
        """
        Iterate over matched predicted node ids to find the best matching GT object for each predicted node.
        Updates the best_match_gt2pred dictionary with the best matches.
        """
        # first iterate over the 1st matched pred ids
        for pred_node_id in total_matched_pred_ids:

            if pred_node_id in used_pred_ids:
                continue

            best_matched_gt_nodes = []

            for item in scan_level_out["nodes_mapping"]:
                                     
                gt_id = str(item.get('gt_obj_id'))

                # already final matched gt will not be considered again
                if gt_id in used_gt_ids:
                    continue
            
                match_ids = item.get('match_pred_node_ids', [])
                scores = item.get('similarity_scores', [])
            
                if not match_ids or not scores:
                    continue
                if len(match_ids) != len(scores):
                    continue
                if len(match_ids) < iterate_time:
                    continue
            
                order = np.argsort(scores)
                idx = order[-iterate_time]
                if idx >= len(match_ids):
                    continue
                
                if match_ids[idx] != pred_node_id:
                    continue
            
                best_matched_gt_nodes.append(item)
        
            if not best_matched_gt_nodes:
                continue
            
            pred_label, pred_description = next(((lab, desc) for (pid, lab, desc) in full_candidates if pid == pred_node_id), ("none", ""))


            if len(best_matched_gt_nodes) == 1:

                gt_node = best_matched_gt_nodes[0]
                gt_obj_id = str(gt_node['gt_obj_id'])
                best_match_gt2pred[gt_obj_id] = pred_node_id
                used_pred_ids.append(pred_node_id)
                used_gt_ids.append(gt_obj_id)

            else:
                gt_candidate_imgs = []
                for gt_node in best_matched_gt_nodes:
                    gt_obj_id = gt_node['gt_obj_id']
                    gt_label = gt_node['gt_label']
                    img_paths = prepare_images_with_selection(gt_node['scan_id'], gt_obj_id, gt_label, SCAN_DIR, TEMP_IMG_DIR, max_images=2, min_area=MIN_BBOX_AREA)

                    if not img_paths:
                        continue 

                    gt_candidate_imgs.append((gt_node, img_paths))

                if not gt_candidate_imgs:
                    continue
                                
                best_gt_obj_id, raw_output = run_qwen_pred_vs_gt_inference(build_prompt_for_pred_vs_gt(gt_candidate_imgs, pred_node_id, pred_label, pred_description), qwen_proc)
            
                if (not best_gt_obj_id) or (str(best_gt_obj_id).lower() == "none"):
                    continue

                best_gt_obj = str(best_gt_obj_id)

                if not best_gt_obj_id or best_gt_obj_id.lower() == "none":
                    continue    

                best_match_gt2pred[best_gt_obj] = pred_node_id
                used_gt_ids.append(best_gt_obj)
                used_pred_ids.append(pred_node_id)

                delete_temporary_images(TEMP_IMG_DIR)

    iterate_over_matched_pred_nodes(scan_level_out, total_matched_pred_ids, full_candidates, qwen_proc, used_pred_ids, used_gt_ids, best_match_gt2pred, 1)
    iterate_over_matched_pred_nodes(scan_level_out, total_matched_pred_ids, full_candidates, qwen_proc, used_pred_ids, used_gt_ids, best_match_gt2pred, 2)
    iterate_over_matched_pred_nodes(scan_level_out, total_matched_pred_ids, full_candidates, qwen_proc, used_pred_ids, used_gt_ids, best_match_gt2pred, 3)


    return best_match_gt2pred

# ==========================================================
# Clean function: 2. Qwen prompt for pred-vs-gt object matching，inserted in itervate_over_matched_pred_nodes
# ==========================================================

def build_prompt_for_pred_vs_gt(gt_candidate_imgs, pred_node_id, pred_label, pred_description):
    """
    Construct prompt for Qwen to match predicted node with unique GT object among all GT candidates. 
    """

    content = []

    for gt_node, img_paths in gt_candidate_imgs:
        content.append({'type':'text', 'text': f"the following images show the ground-truth object with label '{gt_node['gt_label']}' and id '{gt_node['gt_obj_id']}'."})
        for p in img_paths:
            content.append({'type':'image', 'image': p})
    
    guide = f"""
    You are given several text-image pairs, where each text describes a ground-truth object and the corresponding images highlight that object with yellow bounding boxes.
    Below is a description of a certain object description and its ID from a predicted scene graph: {pred_node_id}: [label='{pred_label}'] description: "{pred_description}".

    Your task is to determine which ground-truth object (if any) matches the predicted object description at best among all the provided ground-truth objects.

    OUTPUT ONLY THE ID of the best-matching ground-truth object in JSON format as follows:
    {{
      "best_matching_gt_obj_id": <the ID of the best-matching ground-truth object (string)>
    }}
    """
    content.append({"type": "text", "text": guide})

    return [{"role": "user", "content": content}]

def run_qwen_pred_vs_gt_inference(prompt, qwen_proc):
    """
    Call Qwen to match predicted node with unique GT object among all GT candidates.
    Returns: best_matching_gt_obj_id (str), raw_output_text
    """

    parsed, output_text = run_qwen_vllm(prompt, qwen_proc, max_new_tokens=1024)

    best_matching_gt_obj_id = "none"

    try:
        parsed = json.loads(output_text)
        best_matching_gt_obj_id = [str(parsed.get("best_matching_gt_obj_id", "none")).strip()]

        if len(best_matching_gt_obj_id) == 0 :
            best_matching_gt_obj_id = "none"

        else:
            best_matching_gt_obj_id = best_matching_gt_obj_id[0]

    except Exception as e:
        print(f"[WARN] Failed to parse Qwen JSON output for pred-vs-gt matching: {e}")

    return best_matching_gt_obj_id, output_text

# =========================================================
# Qwen 调用部分：构造 prompt + 解析 JSON 输出
# =========================================================

def build_prompt_for_matching(images, object_name, gt_object_id, candidates):
    """
    构造给 Qwen 的对话：
    - 多张 bbox 图像：同一个 GT object
    - 一系列候选 predicted nodes: [(pred_id, label, description), ...]
    让模型判断：哪一个 pred_id 对应图中 bbox highlight 的同一物体；
    若都不对应 → 'none'；同时给出 confidence。
    """
    content = [{"type": "image", "image": p} for p in images]

    # 构造候选描述文本
    cand_lines = []
    for pid, plabel, pdesc in candidates:
        plabel_txt = plabel if plabel else "unknown"
        pdesc_txt = pdesc if pdesc else ""
        cand_lines.append(
            f"ID {pid}: [label='{plabel_txt}'] description: \"{pdesc_txt}\""
        )
    cand_block = "\n".join(cand_lines) if cand_lines else "(No candidate nodes)"

    guide = f"""
You are given several images. In ALL these images, the yellow bounding boxes highlight the SAME ground-truth object.

This ground-truth object has label: '{object_name}' and id: '{gt_object_id}'.

Below is a list of candidate objects from a predicted scene graph. Each candidate has an ID and a text description:

{cand_block}

Your task:
1. Carefully look at the object inside the bounding boxes (except 'floor' and 'wall', they do not have yellow bounding boxes) across all images.
2. Read ALL candidate descriptions.
3. Decide whether ANY of these descriptions clearly refers to EXACTLY the same physical object as highlighted by the bounding boxes.
4. Choose three candidate descriptions that best match the ground truth object with corresponding similarity score.
6. If NONE of the descriptions matches the object in the bounding boxes, fill 'none' in the three placeholder <> in "match_pred_node_id" and fill '1' in the three placeholder <> in "score".
7. Otherwise, fill the ID of the best-matching candidate in "match_pred_node_id" and fill the corresponding similarity score in "score".


Similarity score:
- 5 means a perfect match, all details of ground-truth object are satisfied in the description.
- 4 means a good match, most details of ground-truth object are satisfied in the description.
- 3 means a partial match, some details of ground-truth object are satisfied in the description, but others are unknown or uncertain, due to a lack of comprehensive visibility.
- 2 means a weak match, only very generic details of ground-truth object are satisfied in the description, but others are wrong.
- 1 means a poor match, almost none of the details of ground-truth object are satisfied in the description.

Important:
- Only choose an ID if the description is clearly consistent with the visual appearance, category, and context (e.g. position, shape, size).
- When the ground-truth object is something generic like 'floor' and 'wall', there are not yellow bounding boxes for them in the images, carefully judge them, because there are always other objects covering the floor partially.
- In 'score', except score '1' for 'none', other score levels should only be used once, i.e., do not assign the same score to multiple candidates for one ground-truth object. 
- It is impossible that one candidate ID appears in one "match_pred_node_ids" multiple times.

OUTPUT STRICTLY IN JSON, no extra text:

{{
  "match_pred_node_ids": [<the chosen candidate ID (string), or 'none' if no match>, <the chosen candidate ID (string), or 'none' if no match>, <the chosen candidate ID (string), or 'none' if no match>],
  "similarity_scores": [<a integer 1-5>, <a integer 1-5>, <a integer 1-5>]
}}
"""
    content.append({"type": "text", "text": guide})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def run_qwen_match_inference(prompt, qwen_proc):
    """
    调用 Qwen 进行匹配判断，
    返回: match_pred_node_id (str 或 'none'), confidence (float), raw_output_text
    """

    parsed, output_text = run_qwen_vllm(prompt, qwen_proc, max_new_tokens=2048)

    match_id = "none"
    ids = []
    score_vals = []

    try:
        parsed = json.loads(output_text)
        raw_ids = parsed.get("match_pred_node_ids", [])
        if isinstance(raw_ids, (int,float,str)):
            raw_ids = [raw_ids]
        
        for v in raw_ids:
            if isinstance(v, (int,float)):
                vid = str(v)
            else:
                vid = str(v).strip()
            ids.append(vid)

        scores = parsed.get("similarity_scores", [])

        if isinstance(scores, (int,float,str)):
            scores = [scores]

        for val in scores:
            try:
                s = int(val)
            except Exception:
                s = 1
            score_vals.append(s)

        while len(ids) < 3:
            ids.append("none")

        while len(score_vals) < 3:
            score_vals.append(1)

    except Exception as e:
        print(f"[WARN] Failed to parse Qwen JSON output (3-candidate) : {e}")
        ids = ["none", "none", "none"]
        score_vals = [1, 1, 1]



    return ids, score_vals, output_text


# =========================================================
# 辅助：读取已有结果，构建已处理 obj / 已用 pred id
# =========================================================

def load_existing_results(out_jsonl_path):
    """
    读取已有的 obj_match_results.jsonl （最新稳定版）
    支持：
      - 完整 scan-level 记录（含 gt2pred，必须为 dict 才算完成的 scan）
      - 部分 per-GT-level 记录（match_pred_node_ids 或旧 match_pred_node_id）
    返回:
      processed_objs: set[(scan_id, gt_obj_id)]
      processed_map:  dict[scan_id] -> set(gt_obj_id)
      saved_nodes_mapping: dict[scan_id] -> list[per-GT rec]
      finished_scans: set[scan_id]  # 仅当存在合法 scan-level 记录
    """

    processed_objs = set()
    processed_map = defaultdict(set)
    saved_nodes_mapping_dict = defaultdict(dict)   # scan_id -> {gt_obj_id: rec}
    finished_scans = set()

    if not os.path.exists(out_jsonl_path):
        return processed_objs, processed_map, {}, finished_scans

    with open(out_jsonl_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue

            try:
                rec = json.loads(raw)
            except Exception:
                continue

            scan_id = rec.get("scan_id")
            if not scan_id:
                continue

            # ================================================================
            # case 1: 完整 scan-level 记录（必须含 gt2pred 且是 dict）
            # ================================================================
            if "gt2pred" in rec and isinstance(rec["gt2pred"], dict):
                finished_scans.add(scan_id)

                nm_list = rec.get("nodes_mapping", [])
                for item in nm_list:
                    gt_obj_id = item.get("gt_obj_id")
                    if gt_obj_id is None:
                        continue
                    gid = str(gt_obj_id)

                    processed_objs.add((scan_id, gid))
                    processed_map[scan_id].add(gid)

                    # 覆盖模式（如果 jsonl 中有多个版本）
                    saved_nodes_mapping_dict[scan_id][gid] = item
                continue

            # ================================================================
            # case 2: per-GT-level 记录（旧格式 or 3-candidate 格式）
            # ================================================================
            gt_obj_id = rec.get("gt_obj_id")
            if gt_obj_id is None:
                continue

            gid = str(gt_obj_id)
            processed_objs.add((scan_id, gid))
            processed_map[scan_id].add(gid)

            # ---- 构造统一结构：match_pred_node_ids 必为长度 3 的 list ----
            match_ids = rec.get("match_pred_node_ids")

            if match_ids is None:
                # 老格式只有一个 id
                single = rec.get("match_pred_node_id")
                if single is None:
                    match_ids = ["none", "none", "none"]
                else:
                    match_ids = [single, "none", "none"]

            # similarity_scores 同理处理
            scores = rec.get("similarity_scores")
            if scores is None:
                conf = rec.get("raw_confidence", 1.0)
                scores = [conf, 1.0, 1.0]

            item = {
                "scan_id": scan_id,
                "gt_obj_id": gid,
                "gt_label": rec.get("gt_label", "unknown"),
                "match_pred_node_ids": match_ids,
                "similarity_scores": scores,
            }

            saved_nodes_mapping_dict[scan_id][gid] = item

    # dict-of-dict → dict-of-list
    saved_nodes_mapping = {
        sid: list(v.values()) for sid, v in saved_nodes_mapping_dict.items()
    }

    return processed_objs, processed_map, saved_nodes_mapping, finished_scans



# =========================================================
# 主流程
# =========================================================

def main():
    # ============================================================
    # 1. 加载模型
    # ============================================================
    print("[Load] Qwen3-VL-32B-Instruct model ...")
    # qwen_model = AutoModelForImageTextToText.from_pretrained(
    #     MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    # )
    qwen_proc = AutoProcessor.from_pretrained(MODEL_NAME)

    # ============================================================
    # 2. 加载验证集 scan 列表
    # ============================================================
    scan_ids = load_valid_scans(VALID_SCANS_PATH)
    print(f"[Info] Loaded {len(scan_ids)} validation scans.")

    # ============================================================
    # 3. 若已有 JSONL，读取已处理对象 / 已完成 scan / 已保存 per-GT 记录
    # ============================================================
    processed_objs, processed_map, saved_nodes_mapping, finished_scans = load_existing_results(OUT_JSONL)
    print(f"[Info] Loaded existing results: {len(processed_objs)} processed (scan,obj) pairs.")

    # 开文件句柄（追加）
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    out_f = open(OUT_JSONL, "a")

    # ============================================================
    # 4. 遍历所有 scan（全部按增量）
    # ============================================================
    for idx, scan_id in enumerate(sorted(scan_ids)):
        # 如果这个 scan 已经有 gt2pred 记录，说明整个 scan 已经完整处理过，直接跳过
        if scan_id in finished_scans:
            print(f"[Skip Scan] {scan_id} already finished (gt2pred found).")
            continue

        print("\n" + "=" * 80)
        print(f"[Scan {idx+1}/{len(scan_ids)}] {scan_id}")

        # ------------------------------------------------------------
        # 载入 GT / Pred scene graph
        # ------------------------------------------------------------
        gt_sg = load_gt_scan_sg(scan_id)
        if gt_sg is None:
            print(f"[Skip] GT SG missing for scan {scan_id}")
            continue

        pred_sg = load_pred_global_sg(scan_id)
        if pred_sg is None:
            print(f"[Skip] Predicted SG missing for scan {scan_id}")
            continue

        # ------------------------------------------------------------
        # 构造 predicted candidate node 列表
        # ------------------------------------------------------------
        pred_nodes = pred_sg.get("nodes", [])
        full_candidates = [
            (str(n.get("id")), n.get("label") or n.get("name") or "",
             n.get("description") or "")
            for n in pred_nodes
        ]

        # 这个 scan 的临时结果容器
        scan_level_out = {"scan_id": scan_id}
        scan_level_out["nodes_mapping"] = []
        total_matched_pred_nodes = set()

        # 如果之前这个 scan 已经有部分 GT 算过候选结果，先恢复进来
        if scan_id in saved_nodes_mapping:
            for rec in saved_nodes_mapping[scan_id]:
                scan_level_out["nodes_mapping"].append(rec)
                match_ids = rec.get("match_pred_node_ids", [])
                for pid in match_ids:
                    if isinstance(pid, str) and pid.lower() != "none":
                        total_matched_pred_nodes.add(pid)

        # ------------------------------------------------------------
        # 遍历 GT nodes（按 object 增量处理）
        # ------------------------------------------------------------
        for node in gt_sg.get("nodes", []):
            gt_obj_id = str(node.get("id"))
            gt_label = node.get("label", "unknown")

            # 如果已经处理过该 GT object → 直接跳过（不会重算）
            if (scan_id, gt_obj_id) in processed_objs:
                print(f"[Skip GT] {scan_id} obj {gt_obj_id} already processed.")
                continue

            print(f"\n[GT Obj] scan={scan_id}, obj={gt_obj_id}, label='{gt_label}'")

            # ------------------------------------------------------------
            # 4.1 准备 bbox 图像
            # ------------------------------------------------------------
            prepared_images = prepare_images_with_selection(
                scan_id, gt_obj_id, gt_label,
                SCAN_DIR,
                TEMP_IMG_DIR,
                max_images=MAX_IMAGES_PER_OBJ,
                min_area=MIN_BBOX_AREA
            )

            if not prepared_images:
                print(f"[Warn] No valid bbox images for scan={scan_id} obj={gt_obj_id}. Mark as none.")
                match_ids, score_vals = ["none", "none", "none"], [1.0, 1.0, 1.0]
            else:
                # ------------------------------------------------------------
                # 4.2 构造 candidates（这里不做去重，后面 pred_vs_gt 再处理冲突）
                # ------------------------------------------------------------
                candidates = [(pid, lab, desc) for (pid, lab, desc) in full_candidates]

                if not candidates:
                    print(f"[Warn] No pred candidates for scan {scan_id}.")
                    match_ids, score_vals = ["none", "none", "none"], [0.0, 0.0, 0.0]
                else:
                    # --------------------------------------------------------
                    # 4.3 调 Qwen 进行 object-description 判断
                    # --------------------------------------------------------
                    prompt = build_prompt_for_matching(prepared_images, gt_label, gt_obj_id, candidates)
                    match_ids, score_vals, raw_out = run_qwen_match_inference(
                        prompt, qwen_proc
                    )  # match_ids: [id1, id2, id3], score_vals: [s1, s2, s3]
                    print(f"[RawMatch] → {match_ids}, scores={score_vals}")

            # ------------------------------------------------------------
            # 4.4 保存当前 GT 的候选结果（同时写入内存 + 文件）
            # ------------------------------------------------------------
            rec = {
                "scan_id": scan_id,
                "gt_obj_id": gt_obj_id,
                "gt_label": gt_label,
                "match_pred_node_ids": match_ids,
                "similarity_scores": score_vals,
            }

            # 内存里先 append（后面 pred_vs_gt 要用）
            scan_level_out["nodes_mapping"].append(rec)
            # 记录目前这个 GT 用到的所有候选 pred_id（用于统计）
            for pid in match_ids:
                if isinstance(pid, str) and pid.lower() != "none":
                    total_matched_pred_nodes.add(pid)

            # **关键：立刻写入一行 per-GT 记录，保证中途 kill 也能恢复**
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

            # ------------------------------------------------------------
            # 4.5 标记为已处理
            # ------------------------------------------------------------
            processed_objs.add((scan_id, gt_obj_id))
            processed_map[scan_id].add(gt_obj_id)

            delete_temporary_images(TEMP_IMG_DIR)

        # ------------------------------------------------------------
        # 4.6 整个 scan 的 GT 都跑完后 → 做一次全局的 pred_vs_gt
        # ------------------------------------------------------------
        if total_matched_pred_nodes:
            best_match_gt2pred = pred_vs_gt(
                scan_level_out,
                list(total_matched_pred_nodes),
                full_candidates,
                qwen_proc,
            )
        else:
            best_match_gt2pred = {}

        # ------------------------------------------------------------
        # (A) 写一条 scan-level final 记录（含3-candidate + gt2pred）
        # ------------------------------------------------------------
        final_rec = {
            "scan_id": scan_id,
            "gt2pred": best_match_gt2pred,
            "nodes_mapping": scan_level_out["nodes_mapping"],
        }
        out_f.write(json.dumps(final_rec) + "\n")
        out_f.flush()

        # ------------------------------------------------------------
        # (B) 为该 scan 写 N 条旧格式 {gt_obj_id, match_pred_node_id}
        # ------------------------------------------------------------
        for item in scan_level_out["nodes_mapping"]:
            gid = str(item["gt_obj_id"])
            gt_label = item["gt_label"]
            final_pred = best_match_gt2pred.get(gid, "none")

            old_style_rec = {
                "scan_id": scan_id,
                "gt_obj_id": gid,
                "gt_label": gt_label,
                "match_pred_node_id": final_pred,   # <= 只有一条
            }
            out_f.write(json.dumps(old_style_rec) + "\n")

        out_f.flush()

        # ------------------------------------------------------------
        # (C) 清理 JSONL 文件中本 scan 的中间格式 (3-candidate) 行
        # ------------------------------------------------------------
        out_f.close()

        jsonl_path = OUT_JSONL
        tmp_path = jsonl_path + ".tmp"

        with open(jsonl_path, "r") as fin, open(tmp_path, "w") as fout:
            for line in fin:
                try:
                    rec = json.loads(line.strip())
                except Exception:
                    continue

                # 本 scan_id 的情况
                if rec.get("scan_id") == scan_id:

                    # 保留 scan-level final
                    if "gt2pred" in rec:
                        fout.write(json.dumps(rec) + "\n")
                        continue

                    # 保留旧格式（match_pred_node_id）
                    if "match_pred_node_id" in rec:
                        fout.write(json.dumps(rec) + "\n")
                        continue

                    # 其余（中间格式：match_pred_node_ids/similarity_scores）全部删除
                    continue

                # 其他 scan 原样保留
                fout.write(json.dumps(rec) + "\n")

        os.replace(tmp_path, jsonl_path)
        print(f"[Cleaned] Removed intermediate GT records for scan {scan_id}")

        # 清理完之后，重新以 append 方式打开文件，供下一个 scan 使用
        out_f = open(OUT_JSONL, "a", encoding="utf-8")
    # ============================================================
    # 结束
    # ============================================================
    out_f.close()
    print(f"\n[Done] All results saved to: {OUT_JSONL}")

if __name__ == "__main__":
    main()
