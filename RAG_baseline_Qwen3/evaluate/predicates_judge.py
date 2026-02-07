# predicate_judge.py
# -*- coding: utf-8 -*-

import os
import re
import json
import uuid
import shutil
from typing import List, Dict, Tuple, Any, Optional

from PIL import Image, ImageDraw

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


# ---------------------------------------------------------
# 0) 基础工具
# ---------------------------------------------------------

def _atomic_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def _make_unique_tmp_dir(base_dir: str) -> str:
    _atomic_mkdir(base_dir)
    d = os.path.join(base_dir, f"pred_judge_{uuid.uuid4().hex[:12]}")
    os.makedirs(d, exist_ok=True)
    return d

def _safe_rmtree(path: str):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def _extract_json_obj(text: str) -> Dict[str, Any]:
    """
    从模型输出中提取第一个 { ... } JSON。
    """
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _norm_str(x: Any) -> str:
    return str(x).strip()


# ---------------------------------------------------------
# 1) 图片路径：frame -> jpg
# ---------------------------------------------------------

def frame_to_path(scan_dir: str, scan_id: str, frame: str) -> str:
    """
    参考 gen_predicate.py 的约定：
      {SCAN_DIR}/{scan_id}/sequence/{frame}.color.jpg
    """
    return os.path.join(scan_dir, scan_id, "sequence", f"{frame}.color.jpg")


# ---------------------------------------------------------
# 2) bbox 收集 + 画框（基本复用 gen_predicate.py）
# ---------------------------------------------------------

def _collect_valid_bboxes_for_obj(
    bbox_data: Dict[str, Any],
    min_area: int
) -> List[Tuple[str, List[float]]]:
    """
    返回该 obj 的所有 (frame, bbox) 且面积 > min_area
    bbox: [x1,y1,x2,y2]
    """
    seen_frames = bbox_data.get("seen_frames", [])
    bounding_boxes = bbox_data.get("bounding_boxes", [])
    out = []
    for frame, bbox in zip(seen_frames, bounding_boxes):
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > min_area:
            out.append((str(frame), bbox))
    return out


def _draw_pair_bbox(
    scan_dir: str,
    scan_id: str,
    frame: str,
    bbox_src: List[float],
    bbox_tgt: List[float],
    out_dir: str,
    idx: int
) -> str:
    """
    在同一张图上画 src/tgt：
      - src: RED
      - tgt: YELLOW
    """
    img_path = frame_to_path(scan_dir, scan_id, frame)
    if not os.path.exists(img_path):
        return ""

    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # src red
    x1, y1, x2, y2 = bbox_src
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # tgt yellow
    x1, y1, x2, y2 = bbox_tgt
    draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

    out_path = os.path.join(out_dir, f"{scan_id}_joint_{frame}_{idx}.jpg")
    image.save(out_path)
    return out_path


def _draw_single_bbox(
    scan_dir: str,
    scan_id: str,
    frame: str,
    bbox: List[float],
    out_dir: str,
    is_src: bool,
    idx: int
) -> str:
    """
    一张图只强调一个 bbox:
      - src => red
      - tgt => yellow
    """
    img_path = frame_to_path(scan_dir, scan_id, frame)
    if not os.path.exists(img_path):
        return ""

    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = bbox
    color = "red" if is_src else "yellow"
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    role = "src" if is_src else "tgt"
    out_path = os.path.join(out_dir, f"{scan_id}_sep_{role}_{frame}_{idx}.jpg")
    image.save(out_path)
    return out_path


def prepare_pair_images(
    scan_dir: str,
    scan_id: str,
    src_id: str,
    tgt_id: str,
    gt_bboxes: Dict[str, Any],
    tmp_dir: str,
    min_bbox_area: int = 1000,
    num_images_joint: int = 4,
    num_images_per_obj_separate: int = 2,
) -> Tuple[List[str], str]:
    """
    为一个 (src, tgt) pair 生成 4 张图（写入 tmp_dir）：
      - 优先 joint 模式（4 张都是两者同时出现且 bbox > min_bbox_area）
      - 不足则 fallback separate（src两张 + tgt两张）
    返回: (image_paths, mode)  mode ∈ {"joint","separate","none"}
    """
    scan_boxes = gt_bboxes[scan_id]
    if src_id not in scan_boxes or tgt_id not in scan_boxes:
        return [], "none"

    src_data = scan_boxes[src_id]
    tgt_data = scan_boxes[tgt_id]

    src_valid = _collect_valid_bboxes_for_obj(src_data, min_bbox_area)
    tgt_valid = _collect_valid_bboxes_for_obj(tgt_data, min_bbox_area)
    if not src_valid or not tgt_valid:
        return [], "none"

    # --- joint ---
    src_dict = {frame: bbox for frame, bbox in src_valid}
    tgt_dict = {frame: bbox for frame, bbox in tgt_valid}
    joint_frames = sorted(set(src_dict.keys()) & set(tgt_dict.keys()))

    joint_candidates = []
    for f in joint_frames:
        bbox_s = src_dict[f]
        bbox_t = tgt_dict[f]
        area_s = (bbox_s[2] - bbox_s[0]) * (bbox_s[3] - bbox_s[1])
        area_t = (bbox_t[2] - bbox_t[0]) * (bbox_t[3] - bbox_t[1])
        score = area_s + area_t
        joint_candidates.append((f, bbox_s, bbox_t, score))

    if len(joint_candidates) >= num_images_joint:
        joint_candidates.sort(key=lambda x: x[3], reverse=True)
        chosen = joint_candidates[:num_images_joint]
        paths = []
        for idx, (frame, bbox_s, bbox_t, _) in enumerate(chosen):
            p = _draw_pair_bbox(scan_dir, scan_id, frame, bbox_s, bbox_t, tmp_dir, idx)
            if p:
                paths.append(p)
        if len(paths) == num_images_joint:
            return paths, "joint"

    # --- separate fallback ---
    src_valid_sorted = sorted(
        src_valid,
        key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]),
        reverse=True
    )
    tgt_valid_sorted = sorted(
        tgt_valid,
        key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]),
        reverse=True
    )

    src_chosen = src_valid_sorted[:num_images_per_obj_separate]
    tgt_chosen = tgt_valid_sorted[:num_images_per_obj_separate]

    paths = []
    for idx, (frame, bbox) in enumerate(src_chosen):
        p = _draw_single_bbox(scan_dir, scan_id, frame, bbox, tmp_dir, is_src=True, idx=idx)
        if p:
            paths.append(p)
    for idx, (frame, bbox) in enumerate(tgt_chosen):
        p = _draw_single_bbox(scan_dir, scan_id, frame, bbox, tmp_dir, is_src=False, idx=idx)
        if p:
            paths.append(p)

    if len(paths) == (num_images_per_obj_separate * 2):
        return paths, "separate"
    return [], "none"


# ---------------------------------------------------------
# 3) vLLM 轻量 VLM：全局单例
# ---------------------------------------------------------

_GLOBAL_VLLM = None
_GLOBAL_PROC = None
_GLOBAL_MODEL_NAME = None

def get_vlm_engine(model_name: str, gpu_memory_utilization: float) -> Tuple[LLM, AutoProcessor]:
    """
    vLLM + AutoProcessor 全局缓存（避免反复初始化）
    注意：如果你想在同一进程里切换 model_name，需要重启进程。
    """
    global _GLOBAL_VLLM, _GLOBAL_PROC, _GLOBAL_MODEL_NAME
    if _GLOBAL_VLLM is None:
        _GLOBAL_MODEL_NAME = model_name
        _GLOBAL_VLLM = LLM(
            model=model_name,
            mm_encoder_tp_mode="data",
            tensor_parallel_size=max(torch.cuda.device_count(), 1),
            enable_expert_parallel=False,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
    else:
        # 防止误用：同一进程切模型会造成 processor/权重不一致
        if _GLOBAL_MODEL_NAME != model_name:
            raise RuntimeError(
                f"VLM engine already initialized with '{_GLOBAL_MODEL_NAME}', "
                f"but you requested '{model_name}'. Please restart the process to switch models."
            )

    if _GLOBAL_PROC is None:
        _GLOBAL_PROC = AutoProcessor.from_pretrained(model_name)

    return _GLOBAL_VLLM, _GLOBAL_PROC


def run_vlm_judge(
    llm,
    proc,
    image_paths: List[str],
    src_label: str,
    tgt_label: str,
    gt_predicate_canon: str,
    pred_predicates: List[str],
    max_new_tokens: int = 160,
) -> Tuple[Dict[str, Any], str]:
    """
    返回: (parsed_json, raw_text)

    期望 JSON：
      {
        "equivalent": true/false,
        "mapped_pred": "<allowed or none>",
        "support_pred": "<one of pred_predicates or ''>"
      }
    """

    pred_predicates_raw = ", ".join([p for p in pred_predicates if _norm_str(p)])

    guide = f"""
You are a strict predicate equivalence judge for indoor visual relationships.

You are given FOUR images from the same scene.
- RED box = SOURCE object (SRC): "{src_label}"
- YELLOW box = TARGET object (TGT): "{tgt_label}"

Details:
There is relation between SRC and TGT, called edge, consisting of source, predicate, target.
Predicate describes the relationship from SRC to TGT, for example, chair is close by table, so chair is src, table is tgt, "is close by" is predicate.

Ground-truth predicate: "{gt_predicate_canon}"

Predicted predicate(s) (may be natural language variants):
{pred_predicates_raw}

Task:
1) Decide whether ANY predicted predicate expresses the similar meaning as the ground-truth predicate.
2) If YES: set "equivalent": true, set "support_pred" to the predicted predicate string that supports it.
3) If NO: set "equivalent": false, set "mapped_pred": "none", "support_pred": "".
4) If you see images, set "ok": true, else false.

Return STRICTLY one JSON object, no extra text:
{{
  "equivalent": true/false,
  "ok": true/false,
  "support_pred": "<one of predicted predicates or empty>"
}}
""".strip()

    content = [{"type": "image", "image": p} for p in image_paths]
    content.append({"type": "text", "text": guide})
    messages = [{"role": "user", "content": content}]

    prompt_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    request = {
        "prompt": prompt_text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        top_k=-1,
        stop_token_ids=[],
    )

    outputs = llm.generate([request], sampling_params)
    raw_text = outputs[0].outputs[0].text
    j = _extract_json_obj(raw_text)

    # 强约束：mapped_pred 必须在 allowed 或 none
    mapped = _norm_str(j.get("mapped_pred", "none")).lower()
    support = _norm_str(j.get("support_pred", ""))

    # equivalent 默认 False
    equivalent = bool(j.get("equivalent", False))
    ok = bool(j.get("ok", False))

    return {
        "equivalent": equivalent,
        "ok":ok,
        "support_pred": support,
    }, raw_text


# ---------------------------------------------------------
# 4) 对外的“一站式”函数：画图->推理->清理
# ---------------------------------------------------------

def judge_predicate_equivalence_for_pair(
    scan_dir: str,
    scan_id: str,
    src_id: str,
    tgt_id: str,
    gt_bboxes: Dict[str, Any],
    src_label: str,
    tgt_label: str,
    gt_predicate_canon: str,
    pred_predicates: List[str],
    tmp_base_dir: str,
    vlm_model: Any,
    vlm_processor: Any,
    min_bbox_area: int = 1000,
    num_images_joint: int = 4,
    num_images_per_obj_separate: int = 2,
    max_new_tokens: int = 160,
) -> Dict[str, Any]:
    """
    返回:
      {
        "ok": bool,                   # 是否成功生成图片并完成推理
        "mode": "joint/separate/none",
        "equivalent": bool,
        "mapped_pred": str,
        "support_pred": str,
        "raw_output": str
      }

    说明：
    - 如果图片生成失败，ok=False（不会报错），raw_output=""
    - 会自动清理本次调用生成的所有临时图片目录
    """
    tmp_dir = _make_unique_tmp_dir(tmp_base_dir)
    try:
        image_paths, mode = prepare_pair_images(
            scan_dir=scan_dir,
            scan_id=scan_id,
            src_id=str(src_id),
            tgt_id=str(tgt_id),
            gt_bboxes=gt_bboxes,
            tmp_dir=tmp_dir,
            min_bbox_area=min_bbox_area,
            num_images_joint=num_images_joint,
            num_images_per_obj_separate=num_images_per_obj_separate,
        )
        if not image_paths:
            return {
                "ok": False,
                "mode": "none",
                "equivalent": False,
                "mapped_pred": "none",
                "support_pred": "",
                "raw_output": "",
            }

        parsed, raw = run_vlm_judge(
            llm=vlm_model,
            proc=vlm_processor,
            image_paths=image_paths,
            src_label=src_label,
            tgt_label=tgt_label,
            gt_predicate_canon=gt_predicate_canon,
            pred_predicates=pred_predicates,
            max_new_tokens=max_new_tokens,
        )

        return {
            "ok": True,
            "mode": mode,
            "equivalent": bool(parsed.get("equivalent", False)),
            "mapped_pred": _norm_str(parsed.get("mapped_pred", "none")),
            "support_pred": _norm_str(parsed.get("support_pred", "")),
            "raw_output": raw,
        }
    finally:
        _safe_rmtree(tmp_dir)
