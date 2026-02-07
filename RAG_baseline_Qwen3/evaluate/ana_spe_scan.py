#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eva_single_scan_with_bboxes.py

Purpose:
- Evaluate ONE scan like eva_final_new.py (same single_scan_detail JSON schema, unchanged)
- Export a folder: <OUT_ROOT>/<scan_id>/
    - edge_eval_<scan_id>.json
    - bbox_images/*.jpg  (GT bbox visualizations with GT id/label + mapped Pred id/label)

Borrowed/aligned logic:
- Evaluation core + schema from eva_final_new.py :contentReference[oaicite:2]{index=2}
- GT bbox loading + rectangle drawing style from eva_qwen8b.py :contentReference[oaicite:3]{index=3}
"""

import os
import json
import ast
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict, Counter

from PIL import Image, ImageDraw, ImageFont


# ===================== 用户只需改这里 =====================

SINGLE_SCAN_ID = "422885b3-192d-25fc-84c9-9b80eea1752d"

# GT / Pred / gt2pred 路径（沿用你当前工程习惯）
GT_SG_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan"
PRED_SG_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/RAG_baseline_Qwen3/RAG_end2end_CoT_agentic"
GT2PRED_RESULT_JSONL = "/hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_baseline_Qwen3/evaluate/CoT_agentic_0/obj_match_results.jsonl"

# 你的要求：在这个目录下建 <scan_id>/ 子文件夹
OUT_ROOT = "/hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_baseline_Qwen3/evaluate/CoT_agentic_0"

# bbox 来源（与你 eva_qwen8b.py 一致：每个 scan 下都有 {scan_id}_gt_bboxes.json） :contentReference[oaicite:4]{index=4}
SCAN_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan"

# 每个 GT obj 导出多少帧 bbox 图
MAX_IMAGES_PER_OBJ = 5
MIN_BBOX_AREA = 1000

# ==========================================================


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(s: str) -> str:
    return str(s).strip().lower() if s else ""


def normalize_predicate(p) -> str:
    if p is None:
        return ""
    if isinstance(p, list):
        if not p:
            return ""
        p = p[0]
    s = str(p).strip().lower()
    mapping = {
        "in front of": "front",
        "infront of": "front",
        "infront": "front",
        "behind of": "behind",
        "on top of": "on top of",
        "next to": "close by",
        "near": "close by",
    }
    return mapping.get(s, s)


def parse_c_edge_multi(rel_value) -> List[str]:
    if rel_value is None:
        return []
    if isinstance(rel_value, list):
        return [str(x).strip() for x in rel_value if str(x).strip()]
    if isinstance(rel_value, str):
        s = rel_value.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        parts = [p.strip().strip("[]'\"") for p in s.split(",")]
        return [p for p in parts if p]
    v = str(rel_value).strip()
    return [v] if v else []


def build_pred_edge_triples(pred_sg: dict) -> Set[Tuple[str, str, str]]:
    triples: Set[Tuple[str, str, str]] = set()
    edges = pred_sg.get("edges", []) or []
    for e in edges:
        s = str(e.get("src"))
        t = str(e.get("tgt"))
        if not s or not t:
            continue
        rel_value = e.get("relationship")
        if rel_value is None:
            rel_value = e.get("c_edge")
        for r in parse_c_edge_multi(rel_value):
            rel_norm = normalize_predicate(r)
            if rel_norm:
                triples.add((s, rel_norm, t))
    return triples


def build_pred_pair_to_rels(pred_sg: dict) -> Dict[Tuple[str, str], Set[str]]:
    pair2rels: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    edges = pred_sg.get("edges", []) or []
    for e in edges:
        s = str(e.get("src"))
        t = str(e.get("tgt"))
        if not s or not t:
            continue
        rel_value = e.get("relationship")
        if rel_value is None:
            rel_value = e.get("c_edge")
        for r in parse_c_edge_multi(rel_value):
            rel_norm = normalize_predicate(r)
            if rel_norm:
                pair2rels[(s, t)].add(rel_norm)
    return pair2rels


def load_gt2pred_from_jsonl(path: str) -> Dict[str, Dict[str, str]]:
    """
    与 eva_final_new.py 一致：兼容 scan-level(gt2pred) 和 old-style per-gt 行 :contentReference[oaicite:5]{index=5}
    """
    scan_map: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            scan_id = str(rec.get("scan_id", "")).strip()
            if not scan_id:
                continue

            if "gt2pred" in rec:
                gt2pred = {str(k): str(v) for k, v in rec["gt2pred"].items()}
                scan_map.setdefault(scan_id, {}).update(gt2pred)
                continue

            if "gt_obj_id" in rec and "match_pred_node_id" in rec:
                gid = str(rec["gt_obj_id"])
                pid = str(rec["match_pred_node_id"])
                scan_map.setdefault(scan_id, {})[gid] = pid

    return scan_map


def evaluate_single_scan_pairlevel_details(
    scan_id: str,
    gt_sg: dict,
    pred_sg: dict,
    gt2pred: Dict[str, str],
) -> Dict[str, Any]:
    """
    输出 single_scan_detail（schema 与你现在 eva_final_new.py 的单 scan 输出一致，不变） :contentReference[oaicite:6]{index=6}
    并且 edge_match_details 是你最近改的 “按 (src,tgt) pair 聚合” 版本。
    """
    gt_nodes = gt_sg.get("nodes", []) or []
    pred_nodes = pred_sg.get("nodes", []) or []

    # ----- obj -----
    obj_total = 0
    obj_hit = 0
    obj_total_per_label = Counter()
    obj_hit_per_label = Counter()

    for n in gt_nodes:
        gid = str(n.get("id"))
        lab = normalize_label(n.get("label") or n.get("name"))
        if not gid:
            continue
        obj_total += 1
        obj_total_per_label[lab] += 1
        pid = gt2pred.get(gid, "none")
        if pid and str(pid).lower() != "none":
            obj_hit += 1
            obj_hit_per_label[lab] += 1

    def safe_div(a, b):
        return float(a) / b if b > 0 else 0.0

    obj_recall = safe_div(obj_hit, obj_total)

    # ----- edges -----
    gt_edges = gt_sg.get("edges", []) or []
    pred_edge_triples = build_pred_edge_triples(pred_sg)
    pred_pair2rels = build_pred_pair_to_rels(pred_sg)

    edge_total_all = 0
    edge_total_both = 0
    edge_hit = 0
    edge_per_rel_total = Counter()
    edge_per_rel_hit = Counter()

    gt_node_label_map = {
        str(n.get("id")): (n.get("label") or n.get("name") or "unknown")
        for n in gt_nodes
    }
    pred_node_label_map = {
        str(n.get("id")): (n.get("label") or n.get("name") or "unknown")
        for n in pred_nodes
    }

    gt_pair2preds_raw: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    gt_pair2preds_norm: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for e in gt_edges:
        s_gt = str(e.get("src"))
        t_gt = str(e.get("tgt"))
        rel_raw = e.get("relationship") or e.get("predicate")
        rel_norm = normalize_predicate(rel_raw)
        if not s_gt or not t_gt or not rel_norm:
            continue
        edge_total_all += 1
        gt_pair2preds_raw[(s_gt, t_gt)].append(rel_raw)
        gt_pair2preds_norm[(s_gt, t_gt)].append(rel_norm)

    edge_match_details: List[Dict[str, Any]] = []

    for (s_gt, t_gt), rels_norm_list in gt_pair2preds_norm.items():
        rels_raw_list = gt_pair2preds_raw[(s_gt, t_gt)]

        pred_src = str(gt2pred.get(s_gt, "none") or "none").strip()
        pred_tgt = str(gt2pred.get(t_gt, "none") or "none").strip()

        src_hit = pred_src.lower() != "none"
        tgt_hit = pred_tgt.lower() != "none"

        src_gt_label = gt_node_label_map.get(s_gt, "unknown")
        tgt_gt_label = gt_node_label_map.get(t_gt, "unknown")

        if src_hit:
            src_pred_label = pred_node_label_map.get(pred_src, "unknown")
            src_pred_id_and_label = f"{pred_src},{src_pred_label}"
        else:
            src_pred_id_and_label = "none,none"

        if tgt_hit:
            tgt_pred_label = pred_node_label_map.get(pred_tgt, "unknown")
            tgt_pred_id_and_label = f"{pred_tgt},{tgt_pred_label}"
        else:
            tgt_pred_id_and_label = "none,none"

        pred_rels_norm = sorted(list(pred_pair2rels.get((pred_src, pred_tgt), set()))) if (src_hit and tgt_hit) else []

        hit_rels_norm: List[str] = []
        if src_hit and tgt_hit:
            for rel_norm in rels_norm_list:
                edge_total_both += 1
                edge_per_rel_total[rel_norm] += 1
                if (pred_src, rel_norm, pred_tgt) in pred_edge_triples:
                    hit_rels_norm.append(rel_norm)
                    edge_hit += 1
                    edge_per_rel_hit[rel_norm] += 1

        predicate_hit = len(hit_rels_norm) > 0

        edge_match_details.append(
            {
                "src_gt_id_and_label": f"{s_gt},{src_gt_label}",
                "tgt_gt_id_and_label": f"{t_gt},{tgt_gt_label}",
                "gt_predicates": rels_raw_list,
                "src_pred_id_and_label": src_pred_id_and_label,
                "tgt_pred_id_and_label": tgt_pred_id_and_label,
                "pred_predicates_norm": pred_rels_norm,
                "predicate_hit_predicates_norm": sorted(list(set(hit_rels_norm))),
                "src_hit": src_hit,
                "tgt_hit": tgt_hit,
                "predicate_hit": predicate_hit,
                "edge_hit": predicate_hit,
            }
        )

    edge_recall = safe_div(edge_hit, edge_total_all)
    both_true_edge_recall = safe_div(edge_hit, edge_total_both)

    single_scan_detail = {
        "scan_id": scan_id,
        "num_gt_obj": obj_total,
        "num_pred_obj": len(pred_nodes),
        "obj_recall": obj_recall,
        "num_gt_edge_all": edge_total_all,
        "num_gt_edge_both_true": edge_total_both,
        "num_matched_edge": edge_hit,
        "edge_recall": edge_recall,
        "both_true_edge_recall": both_true_edge_recall,
        "edge_per_rel_total": dict(edge_per_rel_total),
        "edge_per_rel_hit": dict(edge_per_rel_hit),
        "edge_match_details": edge_match_details,
    }
    return single_scan_detail


def _get_font():
    # HPC 上常见没有字体；失败就退回默认
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        return ImageFont.load_default()

import colorsys
from PIL import Image, ImageDraw, ImageFont

def _get_font_bold(size=20):
    # 和 experience_abstraction.py 一样优先用 DejaVuSans-Bold :contentReference[oaicite:2]{index=2}
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

def _idx_to_color(idx: int) -> tuple:
    # Golden-ratio hue cycling，和 experience_abstraction.py 同思路 :contentReference[oaicite:3]{index=3}
    h = (idx * 0.61803398875) % 1.0
    s = 0.75
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def export_annotated_frames_all_objects(
    scan_id: str,
    gt_sg: dict,
    pred_sg: dict,
    gt2pred: dict,
    out_dir: str,
    scan_dataset_dir: str,
    max_frames: int = 50,
    min_area: int = 1000,
):
    """
    生成“每帧一张图，图中标出该帧所有 obj”的标注图。
    - bbox 数据来自: {scan_dataset_dir}/{scan_id}/{scan_id}_gt_bboxes.json :contentReference[oaicite:4]{index=4}
    - 原图来自: {scan_dataset_dir}/{scan_id}/sequence/frame-XXXXXX.color.jpg :contentReference[oaicite:5]{index=5}
    输出:
      out_dir/annotated_frames/frame-XXXXXX.jpg
    """
    os.makedirs(out_dir, exist_ok=True)
    save_dir = os.path.join(out_dir, "annotated_frames")
    os.makedirs(save_dir, exist_ok=True)

    # 1) 读 GT bbox json
    bbox_json = os.path.join(scan_dataset_dir, scan_id, f"{scan_id}_gt_bboxes.json")
    if not os.path.exists(bbox_json):
        raise FileNotFoundError(f"GT bbox json not found: {bbox_json}")
    gt_bboxes_all = load_json(bbox_json)
    if scan_id not in gt_bboxes_all:
        raise KeyError(f"Scan {scan_id} not found in bbox json: {bbox_json}")

    gt_bboxes = gt_bboxes_all[scan_id]  # dict: gt_obj_id -> {seen_frames, bounding_boxes}

    # 2) label map（GT + Pred），用于把 id 和名字写出来
    gt_label_map = {str(n.get("id")): (n.get("label") or n.get("name") or "unknown")
                    for n in (gt_sg.get("nodes", []) or [])}
    pred_label_map = {str(n.get("id")): (n.get("label") or n.get("name") or "unknown")
                      for n in (pred_sg.get("nodes", []) or [])}

    # 3) 建一个 frame -> [(gt_id, gt_name, bbox)] 的索引
    # gt_bboxes 里保存的是 seen_frames 列表（形如 frame-000123）及其对应 bbox
    frame2objs = {}  # frame_str -> list
    for gt_id, info in gt_bboxes.items():
        gt_id = str(gt_id)
        seen_frames = info.get("seen_frames", []) or []
        bboxes = info.get("bounding_boxes", []) or []
        if len(seen_frames) != len(bboxes):
            continue

        gt_name = gt_label_map.get(gt_id, "unknown")

        for f_str, bbox in zip(seen_frames, bboxes):
            # bbox: [x1,y1,x2,y2]
            try:
                x1, y1, x2, y2 = bbox
                area = max(0, x2 - x1) * max(0, y2 - y1)
            except Exception:
                continue
            if area < min_area:
                continue
            frame2objs.setdefault(f_str, []).append((gt_id, gt_name, bbox))

    # 4) 选最多 max_frames 帧导出：优先选择“obj 数最多”的帧（更有信息量）
    all_frames = sorted(frame2objs.keys())
    all_frames.sort(key=lambda f: len(frame2objs.get(f, [])), reverse=True)
    chosen_frames = all_frames[:max_frames]

    font = _get_font_bold(size=20)

    for f_str in chosen_frames:
        # f_str 形如 "frame-000123"
        img_path = os.path.join(scan_dataset_dir, scan_id, "sequence", f"{f_str}.color.jpg")
        if not os.path.exists(img_path):
            # 兼容 png/jpeg
            img_path = os.path.join(scan_dataset_dir, scan_id, "sequence", f"{f_str}.color.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(scan_dataset_dir, scan_id, "sequence", f"{f_str}.color.jpeg")
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 用于避免文字重叠：借鉴 experience_abstraction 的策略，label 位置冲突就往下挪 :contentReference[oaicite:6]{index=6}
        placed_label_boxes = []

        def overlaps(b1, b2):
            x1, y1, x2, y2 = b1
            a1, b1_, a2, b2_ = b2
            return not (x2 < a1 or a2 < x1 or y2 < b1_ or b2_ < y1)

        def find_non_overlapping_pos(x, y, text, margin=2):
            x0, y0, x1, y1 = draw.textbbox((x, y), text, font=font)
            h = y1 - y0
            tx1, ty1, tx2, ty2 = x0, y0, x1, y1
            while any(overlaps((tx1, ty1, tx2, ty2), b) for b in placed_label_boxes):
                ty1 += h + margin
                ty2 += h + margin
            return tx1, ty1, tx2, ty2

        # 为每个 obj 固定颜色：按 gt_id 排序，保证每次颜色稳定
        objs = frame2objs.get(f_str, [])
        objs = sorted(objs, key=lambda x: int(x[0]) if x[0].isdigit() else 10**9)

        for color_idx, (gt_id, gt_name, bbox) in enumerate(objs):
            color = _idx_to_color(color_idx)

            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 拼一行文本：GT + Pred 映射（这样你对照图能直接看到 mapping）
            pred_id = str(gt2pred.get(gt_id, "none") or "none").strip()
            pred_name = pred_label_map.get(pred_id, "unknown") if pred_id.lower() != "none" else "none"
            text = f"{gt_id}:{gt_name}  ->  {pred_id}:{pred_name}"

            tx1, ty1, tx2, ty2 = find_non_overlapping_pos(x1, y1, text)
            placed_label_boxes.append((tx1, ty1, tx2, ty2))
            draw.text((tx1 + 2, ty1), text, fill=color, font=font)

        out_path = os.path.join(save_dir, f"{f_str}.jpg")
        img.save(out_path, quality=95)

    print(f"[OK] Saved annotated frames: {save_dir}")


def main():
    scan_id = SINGLE_SCAN_ID

    # 1) load gt2pred
    gt2pred_all = load_gt2pred_from_jsonl(GT2PRED_RESULT_JSONL)
    if scan_id not in gt2pred_all:
        raise KeyError(f"Scan {scan_id} not found in gt2pred jsonl: {GT2PRED_RESULT_JSONL}")
    gt2pred = gt2pred_all[scan_id]

    # 2) load GT/Pred SG
    gt_path = os.path.join(GT_SG_DIR, scan_id, f"{scan_id}_scan_sg.json")
    pred_path = os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(gt_path)
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)

    gt_sg = load_json(gt_path)
    pred_sg = load_json(pred_path)

    # 3) create output folder: OUT_ROOT/<scan_id>/
    scan_out_dir = os.path.join(OUT_ROOT, scan_id)
    os.makedirs(scan_out_dir, exist_ok=True)

    # 4) evaluate single scan -> JSON (schema unchanged)
    single_scan_detail = evaluate_single_scan_pairlevel_details(scan_id, gt_sg, pred_sg, gt2pred)

    out_json = os.path.join(scan_out_dir, f"edge_eval_{scan_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(single_scan_detail, f, indent=2, ensure_ascii=False)

    # 5) export bbox images for ALL GT objects
    export_annotated_frames_all_objects(
        scan_id=scan_id,
        gt_sg=gt_sg,
        pred_sg=pred_sg,
        gt2pred=gt2pred,
        out_dir=scan_out_dir,                     # OUT_ROOT/<scan_id>/
        scan_dataset_dir=SCAN_DIR,                # /hkfs/.../data/3RScan
        max_frames=50,                            # 你想导出多少帧（按“obj最多”优先）
        min_area=1000
    )


    print(f"[OK] Wrote: {out_json}")
    print(f"[OK] BBox images in: {os.path.join(scan_out_dir, 'bbox_images')}")


if __name__ == "__main__":
    main()
