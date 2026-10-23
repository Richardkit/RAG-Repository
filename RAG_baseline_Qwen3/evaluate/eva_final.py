#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eva_final_new_v2.py

Global multi-scan final evaluation script:
- Evaluate obj / edge recall & mrecall using gt2pred one-to-one matching results
- Support multi-predicate list logic for c_edge in predicted SG
- For edges, simultaneously provide:
    1) edge_recall:     denominator is all valid GT edges
    2) both_true_recall: denominator is GT edges where both src and tgt are matched (legacy meaning)
- Optional: output detailed per-edge hit information json for SINGLE_SCAN_ID
"""

import os
import json
import ast
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any


# ===================== Path Configuration (please modify according to actual situation) =====================
GT_SG_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan"
PRED_SG_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/Ablation/ab_scale_of knowledge_base_thr_0.5_w_weight/kb_0"
GT2PRED_RESULT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/scripts_thr_0.5_weight_kb_scale/scale_0/obj_match_results.jsonl"
EVA_DIR = os.path.dirname(GT2PRED_RESULT)
GLOBAL_OUT_JSON = os.path.join(EVA_DIR, "eva_final_results.json")
# If you want to additionally output detailed edge hit information for a specific scan, fill in scan_id here;
# Set to "" or None if not needed
SINGLE_SCAN_ID = None # e.g. "0cf75f4b-564d-23f5-899f-ff97701d3f3b"
SINGLE_SCAN_RECORD_SAVE = EVA_DIR  # single scan output: edge_eval_<scan_id>.json

TMP_IMG_SAVE = "/hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_baseline_Qwen3/evaluate/tmp_img_save"
MODEL_NAME = "/hkfs/work/workspace/scratch/rg5174-ragspace/models/Qwen3-VL-32B-Instruct"
USE_VLM_JUDGE = True

# =======================================================================
if USE_VLM_JUDGE:
    from predicates_judge import judge_predicate_equivalence_for_pair, get_vlm_engine
else:
    judge_predicate_equivalence_for_pair = None
    get_vlm_engine = None

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(s: str) -> str:
    """Normalize object label / name"""
    if not s:
        return ""
    return str(s).strip().lower()


def normalize_predicate(p) -> str:
    """Normalize predicate, compatible with list / non-string input"""
    if p is None:
        return ""
    if isinstance(p, list):
        # guard: if input is already a list, take only the first element
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
    """
    Parse c_edge / relationship in predicted edges:
      - If it is already a list: ["standing on", "attached to"]
      - If it is a string-form list: "['standing on', 'attached to']"
      - Otherwise treat as a single predicate
    """
    if rel_value is None:
        return []

    # already a list
    if isinstance(rel_value, list):
        return [str(x).strip() for x in rel_value if str(x).strip()]

    # string: may be "['a', 'b']"
    if isinstance(rel_value, str):
        s = rel_value.strip()
        if not s:
            return []
        # try literal_eval
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        # fall back to splitting by comma
        parts = [p.strip().strip("[]'\"") for p in s.split(",")]
        return [p for p in parts if p]

    # other types: convert directly to a single string
    v = str(rel_value).strip()
    return [v] if v else []


# ===================== Read gt2pred Results =====================

def load_gt2pred_from_jsonl(path: str) -> Dict[str, Dict[str, str]]:
    """
    Read scan_id -> {gt_obj_id -> pred_node_id} mapping from a jsonl file.
    Compatible with two line formats:
      1) {"scan_id": ..., "gt2pred": {...}, ...}
      2) {"scan_id": ..., "gt_obj_id": "...", "match_pred_node_id": "...", ...}
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

            # format 1: with gt2pred (scan-level)
            if "gt2pred" in rec:
                gt2pred = {str(k): str(v) for k, v in rec["gt2pred"].items()}
                scan_map.setdefault(scan_id, {}).update(gt2pred)
                continue

            # format 2: old format, one gt per line
            if "gt_obj_id" in rec and "match_pred_node_id" in rec:
                gid = str(rec["gt_obj_id"])
                pid = str(rec["match_pred_node_id"])
                scan_map.setdefault(scan_id, {})[gid] = pid

    return scan_map


def load_gt2pred(path: str) -> Dict[str, Dict[str, str]]:
    """
    Unified entry point: path can be json or jsonl.
    - json: list[{"scan_id":..., "gt2pred": {...}}, ...]
    - jsonl: one dict per line (see description above)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".jsonl"):
        return load_gt2pred_from_jsonl(path)

    data = load_json(path)
    scan_map: Dict[str, Dict[str, str]] = {}
    if isinstance(data, list):
        for rec in data:
            scan_id = str(rec.get("scan_id", "")).strip()
            gt2pred = rec.get("gt2pred") or {}
            if not scan_id:
                continue
            scan_map.setdefault(scan_id, {}).update(
                {str(k): str(v) for k, v in gt2pred.items()}
            )
    elif isinstance(data, dict):
        for scan_id, gt2pred in data.items():
            scan_map[str(scan_id)] = {str(k): str(v) for k, v in gt2pred.items()}
    else:
        raise ValueError("Unsupported JSON format for gt2pred")

    return scan_map


# ===================== Pred Edges -> Triples =====================
def build_pred_pair2preds(pred_sg: dict):
    """
    Build two maps:
      - directed:   (src, tgt) -> set(predicates)
      - undirected: (min(src,tgt), max(src,tgt)) -> set(predicates)
    Supports multi-predicate c_edge list (same as build_pred_edge_triples).
    """
    directed = defaultdict(set)
    undirected = defaultdict(set)

    edges = pred_sg.get("edges", []) or []
    for e in edges:
        s = str(e.get("src")).strip()
        t = str(e.get("tgt")).strip()
        if not s or not t:
            continue

        rel_value = e.get("relationship")
        if rel_value is None:
            rel_value = e.get("c_edge")

        pred_list = parse_c_edge_multi(rel_value)
        for r in pred_list:
            rel_norm = normalize_predicate(r)
            if not rel_norm:
                continue
            directed[(s, t)].add(rel_norm)
            a, b = (s, t) if s <= t else (t, s)
            undirected[(a, b)].add(rel_norm)

    return directed, undirected

def build_pred_edge_triples(pred_sg: dict) -> Set[Tuple[str, str, str]]:
    """
    Build all (src, predicate_norm, tgt) triples from the predicted scene graph.
    Supports one edge corresponding to multiple predicates (from c_edge or relationship).
    """
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

        pred_list = parse_c_edge_multi(rel_value)
        for r in pred_list:
            rel_norm = normalize_predicate(r)
            if rel_norm:
                triples.add((s, rel_norm, t))

    return triples

def build_pred_pair_to_rels(pred_sg: dict) -> Dict[Tuple[str, str], Set[str]]:
    """
    Build (src_pred, tgt_pred) -> set(predicate_norm) from the predicted scene graph
    """
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

        pred_list = parse_c_edge_multi(rel_value)
        for r in pred_list:
            rel_norm = normalize_predicate(r)
            if rel_norm:
                pair2rels[(s, t)].add(rel_norm)
    return pair2rels

# ===================== Evaluate Single Scan =====================

def evaluate_single_scan(
    scan_id: str,
    gt_sg: dict,
    pred_sg: dict,
    gt2pred: Dict[str, str],
    llm: Any,
    proc: Any,
    want_edge_details: bool = False,
):
    """
    Returns:
      obj_total, obj_hit, obj_per_label_total, obj_per_label_hit,
      edge_total_all, edge_total_both, edge_hit,
      edge_per_rel_total, edge_per_rel_hit,
      edge_match_details (None if want_edge_details=False)

    Notes:
      - edge_total_all  : number of all valid GT edges (src/tgt/predicate all present)
      - edge_total_both : number of GT edges where both src and tgt are mapped (old script's edge_total)
      - edge_hit        : number of GT edges with fully matched triples
    """

    # ---------- Objects ----------
    gt_nodes = gt_sg.get("nodes", []) or []
    pred_nodes = pred_sg.get("nodes", []) or []
    hit_edges: List[Dict[str, Any]] = []

    obj_total = 0
    obj_hit = 0
    obj_per_label_total = Counter()
    obj_per_label_hit = Counter()
    gt_bboxes_path = os.path.join(GT_SG_DIR, scan_id, f"{scan_id}_gt_bboxes.json")
    with open(gt_bboxes_path, "r", encoding="utf-8") as f:
        gt_bboxes = json.load(f)

    for n in gt_nodes:
        gid = str(n.get("id"))
        label = normalize_label(n.get("label") or n.get("name"))
        if not gid:
            continue

        obj_total += 1
        obj_per_label_total[label] += 1

        pred_id = gt2pred.get(gid, "none")
        if pred_id and pred_id.lower() != "none":
            obj_hit += 1
            obj_per_label_hit[label] += 1

    # ---------- Relations ----------
    gt_edges = gt_sg.get("edges", []) or []
    pred_edge_triples = build_pred_edge_triples(pred_sg)
    pred_pair2preds_dir, pred_pair2preds_undir = build_pred_pair2preds(pred_sg)
    pred_pair2rels = build_pred_pair_to_rels(pred_sg)

    edge_total_all = 0         # all valid GT edges (per-edge count)
    edge_total_both = 0        # GT edges where both src/tgt are mapped (per-edge count)
    edge_hit = 0               # GT edges with fully matched triples (per-edge count)
    edge_total_old = 0         # # old triplet denominator: node both true and (pred_src,pred_tgt) pair exists in pred_edges
    both_true_no_edge_cnt = 0
    predicate_total_all = 0
    predicate_total_both = 0
    predicate_hit_directed = 0
    predicate_hit_undirected = 0
    edge_per_rel_total = Counter()
    edge_per_rel_hit = Counter()

    edge_match_details: List[Dict[str, Any]] = []

    # record GT node id -> label for detail output
    gt_node_label_map = {
        str(n.get("id")): (n.get("label") or n.get("name") or "unknown")
        for n in gt_nodes
    }

    # record Pred node id -> label for detail output (note: pred_sg label field may be 'label' or 'name')
    pred_node_label_map = {
        str(n.get("id")): (n.get("label") or n.get("name") or "unknown")
        for n in pred_nodes
    }

    # --------- (A) aggregate GT edges into pair -> multi-predicate ---------
    # key: (src_gt, tgt_gt)
    gt_pair2preds_raw: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    gt_pair2preds_norm: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for e in gt_edges:
        s_gt = str(e.get("src"))
        t_gt = str(e.get("tgt"))
        rel_gt_raw = e.get("relationship") or e.get("predicate")
        rel_gt_norm = normalize_predicate(rel_gt_raw)

        # skip invalid edges (missing src/tgt/predicate)
        if not s_gt or not t_gt or not rel_gt_norm:
            continue

        # still count per-edge denominator (preserving the original recall definition)
        edge_total_all += 1
        predicate_total_all += 1

        gt_pair2preds_raw[(s_gt, t_gt)].append(rel_gt_raw)
        gt_pair2preds_norm[(s_gt, t_gt)].append(rel_gt_norm)

    # --------- (B) output details by pair; accumulate per-edge hit statistics ---------
    for (s_gt, t_gt), rels_norm_list in gt_pair2preds_norm.items():
        rels_raw_list = gt_pair2preds_raw[(s_gt, t_gt)]

        # GT -> Pred mapping (consistent at pair granularity)
        pred_src = gt2pred.get(s_gt, "none")
        pred_tgt = gt2pred.get(t_gt, "none")

        pred_src = str(pred_src).strip() if pred_src is not None else "none"
        pred_tgt = str(pred_tgt).strip() if pred_tgt is not None else "none"
        pred_src = (pred_src or "none").strip()
        pred_tgt = (pred_tgt or "none").strip()

        src_hit = bool(pred_src) and pred_src.lower() != "none"
        tgt_hit = bool(pred_tgt) and pred_tgt.lower() != "none"

        # combine pred id + label
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

        # all predicates (norm) for this (pred_src, pred_tgt) on the predicted side
        pred_rels_norm = sorted(list(pred_pair2rels.get((pred_src, pred_tgt), set()))) if (src_hit and tgt_hit) else []
        pred_rels_norm_copy = pred_rels_norm.copy()

        # set of matched predicates (norm)
        hit_rels_norm: List[str] = []
        used_judge_pred_predicate = set()
        if src_hit and tgt_hit:
            # note: pair-level hit list; per-edge edge_hit/edge_per_rel_hit still accumulated per original definition
            for rel_norm in rels_norm_list:
                # per-edge both_true denominator and per-rel denominator follow the original definition
                edge_total_both += 1
                edge_per_rel_total[rel_norm] += 1
                if len(pred_rels_norm) == 0:
                    both_true_no_edge_cnt += 1
                    continue
                # old-logic GT edge denominator (only GT edges with both src/tgt mapped and at least one predicate count)
                if len(pred_rels_norm) > 0:
                    edge_total_old += 1

                # ---------- predicate recall (pair-conditioned) ----------
                pred_rels_dir = pred_pair2rels.get((pred_src, pred_tgt), set())
                pred_rels_undir = pred_rels_dir | pred_pair2rels.get((pred_tgt, pred_src), set())

                is_predicate_hit_directed = (rel_norm in pred_rels_dir)
                is_predicate_hit_undirected = (rel_norm in pred_rels_undir)

                predicate_total_both += 1
                if is_predicate_hit_directed:
                    predicate_hit_directed += 1
                if is_predicate_hit_undirected:
                    predicate_hit_undirected += 1

                # # 1) hard hit: exact match
                # if (pred_src, rel_norm, pred_tgt) in pred_edge_triples:
                #     hit_rels_norm.append(rel_norm)
                #     edge_hit += 1
                #     edge_per_rel_hit[rel_norm] += 1
                #     if rel_norm in pred_rels_norm_copy:
                #         pred_rels_norm_copy.remove(rel_norm)

                # # 2) temporary rule: GT=left allows pred=right; GT=right allows pred=left
                # elif rel_norm in ("left", "right"):
                #     flipped = "right" if rel_norm == "left" else "left"
                #     if (pred_src, flipped, pred_tgt) in pred_edge_triples:
                #         # still count as hit under GT's rel_norm (because it is counted as a hit)
                #         hit_rels_norm.append(rel_norm)
                #         edge_hit += 1
                #         edge_per_rel_hit[rel_norm] += 1

                #         # consume the flipped entry to prevent the same pred from being reused
                #         if flipped in pred_rels_norm_copy:
                #             pred_rels_norm_copy.remove(flipped)

                # else: (add judge recovery logic here if needed)

                if (pred_src, rel_norm, pred_tgt) in pred_edge_triples and rel_norm in pred_rels_norm_copy:
                    hit_rels_norm.append(rel_norm)

                    # per-edge hit statistics
                    edge_hit += 1

                    hit_edges.append({
                        "gt_src_id": s_gt,
                        "gt_tgt_id": t_gt,
                        "gt_src_label": src_gt_label,
                        "gt_tgt_label": tgt_gt_label,
                        "gt_predicate_norm": rel_norm,
                        "pred_src_id": pred_src,
                        "pred_tgt_id": pred_tgt,
                        "pred_src_label": pred_node_label_map.get(pred_src, "unknown"),
                        "pred_tgt_label": pred_node_label_map.get(pred_tgt, "unknown"),
                        "matched_by": "hard",
                        "pred_predicates_norm": pred_rels_norm,   # all predicates given by the model for this pair
                    })
                    edge_per_rel_hit[rel_norm] += 1

                    if rel_norm not in pred_rels_norm_copy:
                        print("[REMOVE_FAIL]",
                            "scan", scan_id,
                            "gt_pair", (s_gt, t_gt),
                            "pred_pair", (pred_src, pred_tgt),
                            "rel_norm", rel_norm,
                            "pred_rels_norm", pred_rels_norm,
                            "rels_norm_list", rels_norm_list)
                    pred_rels_norm_copy.remove(rel_norm)

                else:
                    if pred_rels_norm_copy and USE_VLM_JUDGE:
                        res = judge_predicate_equivalence_for_pair(
                            scan_dir=GT_SG_DIR, scan_id=scan_id,
                            src_id=s_gt, tgt_id=t_gt,
                            gt_bboxes=gt_bboxes, src_label=src_gt_label, tgt_label=tgt_gt_label,
                            gt_predicate_canon=rel_norm, pred_predicates=pred_rels_norm_copy,
                            tmp_base_dir=TMP_IMG_SAVE, vlm_model=llm, vlm_processor=proc
                        )

                        if res["equivalent"] and res["ok"]:
                            hit_rels_norm.append(rel_norm)
                            edge_hit += 1

                            hit_edges.append({
                                "gt_src_id": s_gt,
                                "gt_tgt_id": t_gt,
                                "gt_src_label": src_gt_label,
                                "gt_tgt_label": tgt_gt_label,
                                "gt_predicate_norm": rel_norm,
                                "pred_src_id": pred_src,
                                "pred_tgt_id": pred_tgt,
                                "pred_src_label": pred_node_label_map.get(pred_src, "unknown"),
                                "pred_tgt_label": pred_node_label_map.get(pred_tgt, "unknown"),
                                "matched_by": "vlm_judge",
                                "support_pred": normalize_predicate(res.get("support_pred", "")),
                                "pred_predicates_norm": pred_rels_norm,
                                "gt_edge_key": f"{normalize_label(src_gt_label)}||{rel_norm}||{normalize_label(tgt_gt_label)}"
                            })

                            edge_per_rel_hit[rel_norm] += 1
                            support = normalize_predicate(res.get("support_pred", ""))
                            if support in pred_rels_norm_copy:
                                pred_rels_norm_copy.remove(support)


        predicate_hit = len(hit_rels_norm) > 0
        edge_hit_flag = predicate_hit  # pair-level: any matched predicate counts as True

        if want_edge_details:
            edge_match_details.append(
                {
                    "src_gt_id_and_label": f"{s_gt},{src_gt_label}",
                    "tgt_gt_id_and_label": f"{t_gt},{tgt_gt_label}",
                    # GT side multi-predicate: both original and norm provided
                    "gt_predicates": rels_raw_list,
                    # Pred side id+label
                    "src_pred_id_and_label": src_pred_id_and_label,
                    "tgt_pred_id_and_label": tgt_pred_id_and_label,
                    # predicates (norm) for this pair on the predicted side
                    "pred_predicates_norm": pred_rels_norm,
                    # matched predicates (norm)
                    "predicate_hit_predicates_norm": sorted(list(set(hit_rels_norm))),
                    "src_hit": src_hit,
                    "tgt_hit": tgt_hit,
                    "predicate_hit": predicate_hit,
                    "edge_hit": edge_hit_flag,
                }
            )

    return (
        obj_total,
        obj_hit,
        obj_per_label_total,
        obj_per_label_hit,
        edge_total_all,
        edge_total_both,
        edge_hit,
        edge_per_rel_total,
        edge_per_rel_hit,
        edge_match_details if want_edge_details else None,
        both_true_no_edge_cnt,
        edge_total_old,

        # --- NEW: predicate stats ---
        predicate_total_all,
        predicate_total_both,
        predicate_hit_directed,
        predicate_hit_undirected,

        hit_edges
    )


# ===================== Main Evaluation Pipeline =====================

def main():
    gt2pred_all = load_gt2pred(GT2PRED_RESULT)
    scan_ids = sorted(gt2pred_all.keys())

    llm, proc = None, None
    if USE_VLM_JUDGE:
        llm, proc = get_vlm_engine(MODEL_NAME, gpu_memory_utilization=0.8)

    print(f"[INFO] Loaded gt2pred for {len(scan_ids)} scans")

    # global statistics
    global_obj_total = 0
    global_obj_hit = 0

    global_edge_total_all = 0      # all valid GT edges
    global_edge_total_both = 0     # GT edges where both src/tgt are mapped
    global_edge_hit = 0
    global_both_true_no_edge_cnt = 0
    global_edge_total_old = 0
    global_pred_total_all = 0
    global_pred_total_both = 0
    global_pred_hit_dir = 0
    global_pred_hit_undir = 0


    global_obj_total_per_label = Counter()
    global_obj_hit_per_label = Counter()
    global_edge_total_per_rel = Counter()
    global_edge_hit_per_rel = Counter()

    single_scan_detail: Any = None  # detailed result for SINGLE_SCAN_ID

    for scan_id in scan_ids:
        gt2pred = gt2pred_all[scan_id]

        gt_path = os.path.join(GT_SG_DIR, scan_id, f"{scan_id}_scan_sg.json")
        pred_path = os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json")

        if not os.path.exists(gt_path):
            print(f"[WARN] GT sg not found for scan {scan_id}: {gt_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"[WARN] Pred sg not found for scan {scan_id}: {pred_path}")
            continue

        gt_sg = load_json(gt_path)
        pred_sg = load_json(pred_path)

        want_details = bool(SINGLE_SCAN_ID) and (scan_id == SINGLE_SCAN_ID)

        (
            obj_total,
            obj_hit,
            obj_total_per_label,
            obj_hit_per_label,
            edge_total_all,
            edge_total_both,
            edge_hit,
            edge_total_per_rel,
            edge_hit_per_rel,
            edge_details,
            both_true_no_edge_cnt,
            edge_total_old,
            predicate_total_all,
            predicate_total_both,
            predicate_hit_directed,
            predicate_hit_undirected,
            hit_edges
        ) = evaluate_single_scan(
            scan_id,
            gt_sg,
            pred_sg,
            gt2pred,
            llm,
            proc,
            want_edge_details=want_details,
        )

        # update global statistics
        global_obj_total += obj_total
        global_obj_hit += obj_hit
        global_obj_total_per_label.update(obj_total_per_label)
        global_obj_hit_per_label.update(obj_hit_per_label)
        global_both_true_no_edge_cnt += both_true_no_edge_cnt
        global_edge_total_all += edge_total_all
        global_edge_total_both += edge_total_both
        global_edge_hit += edge_hit
        global_edge_total_per_rel.update(edge_total_per_rel)
        global_edge_hit_per_rel.update(edge_hit_per_rel)
        global_edge_total_old += edge_total_old
        global_pred_total_all += predicate_total_all
        global_pred_total_both += predicate_total_both
        global_pred_hit_dir += predicate_hit_directed
        global_pred_hit_undir += predicate_hit_undirected


        # per-scan statistics
        def safe_div(a, b):
            return float(a) / b if b > 0 else 0.0

        obj_recall_scan = safe_div(obj_hit, obj_total)
        edge_recall_scan = safe_div(edge_hit, edge_total_all)      # all GT edges as denominator
        both_true_recall_scan = safe_div(edge_hit, edge_total_both)
        old_triplet_recall_scan = safe_div(edge_hit, edge_total_old)


        print(
            f"[SCAN] {scan_id}: "
            f"obj {obj_hit}/{obj_total} (recall={obj_recall_scan:.4f}), "
            f"edge_hit={edge_hit}, "
            f"GT_edges_all={edge_total_all}, "
            f"GT_edges_both_true={edge_total_both}, "
            f"edge_recall={edge_recall_scan:.4f}, "
            f"both_true_edge_recall={both_true_recall_scan:.4f}"
            f", old_triplet_recall={old_triplet_recall_scan:.4f}"
        )

        # if needed, store the detailed result for this scan
        if want_details and edge_details is not None:
            single_scan_detail = {
                "scan_id": scan_id,
                "num_gt_obj": obj_total,
                "num_pred_obj": len(pred_sg.get("nodes", []) or []),
                "obj_recall": obj_recall_scan,
                "num_gt_edge_all": edge_total_all,
                "num_gt_edge_both_true": edge_total_both,
                "num_matched_edge": edge_hit,
                "edge_recall": edge_recall_scan,
                "both_true_edge_recall": both_true_recall_scan,
                "edge_per_rel_total": dict(edge_total_per_rel),
                "edge_per_rel_hit": dict(edge_hit_per_rel),
                "edge_match_details": edge_details,
            }

        # (NEW) save hit edges json per scan
        scan_out_dir = os.path.join(PRED_SG_DIR, scan_id)  # the "scan folder"
        os.makedirs(scan_out_dir, exist_ok=True)
        hit_save_path = os.path.join(scan_out_dir, f"{scan_id}_hit_edges.json")

        hit_out = {
            "scan_id": scan_id,
            "pred_sg_dir": PRED_SG_DIR,
            "num_hit_edges": len(hit_edges),
            "hit_edges": hit_edges,
        }
        with open(hit_save_path, "w", encoding="utf-8") as f:
            json.dump(hit_out, f, indent=2, ensure_ascii=False)

        print(f"[HIT] saved {len(hit_edges)} hit edges -> {hit_save_path}")

    # ========== Global Summary ==========

    def safe_div(a, b):
        return float(a) / b if b > 0 else 0.0

    obj_recall = safe_div(global_obj_hit, global_obj_total)
    obj_recalls = []
    for lab, tot in global_obj_total_per_label.items():
        hit = global_obj_hit_per_label.get(lab, 0)
        if tot > 0:
            obj_recalls.append(safe_div(hit, tot))
    obj_mrecall = sum(obj_recalls) / len(obj_recalls) if obj_recalls else 0.0

    # edge: two denominators
    edge_recall = safe_div(global_edge_hit, global_edge_total_all)
    both_true_edge_recall = safe_div(global_edge_hit, global_edge_total_both)
    both_true_no_edge_pct_scan = safe_div(global_both_true_no_edge_cnt, global_edge_total_both)


    edge_recalls = []
    for rel, tot in global_edge_total_per_rel.items():
        hit = global_edge_hit_per_rel.get(rel, 0)
        if tot > 0:
            edge_recalls.append(safe_div(hit, tot))
    edge_mrecall = sum(edge_recalls) / len(edge_recalls) if edge_recalls else 0.0

    old_triplet_recall = safe_div(global_edge_hit, global_edge_total_old)

    print("\n========== FINAL STATS ==========")
    print(f"Total GT objects       : {global_obj_total}")
    print(f"Matched objects        : {global_obj_hit}")
    print(f"Object recall          : {obj_recall:.4f}")
    print(f"Object mrecall         : {obj_mrecall:.4f}")
    print("")
    print(f"Total GT edges (all)   : {global_edge_total_all}")
    print(f"GT edges (both_true)   : {global_edge_total_both}")
    print(f"Matched edges          : {global_edge_hit}")
    print(f"Edge recall (all)      : {edge_recall:.4f}")
    print(f"Both-true edge recall  : {both_true_edge_recall:.4f}")
    print(f"Edge mrecall (per rel) : {edge_mrecall:.4f}")
    print(f"Both-true no-edge pct  : {both_true_no_edge_pct_scan:.4f}")
    print(f"Old triplet recall     : {old_triplet_recall:.4f}")
    print(f"Predicate recall (all, directed)   : {safe_div(global_pred_hit_dir, global_pred_total_all):.4f}")
    print(f"Predicate recall (all, undirected) : {safe_div(global_pred_hit_undir, global_pred_total_all):.4f}")
    print(f"Predicate recall (both, directed)  : {safe_div(global_pred_hit_dir, global_pred_total_both):.4f}")
    print(f"Predicate recall (both, undirected): {safe_div(global_pred_hit_undir, global_pred_total_both):.4f}")

    # compute total predicted nodes and edges
    total_predicted_nodes = sum(
        len(load_json(os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json")).get("nodes", []))
        for scan_id in scan_ids if os.path.exists(os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json"))
    )

    total_predicted_edges = sum(
        len(load_json(os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json")).get("edges", []))
        for scan_id in scan_ids if os.path.exists(os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json"))
    )

    # add predicted nodes and edges to global statistics
    out_detail = {
        "obj": {
            "total": global_obj_total,
            "hit": global_obj_hit,
            "recall": obj_recall,
            "mrecall": obj_mrecall,
            "per_label_total": dict(global_obj_total_per_label),
            "per_label_hit": dict(global_obj_hit_per_label),
        },
        "edge": {
            "total_all": global_edge_total_all,
            "total_both_true": global_edge_total_both,
            "hit": global_edge_hit,
            "recall_all": edge_recall,
            "recall_both_true": both_true_edge_recall,
            "mrecall": edge_mrecall,
            "both_true_no_edge_cnt": global_both_true_no_edge_cnt,
            "both_true_no_edge_pct": safe_div(global_both_true_no_edge_cnt, global_edge_total_both),
            "per_rel_total": dict(global_edge_total_per_rel),
            "per_rel_hit": dict(global_edge_hit_per_rel),
            "total_old": global_edge_total_old,
            "old_triplet_recall": old_triplet_recall,
        },
        "predicted": {
            "total_nodes": total_predicted_nodes,
            "total_edges": total_predicted_edges,
        },
    }

    with open(GLOBAL_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_detail, f, indent=2)
    print(f"\n[INFO] Detailed stats saved to {GLOBAL_OUT_JSON}")

    # detailed json for a single scan
    if SINGLE_SCAN_ID:
        if single_scan_detail is None:
            print(
                f"[WARN] SINGLE_SCAN_ID={SINGLE_SCAN_ID} was not evaluated "
                f"(possibly GT/Pred SG missing or scan not found in gt2pred)"
            )
        else:

            with open(os.path.join(SINGLE_SCAN_RECORD_SAVE, f"edge_eval_{SINGLE_SCAN_ID}.json"), "w", encoding="utf-8") as f:
                json.dump(single_scan_detail, f, indent=2)
            print(f"[INFO] Single-scan detail saved to {os.path.join(SINGLE_SCAN_RECORD_SAVE, f'edge_eval_{SINGLE_SCAN_ID}.json')}")


if __name__ == "__main__":
    main()
