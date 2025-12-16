#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eva_final_new_v2.py

全局多 scan 的最终评估脚本：
- 使用 gt2pred 一对一匹配结果评估 obj / edge recall & mrecall
- 支持 predicted SG 中 c_edge 为多谓词 list 的逻辑
- 对 edge 同时给出：
    1) edge_recall:     以所有有效 GT edges 为分母
    2) both_true_recall:以 src 和 tgt 都匹配上的 GT edges 为分母（旧版含义）
- 可选：对 SINGLE_SCAN_ID 输出详细的 per-edge 命中情况 json
"""

import os
import json
import ast
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any

# ===================== 路径配置（请按实际情况修改） =====================

GT_SG_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan"
PRED_SG_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/RAG_baseline_Qwen3/end2end_slide"
GT2PRED_RESULT = "/hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_baseline_Qwen3/evaluate/norag_scan_result/norag_obj_match_results.jsonl"

# 如果想额外输出某个 scan 的详细 edge 命中情况，在这里填 scan_id；
# 不需要就设为 "" 或 None
SINGLE_SCAN_ID = "38770ca1-86d7-27b8-8619-ab66f67d9adf"  # 例如 "0cf75f4b-564d-23f5-899f-ff97701d3f3b"

GLOBAL_OUT_JSON = "/hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_baseline_Qwen3/evaluate/norag_scan_result/eva_final_norag_results.json"
SINGLE_SCAN_OUT_PREFIX = "edge_eval_"  # 单 scan 输出：edge_eval_<scan_id>.json

# =======================================================================


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(s: str) -> str:
    """归一化 object label / name"""
    if not s:
        return ""
    return str(s).strip().lower()


def normalize_predicate(p) -> str:
    """归一化 predicate，兼容 list / 非字符串输入"""
    if p is None:
        return ""
    if isinstance(p, list):
        # 防御：如果传进来就是 list，只取第一个
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
    解析 predicted edge 里的 c_edge / relationship：
      - 如果本身就是 list: ["standing on", "attached to"]
      - 如果是字符串形式的 list: "['standing on', 'attached to']"
      - 否则按单一 predicate 处理
    """
    if rel_value is None:
        return []

    # 已经是 list
    if isinstance(rel_value, list):
        return [str(x).strip() for x in rel_value if str(x).strip()]

    # 字符串：有可能是 "['a', 'b']"
    if isinstance(rel_value, str):
        s = rel_value.strip()
        if not s:
            return []
        # 尝试 literal_eval
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        # 再退而求其次按逗号分割
        parts = [p.strip().strip("[]'\"") for p in s.split(",")]
        return [p for p in parts if p]

    # 其他类型：直接转成单个字符串
    v = str(rel_value).strip()
    return [v] if v else []


# ===================== 读取 gt2pred 结果 =====================

def load_gt2pred_from_jsonl(path: str) -> Dict[str, Dict[str, str]]:
    """
    从 jsonl 文件读取 scan_id -> {gt_obj_id -> pred_node_id} 映射。
    兼容两种行格式：
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

            # 形式 1：带 gt2pred（scan-level）
            if "gt2pred" in rec:
                gt2pred = {str(k): str(v) for k, v in rec["gt2pred"].items()}
                scan_map.setdefault(scan_id, {}).update(gt2pred)
                continue

            # 形式 2：旧格式的一行一个 gt
            if "gt_obj_id" in rec and "match_pred_node_id" in rec:
                gid = str(rec["gt_obj_id"])
                pid = str(rec["match_pred_node_id"])
                scan_map.setdefault(scan_id, {})[gid] = pid

    return scan_map


def load_gt2pred(path: str) -> Dict[str, Dict[str, str]]:
    """
    统一入口：path 可以是 json 或 jsonl。
    - json: list[{"scan_id":..., "gt2pred": {...}}, ...]
    - jsonl: 每行一个 dict（见上面的说明）
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


# ===================== Pred 边 → 三元组 =====================

def build_pred_edge_triples(pred_sg: dict) -> Set[Tuple[str, str, str]]:
    """
    从 predicted scene graph 中构造所有 (src, predicate_norm, tgt) 三元组。
    支持一条 edge 对应多个谓词（来自 c_edge 或 relationship）。
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


# ===================== 评估单个 scan =====================

def evaluate_single_scan(
    scan_id: str,
    gt_sg: dict,
    pred_sg: dict,
    gt2pred: Dict[str, str],
    want_edge_details: bool = False,
):
    """
    返回：
      obj_total, obj_hit, obj_per_label_total, obj_per_label_hit,
      edge_total_all, edge_total_both, edge_hit,
      edge_per_rel_total, edge_per_rel_hit,
      edge_match_details (若 want_edge_details=False，则为 None)

    说明：
      - edge_total_all  ：所有有效 GT edges 数量（src/tgt/predicate 均存在）
      - edge_total_both ：src 和 tgt 都映射成功的 GT edges 数量（旧脚本的 edge_total）
      - edge_hit        ：三元组完全匹配的 GT edges 数量
    """

    # ---------- 对象 ----------
    gt_nodes = gt_sg.get("nodes", []) or []
    pred_nodes = pred_sg.get("nodes", []) or []

    obj_total = 0
    obj_hit = 0
    obj_per_label_total = Counter()
    obj_per_label_hit = Counter()

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

    # ---------- 关系 ----------
    gt_edges = gt_sg.get("edges", []) or []
    pred_edge_triples = build_pred_edge_triples(pred_sg)

    edge_total_all = 0         # 所有有效 GT edges
    edge_total_both = 0        # src/tgt 都映射上的 GT edges（旧的 edge_total）
    edge_hit = 0
    edge_per_rel_total = Counter()
    edge_per_rel_hit = Counter()

    edge_match_details: List[Dict[str, Any]] = []

    # 记录 GT node id → label，方便写详情
    gt_node_label_map = {
        str(n.get("id")): (n.get("label") or n.get("name") or "unknown")
        for n in gt_nodes
    }

    for e in gt_edges:
        s_gt = str(e.get("src"))
        t_gt = str(e.get("tgt"))
        rel_gt_raw = e.get("relationship") or e.get("predicate")
        rel_gt = normalize_predicate(rel_gt_raw)

        # 无效 edge（缺 src/tgt/predicate）直接跳过，不计入任何分母
        if not s_gt or not t_gt or not rel_gt:
            continue

        # 先计入“所有有效 GT edges”的分母
        edge_total_all += 1

        # GT → Pred 映射
        pred_src = gt2pred.get(s_gt, "none")
        pred_tgt = gt2pred.get(t_gt, "none")

        if isinstance(pred_src, (int, float)):
            pred_src = str(pred_src)
        if isinstance(pred_tgt, (int, float)):
            pred_tgt = str(pred_tgt)

        pred_src = (pred_src or "none").strip()
        pred_tgt = (pred_tgt or "none").strip()

        src_hit = bool(pred_src) and pred_src.lower() != "none"
        tgt_hit = bool(pred_tgt) and pred_tgt.lower() != "none"

        predicate_hit = False
        edge_hit_flag = False

        # 只有 src/tgt 都映射上的 edge 才计入“both_true”的分母和 per-rel 统计
        if src_hit and tgt_hit:
            edge_total_both += 1
            edge_per_rel_total[rel_gt] += 1

            if (pred_src, rel_gt, pred_tgt) in pred_edge_triples:
                predicate_hit = True
                edge_hit_flag = True
                edge_hit += 1
                edge_per_rel_hit[rel_gt] += 1

        if want_edge_details:
            edge_match_details.append(
                {
                    "scan_id": scan_id,
                    "src_gt_id": s_gt,
                    "tgt_gt_id": t_gt,
                    "gt_predicate": rel_gt_raw,
                    "gt_predicate_norm": rel_gt,
                    "src_gt_label": gt_node_label_map.get(s_gt, "unknown"),
                    "tgt_gt_label": gt_node_label_map.get(t_gt, "unknown"),
                    "src_pred_id": pred_src,
                    "tgt_pred_id": pred_tgt,
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
    )


# ===================== 主评估流程 =====================

def main():
    gt2pred_all = load_gt2pred(GT2PRED_RESULT)
    scan_ids = sorted(gt2pred_all.keys())

    print(f"[INFO] Loaded gt2pred for {len(scan_ids)} scans")

    # 全局统计
    global_obj_total = 0
    global_obj_hit = 0

    global_edge_total_all = 0      # 所有有效 GT edges
    global_edge_total_both = 0     # src/tgt 都映射上的 GT edges
    global_edge_hit = 0

    global_obj_total_per_label = Counter()
    global_obj_hit_per_label = Counter()
    global_edge_total_per_rel = Counter()
    global_edge_hit_per_rel = Counter()

    single_scan_detail: Any = None  # SINGLE_SCAN_ID 的详细结果

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
        ) = evaluate_single_scan(
            scan_id,
            gt_sg,
            pred_sg,
            gt2pred,
            want_edge_details=want_details,
        )

        # 更新全局统计
        global_obj_total += obj_total
        global_obj_hit += obj_hit
        global_obj_total_per_label.update(obj_total_per_label)
        global_obj_hit_per_label.update(obj_hit_per_label)

        global_edge_total_all += edge_total_all
        global_edge_total_both += edge_total_both
        global_edge_hit += edge_hit
        global_edge_total_per_rel.update(edge_total_per_rel)
        global_edge_hit_per_rel.update(edge_hit_per_rel)

        # per-scan 统计
        def safe_div(a, b):
            return float(a) / b if b > 0 else 0.0

        obj_recall_scan = safe_div(obj_hit, obj_total)
        edge_recall_scan = safe_div(edge_hit, edge_total_all)      # 所有 GT edge 为分母
        both_true_recall_scan = safe_div(edge_hit, edge_total_both)  # 旧逻辑分母

        print(
            f"[SCAN] {scan_id}: "
            f"obj {obj_hit}/{obj_total} (recall={obj_recall_scan:.4f}), "
            f"edge_hit={edge_hit}, "
            f"GT_edges_all={edge_total_all}, "
            f"GT_edges_both_true={edge_total_both}, "
            f"edge_recall={edge_recall_scan:.4f}, "
            f"both_true_edge_recall={both_true_recall_scan:.4f}"
        )

        # 若需要，记住这个 scan 的详细结果
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

    # ========== 全局 summary ==========

    def safe_div(a, b):
        return float(a) / b if b > 0 else 0.0

    obj_recall = safe_div(global_obj_hit, global_obj_total)
    obj_recalls = []
    for lab, tot in global_obj_total_per_label.items():
        hit = global_obj_hit_per_label.get(lab, 0)
        if tot > 0:
            obj_recalls.append(safe_div(hit, tot))
    obj_mrecall = sum(obj_recalls) / len(obj_recalls) if obj_recalls else 0.0

    # edge：两种分母
    edge_recall = safe_div(global_edge_hit, global_edge_total_all)
    both_true_edge_recall = safe_div(global_edge_hit, global_edge_total_both)

    edge_recalls = []
    for rel, tot in global_edge_total_per_rel.items():
        hit = global_edge_hit_per_rel.get(rel, 0)
        if tot > 0:
            edge_recalls.append(safe_div(hit, tot))
    edge_mrecall = sum(edge_recalls) / len(edge_recalls) if edge_recalls else 0.0

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

    # 计算总的预测节点和边
    total_predicted_nodes = sum(
        len(load_json(os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json")).get("nodes", []))
        for scan_id in scan_ids if os.path.exists(os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json"))
    )

    total_predicted_edges = sum(
        len(load_json(os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json")).get("edges", []))
        for scan_id in scan_ids if os.path.exists(os.path.join(PRED_SG_DIR, scan_id, f"{scan_id}_global_graph_final.json"))
    )

    # 添加预测节点和边到全局统计
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
            "per_rel_total": dict(global_edge_total_per_rel),
            "per_rel_hit": dict(global_edge_hit_per_rel),
        },
        "predicted": {
            "total_nodes": total_predicted_nodes,
            "total_edges": total_predicted_edges,
        },
    }

    with open(GLOBAL_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_detail, f, indent=2)
    print(f"\n[INFO] Detailed stats saved to {GLOBAL_OUT_JSON}")

    # 单个 scan 的详细 json
    if SINGLE_SCAN_ID:
        if single_scan_detail is None:
            print(
                f"[WARN] SINGLE_SCAN_ID={SINGLE_SCAN_ID} 没有被评估到 "
                f"（可能 GT/Pred SG 缺失或 gt2pred 中没有该 scan）"
            )
        else:
            out_path = SINGLE_SCAN_OUT_PREFIX + f"{SINGLE_SCAN_ID}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(single_scan_detail, f, indent=2)
            print(f"[INFO] Single-scan detail saved to {out_path}")


if __name__ == "__main__":
    main()
