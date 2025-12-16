#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluate all scans under PRED_ROOT.
- Only Recall / Macro-Recall (no precision).
- Per-scan details: per-predicate tp/support (e.g., attached_to: 1/8).
- Append overall recall and macro-recall at the end.
- Fast: one-time batch SBERT encoding + cached cosine, with candidate pruning.
- NEW: head-only name matching -> keep only main class tokens from classes.txt (drop adjectives/modifiers).

Requires: pip install sentence-transformers
"""

import json, re, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

# ===================== CONFIG (edit if needed) =====================
PRED_ROOT = Path("/hkfs/work/workspace/scratch/rg5174-myspace/data/RAG_end2end_slide_new")
GT_ROOT   = Path("/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan")
SUMMARY_TXT = PRED_ROOT / "_eval_recall_summary.txt"

# NEW: classes lexicon (keep only head tokens from this list)
CLASSES_TXT = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/classes.txt"

# Similarity thresholds
OBJ_THR = 0.35
REL_THR = 0.50

# SBERT model config
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DEVICE     = "cpu"   # use "cpu" to avoid GPU init overhead; switch to "cuda" if stable GPU is available
BATCH_SIZE     = 512     # encode batch size
# ==================================================================


# ----------------- Utilities / Normalization -----------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()

def load_json(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_vocab_list(path: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(s)
    return items

# Build class set once (normalized)
CLASSES_NORM: Set[str] = set()
def _init_classes():
    global CLASSES_NORM
    if not CLASSES_NORM:
        cls = load_vocab_list(CLASSES_TXT)
        CLASSES_NORM = {_norm(x) for x in cls}

def extract_head_tokens(text: str, classes_norm: Set[str]) -> List[str]:
    """
    Keep only tokens that are in classes list (normalized).
    This effectively drops adjectives/modifiers and preserves main class words.
    """
    s = _norm(text)
    if not s:
        return []
    toks = re.findall(r"[a-z0-9]+", s)
    keep = [t for t in toks if t in classes_norm]
    return keep

def to_head_only(text: str) -> str:
    """
    Convert any name/label/desc to 'head-only' string by keeping only class tokens.
    Fallback to original normalized text if no class token is found (to avoid empty names).
    """
    _init_classes()
    kept = extract_head_tokens(text, CLASSES_NORM)
    if kept:
        return " ".join(sorted(set(kept)))
    return _norm(text)

def edge_pred_text(e: Dict) -> str:
    return _norm(e.get("c_edge") or e.get("relationship") or e.get("predicate") or "")

def node_text_raw(n: Dict) -> str:
    # GT nodes often {"id","label"} ; Pred nodes often {"id","name","description"}
    return _norm(n.get("name") or n.get("label") or n.get("description") or "")

def node_text_head(n: Dict) -> str:
    # head-only name for embedding/matching
    raw = _norm(n.get("name") or n.get("label") or n.get("description") or "")
    return to_head_only(raw)

def build_id2text_head(graph: Dict) -> Dict[str, str]:
    return { str(n["id"]): node_text_head(n) for n in (graph.get("nodes") or []) }

def collect_edges(graph: Dict) -> List[Tuple[str, str, str]]:
    out = []
    for e in (graph.get("edges") or []):
        s = str(e.get("src","")).strip()
        t = str(e.get("tgt","")).strip()
        p = edge_pred_text(e)
        if s and t and p and s != t:
            out.append((s, p, t))
    return out

def find_pred_any(scan_dir: Path, scan_id: str) -> Optional[Path]:
    """Choose best available prediction file: final -> global_win -> local_win."""
    final = scan_dir / f"{scan_id}_global_graph_final.json"
    if final.exists():
        return final
    globs = sorted(scan_dir.glob(f"{scan_id}_global_sg_win*.json"))
    if globs:
        def _win_idx(p: Path) -> int:
            m = re.search(r"_win(\d{3})\.json$", p.name)
            return int(m.group(1)) if m else -1
        globs.sort(key=_win_idx)
        return globs[-1]
    locals_ = sorted(scan_dir.glob(f"{scan_id}_local_sg_win*.json"))
    if locals_:
        def _win_idx2(p: Path) -> int:
            m = re.search(r"_win(\d{3})\.json$", p.name)
            return int(m.group(1)) if m else -1
        locals_.sort(key=_win_idx2)
        return locals_[-1]
    return None


# ----------------- Global Embed Cache -----------------
MODEL = None
EMB_CACHE: Dict[str, np.ndarray] = {}

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(EMB_MODEL_NAME, device=EMB_DEVICE)
    return MODEL

def encode_texts_batch(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Batch encode (L2-normalized)."""
    model = get_model()
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True).astype(np.float32)
    return vecs

def cache_embeddings(strings: List[str]):
    """Encode and cache strings that are not yet cached."""
    todo = [s for s in strings if s not in EMB_CACHE]
    if not todo:
        return
    vecs = encode_texts_batch(todo)
    for s, v in zip(todo, vecs):
        EMB_CACHE[s] = v

def emb(s: str) -> Optional[np.ndarray]:
    return EMB_CACHE.get(s, None)

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are L2-normalized; cosine = dot product
    return float(np.dot(a, b))


# ----------------- Fast Evaluator (semantic consume) -----------------
def eval_semantic_consume_fast(
    gt_graph: Dict, pred_graph: Dict,
    thr_obj: float = OBJ_THR, thr_rel: float = REL_THR
) -> Dict:
    """
    Semantic triple matching with candidate pruning and cached embeddings.
    Names are converted to 'head-only' strings using classes.txt (drop modifiers).
    Only computes Recall and Macro-Recall (per GT predicate).
    """
    # use head-only names for both GT and Pred
    gt_id2txt = build_id2text_head(gt_graph)
    pr_id2txt = build_id2text_head(pred_graph)

    gt_edges   = collect_edges(gt_graph)
    pred_edges = collect_edges(pred_graph)

    # Quick exits
    if len(gt_edges) == 0:
        return {
            "recall": 0.0, "macro_recall": 0.0, "tp": 0,
            "counts": {"edges_gt": 0, "edges_pred": len(pred_edges)},
            "per_predicate": {}
        }
    if len(pred_edges) == 0:
        # recall = 0; per-predicate recalls all 0
        supp = Counter([gp for (_, gp, _) in gt_edges])
        return {
            "recall": 0.0, "macro_recall": 0.0, "tp": 0,
            "counts": {"edges_gt": len(gt_edges), "edges_pred": 0},
            "per_predicate": {p: {"tp": 0, "support": c} for p, c in supp.items()}
        }

    # Group node ids by (head-only) name for both GT and Pred
    gt_name_to_ids = defaultdict(list)
    pr_name_to_ids = defaultdict(list)
    for nid, name in gt_id2txt.items():
        if name: gt_name_to_ids[name].append(nid)
    for nid, name in pr_id2txt.items():
        if name: pr_name_to_ids[name].append(nid)

    # Collect unique names/predicates within this scan (to keep candidate building small)
    gt_name_set = set(gt_name_to_ids.keys())
    pr_name_set = set(pr_name_to_ids.keys())
    gt_pred_set = {p for (_, p, _) in gt_edges}
    pr_pred_set = {p for (_, p, _) in pred_edges}

    # Candidate maps
    name_candidates: Dict[str, set] = defaultdict(set)  # gt_name(head-only) -> {pred_name(head-only)}
    pred_candidates: Dict[str, set] = defaultdict(set)  # gt_pred -> {pred_pred}

    # Build name candidates (only compare names that appear in this scan)
    for gname in gt_name_set:
        gv = emb(gname)
        if gv is None:
            continue
        for pname in pr_name_set:
            pv = emb(pname)
            if pv is None:
                continue
            if cos_sim(gv, pv) >= thr_obj:
                name_candidates[gname].add(pname)

    # Build predicate candidates
    for gp in gt_pred_set:
        gv = emb(gp)
        if gv is None:
            continue
        for pp in pr_pred_set:
            pv = emb(pp)
            if pv is None:
                continue
            if cos_sim(gv, pv) >= thr_rel:
                pred_candidates[gp].add(pp)

    used = [False] * len(pred_edges)

    # Per-predicate recall stats
    support_by_gt_pred = Counter()
    tp_by_gt_pred      = Counter()

    for (_, gp, _) in gt_edges:
        support_by_gt_pred[gp] += 1

    tp = 0
    for (gs, gp, gt_) in gt_edges:
        gs_txt = gt_id2txt.get(gs, "")
        gt_txt = gt_id2txt.get(gt_, "")

        cand_src_names = name_candidates.get(gs_txt, set())
        cand_tgt_names = name_candidates.get(gt_txt, set())
        cand_preds     = pred_candidates.get(gp, set())
        if not cand_src_names or not cand_tgt_names or not cand_preds:
            continue  # no possible match

        cand_src_ids = set().union(*(pr_name_to_ids[nm] for nm in cand_src_names)) if cand_src_names else set()
        cand_tgt_ids = set().union(*(pr_name_to_ids[nm] for nm in cand_tgt_names)) if cand_tgt_names else set()

        best_j, best_score = -1, -1.0
        for j, (ps, pp, pt) in enumerate(pred_edges):
            if used[j]:
                continue
            if (ps not in cand_src_ids) or (pt not in cand_tgt_ids) or (pp not in cand_preds):
                continue

            # final check with actual cos-sim (vectors are cached)
            ps_name = pr_id2txt.get(ps, "")
            pt_name = pr_id2txt.get(pt, "")
            gv_s = emb(gs_txt); pv_s = emb(ps_name)
            gv_p = emb(gp);     pv_p = emb(pp)
            gv_t = emb(gt_txt); pv_t = emb(pt_name)
            if gv_s is None or pv_s is None or gv_p is None or pv_p is None or gv_t is None or pv_t is None:
                continue

            sim_s = cos_sim(pv_s, gv_s)
            sim_p = cos_sim(pv_p, gv_p)
            sim_t = cos_sim(pv_t, gv_t)
            if (sim_s >= thr_obj) and (sim_p >= thr_rel) and (sim_t >= thr_obj):
                score = min(sim_s, sim_p, sim_t)
                if score > best_score:
                    best_score, best_j = score, j

        if best_j >= 0:
            used[best_j] = True
            tp += 1
            tp_by_gt_pred[gp] += 1

    gt_total = len(gt_edges)
    recall   = tp / gt_total if gt_total else 0.0

    per_pred_recalls = []
    per_pred_counts  = {}
    for p, supp in support_by_gt_pred.items():
        tp_p = tp_by_gt_pred.get(p, 0)
        r_p  = tp_p / supp if supp else 0.0
        per_pred_recalls.append(r_p)
        per_pred_counts[p] = {"tp": tp_p, "support": supp}
    mrecall = (sum(per_pred_recalls)/len(per_pred_recalls)) if per_pred_recalls else 0.0

    return {
        "recall": recall,
        "macro_recall": mrecall,
        "tp": tp,
        "counts": {"edges_gt": gt_total, "edges_pred": len(pred_edges)},
        "per_predicate": per_pred_counts
    }


# ----------------- Main: batch over PRED_ROOT -----------------
def main():
    _init_classes()

    # 1) gather all strings across all scans (HEAD-ONLY names & predicates), then cache embeddings once
    scan_dirs = [d for d in PRED_ROOT.iterdir() if d.is_dir()]
    scan_dirs.sort()

    all_name_strings = set()
    all_pred_strings = set()

    for d in scan_dirs:
        scan_id = d.name
        pred_path = find_pred_any(d, scan_id)
        gt_path   = GT_ROOT / scan_id / f"{scan_id}_scan_sg.json"
        if (pred_path is None) or (not gt_path.exists()):
            continue
        try:
            pred_graph = load_json(pred_path)
            gt_graph   = load_json(gt_path)
        except Exception:
            continue

        # collect head-only names
        for n in (pred_graph.get("nodes") or []):
            s = node_text_head(n)
            if s: all_name_strings.add(s)
        for n in (gt_graph.get("nodes") or []):
            s = node_text_head(n)
            if s: all_name_strings.add(s)

        # predicates unchanged
        for e in (pred_graph.get("edges") or []):
            p = edge_pred_text(e)
            if p: all_pred_strings.add(p)
        for e in (gt_graph.get("edges") or []):
            p = edge_pred_text(e)
            if p: all_pred_strings.add(p)

    # Cache embeddings once (head-only names + predicates)
    cache_embeddings(list(all_name_strings))
    cache_embeddings(list(all_pred_strings))
    print(f"[Diag] Cached head-only names: {len(all_name_strings)} | predicate strings: {len(all_pred_strings)}")

    # 2) evaluate each scan and build summary
    lines: List[str] = []
    overall_tp = 0
    overall_gt = 0
    global_pred_tp      = Counter()  # predicate -> total tp
    global_pred_support = Counter()  # predicate -> total support
    included_scans = 0

    for d in scan_dirs:
        scan_id = d.name
        pred_path = find_pred_any(d, scan_id)
        gt_path   = GT_ROOT / scan_id / f"{scan_id}_scan_sg.json"
        if (pred_path is None) or (not gt_path.exists()):
            print(f"[Skip] {scan_id}: pred/gt missing.")
            continue

        try:
            pred_graph = load_json(pred_path)
            gt_graph   = load_json(gt_path)
        except Exception as e:
            print(f"[Skip] {scan_id}: failed to read json ({e}).")
            continue

        res = eval_semantic_consume_fast(
            gt_graph, pred_graph,
            thr_obj=OBJ_THR, thr_rel=REL_THR
        )

        included_scans += 1
        head = (f"[{scan_id}]  "
                f"recall={res['recall']:.4f}  mRecall={res['macro_recall']:.4f}  "
                f"(GT={res['counts']['edges_gt']}, Pred={res['counts']['edges_pred']})")
        lines.append(head)

        # Per-predicate tp/support
        per = res["per_predicate"]
        per_items = sorted(per.items(), key=lambda kv: (-kv[1]["support"], kv[0]))
        for p, st in per_items:
            tp_p, supp = st["tp"], st["support"]
            lines.append(f"  {p}: {tp_p}/{supp}")
            global_pred_tp[p]      += tp_p
            global_pred_support[p] += supp

        lines.append("")

        overall_tp += res["tp"]
        overall_gt += res["counts"]["edges_gt"]

    # Overall metrics across included scans
    if included_scans == 0 or overall_gt == 0:
        lines.append("=== OVERALL ===")
        lines.append("Overall recall = 0.0000")
        lines.append("Overall macro-recall = 0.0000")
    else:
        overall_recall = overall_tp / overall_gt
        macro_list = []
        for p, supp in global_pred_support.items():
            if supp > 0:
                tp_p = global_pred_tp.get(p, 0)
                macro_list.append(tp_p / supp)
        overall_mrecall = (sum(macro_list)/len(macro_list)) if macro_list else 0.0

        lines.append("=== OVERALL ===")
        lines.append(f"Overall recall = {overall_recall:.4f}")
        lines.append(f"Overall macro-recall = {overall_mrecall:.4f}")

    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Saved] summary -> {SUMMARY_TXT}")

if __name__ == "__main__":
    main()
