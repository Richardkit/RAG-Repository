#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, h5py, ast
from collections import defaultdict, Counter
from pathlib import Path
import tqdm

# ==== 配置 ====
RELATIONSHIPS_H5 = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/relationships.h5"
ROOT = "/hkfs/work/workspace/scratch/rg5174-myspace/data"   # 输出到 ROOT/3RScan/<scan_id>/<scan_id>_scan_sg.json
VISIBLE_ONLY = True  # 若你有 frame_objects_by_proposals.json，可只保留“在帧里出现过”的实例
FRAMES_JSON = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/frame_objects_by_proposals.json"
EXCLUDE_REL = {"same part"}  # 可扩展
# ============

def _relname(x: str) -> str:
    return (x or "").strip().lower()

def load_relationships(h5_path):
    scans = {}
    with h5py.File(h5_path, "r") as fp:
        for scan_id in fp.keys():
            s = fp[scan_id][0].decode()
            data = ast.literal_eval(s)

            nodes = {}
            for k, v in data.get("nodes", {}).items():
                oid = int(k)
                nodes[oid] = {"label": v.get("label"), "instance_id": v.get("instance_id")}

            rels_node = []
            for r in data.get("relationships", []):
                rels_node.append({"src": int(r[0]), "tgt": int(r[1]), "relationship": str(r[3])})

            # 提升到实例级（去重/去自环/过滤）
            oid2inst = {oid: (meta.get("instance_id") if meta.get("instance_id") is not None else oid)
                        for oid, meta in nodes.items()}
            seen = set()
            rels_inst = []
            for e in rels_node:
                rel = _relname(e["relationship"])
                if rel in EXCLUDE_REL: 
                    continue
                si = oid2inst.get(e["src"], e["src"])
                ti = oid2inst.get(e["tgt"], e["tgt"])
                if si == ti:
                    continue
                key = (si, ti, rel)
                if key in seen:
                    continue
                seen.add(key)
                rels_inst.append({"src": si, "tgt": ti, "relationship": rel})

            scans[scan_id] = {"nodes": nodes, "rels_inst": rels_inst}
    return scans

def majority_label_for_instance(inst, inst2oids, nodes_dict):
    labs = [nodes_dict[oid].get("label") for oid in inst2oids.get(inst, []) if nodes_dict.get(oid, {}).get("label")]
    return Counter(labs).most_common(1)[0][0] if labs else ""

def main():
    root = Path(ROOT).resolve()
    frames = {}
    if VISIBLE_ONLY and Path(FRAMES_JSON).exists():
        frames = json.loads(Path(FRAMES_JSON).read_text(encoding="utf-8"))

    scans = load_relationships(RELATIONSHIPS_H5)

    for scan_id, payload in tqdm.tqdm(scans.items(), desc="scan"):
        nodes_meta = payload["nodes"]
        rels_inst  = payload["rels_inst"]

        # OID <-> Instance
        oid2inst = {oid: (meta.get("instance_id") if meta.get("instance_id") is not None else oid)
                    for oid, meta in nodes_meta.items()}
        inst2oids = defaultdict(list)
        for oid, inst in oid2inst.items(): inst2oids[inst].append(oid)

        # 实例集合
        if VISIBLE_ONLY and scan_id in frames:
            insts_visible = set()
            for _, oid_list in frames[scan_id].items():
                for oid in oid_list:
                    if oid in oid2inst:
                        insts_visible.add(oid2inst[oid])
            insts = sorted(insts_visible)
        else:
            insts = sorted(inst2oids.keys())

        inst_set = set(insts)

        # ---- nodes: id + label（无几何、无 description）----
        nodes_out = []
        for inst in insts:
            label = majority_label_for_instance(inst, inst2oids, nodes_meta)
            nodes_out.append({"id": str(int(inst)), "label": str(label)})

        # ---- edges: src/tgt + relationship（小写）----
        edges_out = []
        for e in rels_inst:
            if e["src"] in inst_set and e["tgt"] in inst_set:
                edges_out.append({"src": str(int(e["src"])),
                                  "tgt": str(int(e["tgt"])),
                                  "relationship": e["relationship"]})

        # 保存到 ROOT/3RScan/<scan_id>/<scan_id>_scan_sg.json
        out_dir = root / "3RScan" / scan_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{scan_id}_scan_sg.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes_out, "edges": edges_out}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
