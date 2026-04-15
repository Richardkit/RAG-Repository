"""
The code is used to extract information from relationship.h5 matching frame_objects.json.
Since relationship do not frame information, we need divide the relationships of a scan 
into frame-level based on the output of extract_frame_objects.py.
The output contains frames and objects that can be seen in the corresponding frames


one scan id sctructure in proposal.h5 is like:

proposal.h5
└─ [scan_id]              
   ├─ nodes/              
   │   ├─ {oid} (dataset: 1D int array of frame_ids)
   │   │   
   │   │  # frames_id where the object is visible
   │   │   
   │   └─ ...
   └─ kfs/                # keyframe
       ├─ '{fid}' (dataset: Nx5 float array)
       │    attrs:
       │      'seg2idx' = [(oid, row_idx), ...]
       │    # every raw is a detect box like [xmin, ymin, xmax, ymax, occ]
       │    # every different raw represents a different object
       └─ ...

Important !!! 
nodes in relationship.h5 do not definitly represent an single object, it might be a part of an object
but edges with "same part" is that connects these nodes of same objects, it is necessary to combine 
those nodes together as one single object.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instance-level (object-level) scene graphs per frame, without geometry.
Writes:
  <ROOT>/3RScan/<scan_id>/frame_sg/frame-XXXXXX.json

Per-frame output:
{
  "scan": "<scan_id>",
  "frame_id": <int>,
  "objects": [
    {
      "id": <instance_id:int>,
      "name": "<object name>",
      "edges": [
        {"src": <instance_id>, "tgt": <instance_id>, "pred_id": int, "predicate": str}, ...
      ]
    }, ...
  ]
}
"""

import json
import h5py
import ast
from collections import defaultdict, Counter
from pathlib import Path
import tqdm

# ========= CONFIG: Fill in your paths =========
FRAMES_JSON = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/frame_objects_by_proposals.json"
RELATIONSHIPS_H5 = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/relationships.h5"
ROOT = "/hkfs/work/workspace/scratch/rg5174-myspace/data"
WITHIN_ONLY = False  # True: keep only edges whose both endpoints are visible in the current frame; False: keep all global instance-level edges
# ==============================================

# Relationships to exclude completely (case-insensitive)
EXCLUDE_RELATIONSHIPS = {"same part"}


"""
One thing need to be realized is that the oid in frame_objects.json might not be the same as in relationships.h5,
because in the relationships.h5, some oids are filtered. 
"""
DROP_ORPHANS = True  # True to drop OIDs not present in relationships.h5


def _relname(x: str) -> str:
    return (x or "").strip().lower()

def load_relationships_h5(h5_path):
    """
    Load the relationships.h5 file and return a dict:
      scans[scan_id] = {
        "nodes": { oid(int): {"label": str|None, "instance_id": int|None}, ... },
        "rels_node_level": [ {src:oid, tgt:oid, pred_id:int, relationship:str}, ... ],
        "rels_inst_level": [ {src:inst, tgt:inst, pred_id:int, relationship:str}, ... ]  
          # Directed dedup, no self-loops, excludes relationships in EXCLUDE_RELATIONSHIPS
      }
    """
    scans = {}
    with h5py.File(h5_path, "r") as fp:
        for scan_id in fp.keys():
            s = fp[scan_id][0].decode()
            data = ast.literal_eval(s)

            # ---- Nodes: only keep label and instance_id ----
            nodes = {}
            for k, v in data.get("nodes", {}).items():
                oid = int(k)
                nodes[oid] = {
                    "label": v.get("label"),
                    "instance_id": v.get("instance_id"),
                }

            # ---- Node-level relationships ----
            rels_node = []
            for r in data.get("relationships", []):
                rels_node.append({
                    "src": int(r[0]),
                    "tgt": int(r[1]),
                    "pred_id": int(r[2]),
                    "relationship": str(r[3]),
                })

            # ---- Lift to instance-level (directed dedup) ----
            oid2inst = {oid: ([meta.get("instance_id"), meta.get("label")] if meta.get("instance_id") is not None else oid)
                        for oid, meta in nodes.items()} # this step seems to be useless now, since the oid == instance_id, oid is just manually added order

            seen = set()
            rels_inst = []
            for e in rels_node:
                # Skip relationships that should be excluded (e.g., "same part")
                if _relname(e["relationship"]) in EXCLUDE_RELATIONSHIPS:
                    continue

                si_with_label = oid2inst.get(e["src"], [e["src"],""])  # Instance ID of source; fallback to oid if None
                ti_with_label = oid2inst.get(e["tgt"], [e["tgt"],""])  # Instance ID of target; fallback to oid if None
                if si_with_label[0] == ti_with_label[0]:
                    # Skip self-loops (same instance)
                    continue

                # Directed dedup: keep direction to preserve subject/object order
                key = (si_with_label[0], ti_with_label[0], e["pred_id"], e["relationship"])
                if key in seen:
                    continue
                seen.add(key)

                rels_inst.append({
                    "src": si_with_label[0],
                    "tgt": ti_with_label[0],
                    "pred_id": e["pred_id"],
                    "relationship": e["relationship"],
                    "src_label": si_with_label[1],
                    "tgt_label": ti_with_label[1]
                })

            scans[scan_id] = {
                "nodes": nodes,
                "rels_node_level": rels_node,
                "rels_inst_level": rels_inst,
            }
    return scans

def main():
    root = Path(ROOT).expanduser().resolve()

    # frames_json: scan_id -> frame_id(str) -> [[oid, label], ...]
    with open(FRAMES_JSON, "r", encoding="utf-8") as f:
        frames = json.load(f)

    scans = load_relationships_h5(RELATIONSHIPS_H5)

    for scan_id, frames_dict in tqdm.tqdm(frames.items()):
        if scan_id not in scans:
            continue

        nodes = scans[scan_id]["nodes"]
        rels_inst_all = scans[scan_id]["rels_inst_level"]

        # Maps from oid to instance and from instance to all oids
        oid2inst = {oid: (meta.get("instance_id") if meta.get("instance_id") is not None else oid)
                    for oid, meta in nodes.items()} # this step seems to be useless now, since the oid == instance_id, oid is just manually added order
        inst2oids = defaultdict(list)
        for oid, inst in oid2inst.items():
            inst2oids[inst].append(oid)

        # Pre-index all instance-level edges by instance for fast lookup
        edges_by_inst_full = defaultdict(list)
        for e in rels_inst_all:
            edges_by_inst_full[e["src"]].append(e)
            edges_by_inst_full[e["tgt"]].append(e)

        # Output folder: <root>/3RScan/<scan_id>/frame_sg
        out_dir = root / "3RScan" / scan_id / "frame_sg"
        
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_id_str, obj_list in frames_dict.items():
            frame_id = int(frame_id_str)

            # Collect visible instances in this frame (an instance is visible if any of its parts is visible)
            labels_from_frame = defaultdict(list) # inst -> labels seen in this frame
            insts_in_frame = set()
            orphans = [] # oids missing from relationship.h5

            for item in obj_list:
                oid = item

                if oid in oid2inst:
                    inst = oid2inst[oid] # object in the frame to its instance
                    insts_in_frame.add(inst) # a set only contains unique instances

                else:
                    orphans.append(oid)
                    if not DROP_ORPHANS:
                        insts_in_frame.add(oid)  # Keep orphans as their own "instance"

            insts_in_frame = sorted(insts_in_frame)

            # Compute within-frame edges (directed dedup, exclude self-loops)
            # edges_within stores visible edges in frames
            # edges_by_inst_within stores src and tgt of the visible frames.
            inst_present = set(insts_in_frame)
            seen_w = set()
            edges_within = []
            for e in rels_inst_all:
                if e["src"] in inst_present and e["tgt"] in inst_present:
                    key = (e["src"], e["tgt"], e["pred_id"], e["relationship"])
                    if key not in seen_w:
                        seen_w.add(key)
                        edges_within.append(e)

            edges_by_inst_within = defaultdict(list)
            for e in edges_within:
                edges_by_inst_within[e["src"]].append(e)
                edges_by_inst_within[e["tgt"]].append(e)

            # Build the objects list (instance-level, without geometry)
            objects_out = []
            for inst in insts_in_frame:
                # Name: majority label in this frame; fallback to any non-empty node label in the instance
                name = None

                for oid in inst2oids.get(inst, []):
                    lab = nodes.get(oid, {}).get("label")
                    if lab:
                        name = lab
                        break

                if name is None:
                    name = ""

                # Choose edges according to WITHIN_ONLY flag
                if WITHIN_ONLY:
                    edges_for_inst = edges_by_inst_within.get(inst, [])
                else:
                    edges_for_inst = edges_by_inst_full.get(inst, [])

                objects_out.append({
                    "id": int(inst),
                    "name": str(name),
                    "edges": edges_for_inst,
                })

            # Assemble and write scene graph for this frame
            sg = {
                "scan": scan_id,
                "frame_id": frame_id,
                "objects": objects_out
            }

            out_path = out_dir / f"frame-{frame_id:06d}.json" # frame ID zero padded to 6 digits

            with open(out_path, "w", encoding = "utf-8") as fp:
                fp.write(json.dumps(sg, indent=2))

if __name__ == "__main__":
    main()
