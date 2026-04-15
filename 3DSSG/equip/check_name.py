"""
Check if the nodes[oid]["label"] is empty or "" in relationship.h5
"""

import h5py, ast
h5_path = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data/3RScan_ScanNet20_InSeg/relationships.h5"

scans_with_empty = {}
with h5py.File(h5_path, "r") as f:
    for scan_id in f.keys():
        data = ast.literal_eval(f[scan_id][0].decode())
        empty = [oid for oid, meta in data.get("nodes", {}).items()
                 if not str(meta.get("label") or "").strip()]
        if empty:
            scans_with_empty[scan_id] = empty

print("scans with empty labels:", len(scans_with_empty))
