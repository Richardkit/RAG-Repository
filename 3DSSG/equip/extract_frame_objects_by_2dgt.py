# -*- coding: utf-8 -*-
"""
Read all <scan_id>.2dgt (space-separated text) files from the specified directory,
generate the list of object_ids appearing in each frame, and output JSON
(consistent with the original script format):

{
  "<scan_id>": {
    "<frame_id(str)>": [object_id1, object_id2, ...],
    ...
  },
  ...
}
"""

import os
import glob
import json

# ======= Configuration: you only need to change the path here =======
DGT_DIR  = "/hkfs/work/workspace/scratch/rg5174-myspace/data/data_extra/2dgt"
OUT_JSON = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/frame_objects.json"
# ===================================


def main():
    if not os.path.isdir(DGT_DIR):
        raise FileNotFoundError(f"Directory not found: {DGT_DIR}")

    out = {}  # {scan_id: {frame_id(str): [int,...]}}

    files = sorted(glob.glob(os.path.join(DGT_DIR, "*.2dgt")))
    if not files:
        raise FileNotFoundError(f"No .2dgt files found in: {DGT_DIR}")

    for fp in files:
        scan_id = os.path.splitext(os.path.basename(fp))[0]

        with open(fp, "r", encoding="utf-8") as f:
            header_line = f.readline()
            if not header_line:
                # empty file, skip
                continue
            header = header_line.strip().split()
            idx = {k: i for i, k in enumerate(header)}

            # Only depend on these two columns; other columns (label/occlution_ratio/bbox) are not used
            if "frame_id" not in idx or "object_id" not in idx:
                raise ValueError(f"{fp}: header must contain 'frame_id' and 'object_id'")

            # Use set for deduplication
            per_frame_sets = {}  # {frame_id(str): set(object_id)}

            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                try:
                    fid = int(parts[idx["frame_id"]])
                    oid = int(parts[idx["object_id"]])
                except Exception:
                    # malformed line, skip
                    continue

                fid_str = str(fid)
                s = per_frame_sets.get(fid_str)
                if s is None:
                    s = set()
                    per_frame_sets[fid_str] = s
                s.add(oid)

        # Convert to sorted list
        per_frame_sorted = {k: sorted(list(v)) for k, v in per_frame_sets.items()}

        out[scan_id] = per_frame_sorted

    # Write out JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    print(f"[OK] saved frame-level objects to: {OUT_JSON}")
    print(f"  scans: {len(out)}")
    # Optional: print the number of frames per scan
    for sid, frames in out.items():
        print(f"  - {sid}: {len(frames)} frames")


if __name__ == "__main__":
    main()



