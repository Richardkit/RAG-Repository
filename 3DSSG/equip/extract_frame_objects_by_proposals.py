"""
Use proposal.h5 to form objects in frame-level, note that the values in output dictionary 
are oid, not instance_id, in the output of the code, it contains which oids are visible in each frame.
"""

import h5py
import json
import argparse


def load_seg_ids_from_attr(seg2idx_attr):
    """
    seg2idx is usually a list/ndarray of (segment_id, idx) pairs.
    This function extracts the segment_id (object id) from each pair.
    """
    seg_ids = []
    for pair in seg2idx_attr:
        # The first element can be int, bytes, or str
        k = pair[0]
        if isinstance(k, bytes):
            k = k.decode('utf-8', errors='ignore')
        try:
            k = int(k)
        except Exception:
            # In some cases, the object id might be stored as a string
            pass
        seg_ids.append(k)
    return seg_ids

def extract_frame_objects(pth_h5):
    """
    Extract mapping from frame to the list of objects it contains.

    Returns:
    {
      "<scan_id>": {
        "<frame_id(str)>": [object_id1, object_id2, ...],
        ...
      },
      ...
    }
    """
    out = {}
    with h5py.File(pth_h5, 'r') as f:
        # Iterate through each scan_id group
        for scan_id in f.keys():
            g = f[scan_id]
            if not isinstance(g, h5py.Group):
                continue
            if 'kfs' not in g:
                continue
            kfs = g['kfs']
            per_frame = {}

            # Each frame is a dataset with an attribute 'seg2idx' listing objects in that frame
            for frame_id in kfs.keys():
                dset = kfs[frame_id]
                if 'seg2idx' not in dset.attrs:
                    # No seg2idx attribute found → empty list
                    per_frame[str(frame_id)] = []
                    continue
                seg2idx_attr = dset.attrs['seg2idx']
                seg_ids = load_seg_ids_from_attr(seg2idx_attr)
                per_frame[str(frame_id)] = seg_ids

            out[str(scan_id)] = per_frame
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/proposals.h5", help="Path to proposal.h5")
    parser.add_argument("--out", default="/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/frame_objects_by_proposals.json", help="Output JSON file path")
    args = parser.parse_args()

    data = extract_frame_objects(args.h5)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print(f"Saved to {args.out}")