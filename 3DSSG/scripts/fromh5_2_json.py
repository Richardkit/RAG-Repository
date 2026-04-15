import h5py
import json
import os

# === Configure paths (adjust as needed) ===
RELATIONSHIP_H5 = "data/3RScan_ScanNet20/relationships.h5"
PROPOSAL_H5     = "data/3RScan_ScanNet20/proposals.h5"
SCAN_IDS_TXT    = "data/3RScan_ScanNet20/scan_ids.txt"
CLASSES_TXT     = "data/3RScan_ScanNet20/classes.txt"
OUTPUT_DIR      = "data/3RScan_ScanNet20/scene_graphs_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load basic information ===
scan_ids = [x.strip() for x in open(SCAN_IDS_TXT)]
class_names = [x.strip() for x in open(CLASSES_TXT)]

# === Process only the first scan ===
scan_id = scan_ids[0]
scan_key = scan_id

print(f"🔍 Processing scan_id: {scan_id}")

with h5py.File(PROPOSAL_H5, 'r') as f_prop, h5py.File(RELATIONSHIP_H5, 'r') as f_rel:

    if scan_key not in f_prop or scan_key not in f_rel:
        print(f"❌ Scan {scan_id} not found in H5 files.")
        exit(1)

    node_grp = f_prop[scan_key]["nodes"]

    # === Extract objects ===
    objects = []
    for node_id in node_grp.keys():  # node_id: "0", "1", ...
        node = node_grp[node_id]
        try:
            label_idx = int(node["class"][()])
            label = class_names[label_idx] if 0 <= label_idx < len(class_names) else "unknown"
        except:
            label = "unknown"
        bbox = node["box"][:].tolist() if "box" in node else [0]*6

        objects.append({
            "id": int(node_id),
            "class": label,
            "bbox": bbox
        })

    # === Extract edges (relation matrix) ===
    edges = []
    rel_dataset = f_rel[scan_key]["relation"]
    print("✅ relation shape:", rel_dataset.shape)
    print("✅ relation dtype:", rel_dataset.dtype)

    rel_matrix = rel_dataset[()]  # or rel_dataset[...], this is the recommended approach
    for i in range(rel_matrix.shape[0]):
        for j in range(rel_matrix.shape[1]):
            if rel_matrix[i][j] > 0:
                edges.append({
                    "source": i,
                    "target": j,
                    "relation": int(rel_matrix[i][j])
                })

    # === Save as JSON ===
    out_data = {
        "scan_id": scan_id,
        "objects": objects,
        "edges": edges
    }

    out_path = os.path.join(OUTPUT_DIR, f"{scan_id}.json")
    with open(out_path, "w") as f_out:
        json.dump(out_data, f_out, indent=2)

    print(f"✅ Scene graph saved to: {out_path}")
