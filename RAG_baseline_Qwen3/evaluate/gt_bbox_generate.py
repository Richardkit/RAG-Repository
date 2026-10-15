import json
import os
import sys

from tqdm import tqdm

# Add the directory containing val_m_frame_level_gt_extract.py to the Python path
sys.path.append("/hkfs/work/workspace/scratch/rg5174-ragspace2/Evaluate")

from val_m_frame_level_gt_extract import merge_frame_level_sg_1

def generate_gt_bboxes(scan_dir, valid_scans_path, relationship_h5_path, output_path):
    """
    Generate the gt_bboxes.json file for valid scans.

    Args:
        scan_dir (str): Path to the directory containing scan folders.
        valid_scans_path (str): Path to the 3RScan_validation_subset_20pct.txt file.
        relationship_h5_path (str): Path to the relationships.h5 file.
        output_path (str): Path to save the generated gt_bboxes.json file.
    """
    # Load valid scans
    with open(valid_scans_path, "r") as f:
        valid_scans = set(line.strip() for line in f.readlines())

    scan_list = os.listdir(scan_dir)

    # Initialize the output structure
    # gt_bboxes = {}

    # Process each valid scan
    # for scan_id in valid_scans:
    for scan_id in tqdm(sorted(valid_scans), desc="Processing scans"):

        gt_bboxes = {}
        scan_path = os.path.join(scan_dir, scan_id, "sequence")
        if not os.path.exists(scan_path):
            print(f"Warning: sequence folder not found for scan {scan_id}")
            continue

        # Collect all query images for the scan
        query_images = sorted([
            os.path.join(scan_path, f)
            for f in os.listdir(scan_path)
            if f.endswith(".color.jpg")
        ])

        if not query_images:
            print(f"Warning: No query images found for scan {scan_id}")
            continue

        # Use merge_frame_level_sg_1 to compute bounding boxes
        merged_sg = merge_frame_level_sg_1(query_images, scan_id, relationship_h5_path)

        # Extract objects and their bounding boxes
        gt_bboxes[scan_id] = {}
        for obj in merged_sg["objects"]:
            obj_id = str(obj["id"])
            obj_name = obj.get("name", "unknown")
            bounding_boxes = obj["bboxes"]

            # Filter out invalid bounding boxes and their corresponding frames
            valid_bboxes = []
            valid_frames = []
            for i, bbox in enumerate(bounding_boxes):
                if bbox is not None and bbox != [0, 0, 0, 0]:
                    valid_bboxes.append(bbox)
                    valid_frames.append(query_images[i].split("/")[-1].replace(".color.jpg", ""))

            # Only include the object if it has valid bounding boxes
            if valid_bboxes:
                gt_bboxes[scan_id][obj_id] = {
                    "name": obj_name,
                    "seen_frames": valid_frames,
                    "bounding_boxes": valid_bboxes
                }

        with open(os.path.join(scan_dir, scan_id, f"{scan_id}_gt_bboxes.json"), "w") as f:

            json.dump(gt_bboxes, f, indent=4)

        print(f"Generated gt_bboxes for scan {scan_id} at {os.path.join(scan_dir, scan_id, f'{scan_id}_gt_bboxes.json')}")

    # Save the result to the output file
    # with open(output_path, "w") as f:
    #     json.dump(gt_bboxes, f, indent=4)
    # print(f"Generated gt_bboxes.json at {output_path}")

if __name__ == "__main__":
    # Define file paths
    scan_dir = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan"
    valid_scans_path = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
    relationship_h5_path = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/data_GT/3RScan_3RScan160/relationships.h5"
    output_path = "/hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_baseline_Qwen3/evaluate/gt_bboxes.json"

    # Generate gt_bboxes.json
    generate_gt_bboxes(scan_dir, valid_scans_path, relationship_h5_path, output_path)