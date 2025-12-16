import json
import os
import random
from PIL import Image, ImageDraw

def visual_test(gt_bboxes_path, scan_dir, output_dir):
    """
    Visualize bounding boxes for a randomly selected object (excluding wall and floor).

    Args:
        gt_bboxes_path (str): Path to the gt_bboxes.json file.
        scan_dir (str): Path to the directory containing scan folders.
        output_dir (str): Path to save the edited images.
    """
    # Load the gt_bboxes.json file
    with open(gt_bboxes_path, "r") as f:
        gt_bboxes = json.load(f)

    # Filter objects to exclude "wall" and "floor"
    valid_objects = []
    for scan_id, objects in gt_bboxes.items():
        for obj_id, obj_data in objects.items():
            if obj_data["name"].lower() not in ["wall", "floor"] and obj_data["bounding_boxes"]:
                valid_objects.append((scan_id, obj_id, obj_data))

    if not valid_objects:
        print("No valid objects found in gt_bboxes.json.")
        return

    # Randomly select an object
    scan_id, obj_id, obj_data = random.choice(valid_objects)
    print(f"Selected object: {obj_data['name']} (ID: {obj_id}) in scan {scan_id}")

    # Get the frames and bounding boxes for the selected object
    seen_frames = obj_data["seen_frames"]
    print(f"the object is: {obj_data['name']}")
    bounding_boxes = obj_data["bounding_boxes"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each frame where the object appears
    for frame, bbox in zip(seen_frames, bounding_boxes):
        frame_path = os.path.join(scan_dir, scan_id, "sequence", f"{frame}.color.jpg")
        if not os.path.exists(frame_path):
            print(f"Frame {frame} not found in scan {scan_id}. Skipping.")
            continue

        # Load the image
        image = Image.open(frame_path)
        draw = ImageDraw.Draw(image)

        # Draw the bounding box with a lighter yellow color
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

        # Save the edited image
        output_path = os.path.join(output_dir, f"{scan_id}_{frame}_{obj_id}.jpg")
        image.save(output_path)
        print(f"Saved edited image: {output_path}")

if __name__ == "__main__":
    # Define file paths
    gt_bboxes_path = "/hkfs/work/workspace/scratch/rg5174-myspace/RAG_baseline_Qwen3/evaluate/gt_bboxes.json"
    scan_dir = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan"
    output_dir = "/hkfs/work/workspace/scratch/rg5174-myspace/RAG_baseline_Qwen3/evaluate"

    # Run the visual test
    visual_test(gt_bboxes_path, scan_dir, output_dir)