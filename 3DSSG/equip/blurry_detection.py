import os
import cv2
import json

def is_blurry(image, threshold):
    """
    Detect if an image is blurry using the Laplacian variance method.

    Args:
        image (numpy.ndarray): The input image.
        threshold (float): Variance threshold below which the image is considered blurry.

    Returns:
        bool: True if the image is blurry, False otherwise.
        float: The variance of the Laplacian.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = laplacian.var()

    # Determine if the image is blurry
    return laplacian, variance < threshold, variance

def process_image_sequence(folder_path, output_folder, img_non_blurry, img_blurry, threshold=15.0):
    """
    Process a sequence of images in a folder to check if they are blurry and save blurry images.

    Args:
        folder_path (str): Path to the folder containing the image sequence.
        output_folder (str): Path to the folder where blurry images will be saved.
        threshold (float): Variance threshold below which an image is considered blurry.
    """
    # Ensure the output folder exists
    
    os.makedirs(output_folder, exist_ok=True)

    # Get a sorted list of all image files in the folder
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print(f"No images found in folder: {folder_path}")
        return

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue

        # Check if the image is blurry
        laplacian, blurry, variance = is_blurry(image, threshold)

        # Save blurry images with their variance in the filename
        if blurry:
            # output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_blurry_{variance:.2f}.jpg")
            # cv2.imwrite(output_path, image)
            # print(f"Saved blurry image: {output_path}")
            img_blurry.append(image_path)
        else:
            img_non_blurry.append(image_path)


root_path = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan"
# Path to the folder where blurry images will be saved
output_folder = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/equip/blurry"


with open("/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_validation_set.txt", 'r') as f:
    val_scenes = f.read().splitlines()
    
img_blurry = []
img_non_blurry = []

for scene in val_scenes:

    scene_rgb_path = os.path.join(root_path, scene, "sequence")

    # Process the image sequence for the current scene
    process_image_sequence(scene_rgb_path, output_folder,img_blurry, img_non_blurry)



lack_obj_frames_blurry = []
lack_obj_frames_non_blurry = []

for p in img_non_blurry:
    frame_id = p.split("/")[-1].split(".")[0]
    scan_id = p.split("/")[-3]
    frame_sg_path = os.path.join(root_path, scan_id, "frame_sg", f"{frame_id}.json")

    if not os.path.exists(frame_sg_path):
        print(f"Frame SG file does not exist: {frame_sg_path}")
        continue

    with open(frame_sg_path, 'r') as f:
        frame_sg = json.load(f)

    objects = frame_sg.get("objects", [])
    obj_num = len(objects)

    if obj_num <= 3:
        lack_obj_frames_non_blurry.append(p)

for p in img_blurry:
    frame_id = int(p.split("/")[-1].split("-")[-1].split(".")[0])
    scan_id = p.split("/")[-3]
    frame_sg_path = os.path.join(root_path, scan_id, "frame_sg", f"{frame_id:06d}.json")

    if not os.path.exists(frame_sg_path):
        print(f"Frame SG file does not exist: {frame_sg_path}")
        continue

    with open(frame_sg_path, 'r') as f:
        frame_sg = json.load(f)

    objects = frame_sg.get("objects", [])
    obj_num = len(objects)

    if obj_num <= 3:
        lack_obj_frames_blurry.append(p)

print("Lack object frames (non-blurry):", len(lack_obj_frames_non_blurry))
print("Lack object frames (blurry):", len(lack_obj_frames_blurry))

