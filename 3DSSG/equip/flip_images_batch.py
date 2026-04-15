import os, glob, cv2 as cv
import numpy as np
from math import atan2, degrees

# ======= Configuration (modify as needed) =======
LIST_PATH = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
BASE_DIR  = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan"
SEQ_NAME  = "sequence"        # name of the sequence folder under each scan
OUT_DIR   = "image_filp"      # output folder name (as per your naming)
MIN_LINES = 30                # minimum number of lines required for detection; too few means result is classified as unknown
# ==============================

def _edge_lines(gray):
    """Canny + HoughLinesP to extract line segments"""
    g = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(g, 60, 180)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                           minLineLength=40, maxLineGap=10)
    return lines

def _upright_score_from_lines(lines):
    “””
    Compute “vertical appearance” score: the closer a line segment's direction is to 90°,
    the higher the score (weighted by length).
    Returns (score, n_lines). A larger score means more upright.
    “””
    if lines is None:
        return -1e9, 0
    angles = []
    weights = []
    for x1,y1,x2,y2 in lines[:,0]:
        dx, dy = x2-x1, y2-y1
        L = np.hypot(dx, dy)
        if L <= 0: 
            continue
        ang = degrees(atan2(dy, dx)) % 180.0  # fold into [0, 180)
        # Circular distance to 90° (smaller means more vertical)
        d = min(abs(ang-90), 180-abs(ang-90))
        score = 180 - d*2      # map to a “higher is better” score (linear is sufficient)
        angles.append(ang)
        weights.append(L)
    if not weights:
        return -1e9, 0
    w = np.array(weights)
    s = np.sum((np.array([180 - (np.minimum(np.abs((np.array(angles)-90)), 180-np.abs((np.array(angles)-90))))*2])) * w) / np.sum(w)
    return float(s), len(w)

def _best_upright_rotation(img_bgr):
    “””
    Select the version with the highest “vertical appearance score” among
    {original, 90° counterclockwise, 90° clockwise}.
    Returns (best_rot_flag, best_img, n_lines),
    best_rot_flag in {0, cv.ROTATE_90_COUNTERCLOCKWISE, cv.ROTATE_90_CLOCKWISE}
    “””
    candidates = [(0, img_bgr)]
    candidates.append((cv.ROTATE_90_COUNTERCLOCKWISE, cv.rotate(img_bgr, cv.ROTATE_90_COUNTERCLOCKWISE)))
    candidates.append((cv.ROTATE_90_CLOCKWISE, cv.rotate(img_bgr, cv.ROTATE_90_CLOCKWISE)))

    best = None
    for rot_flag, im in candidates:
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        lines = _edge_lines(gray)
        score, n = _upright_score_from_lines(lines)
        # Too few line segments — treat as extremely poor score
        if n < MIN_LINES:
            score = -1e9
        if (best is None) or (score > best[0]):
            best = (score, rot_flag, im, n)
    return best[1], best[2], best[3]

def process_one_scan(scan_id):
    seq_dir = os.path.join(BASE_DIR, scan_id, SEQ_NAME)
    out_dir = os.path.join(BASE_DIR, scan_id, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    jpgs = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
    changed = 0
    unknown = 0

    for jp in jpgs:
        img = cv.imread(jp)
        if img is None: 
            continue

        # Select the best upright angle
        rot_flag, best_img, n_lines = _best_upright_rotation(img)

        # If the best is the original, skip; otherwise save to image_filp/
        if rot_flag == 0:
            # Possibly too few lines (n_lines==0) to make a determination; skip with “no change”
            if n_lines < MIN_LINES:
                unknown += 1
            continue
        else:
            changed += 1
            out_path = os.path.join(out_dir, os.path.basename(jp))
            cv.imwrite(out_path, best_img)

    return changed, unknown, len(jpgs)

def read_scan_ids(list_path):
    ids = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            # each line in the file is one scan id
            ids.append(line)
    return ids

def main():
    scan_ids = read_scan_ids(LIST_PATH)
    total_changed = total_unknown = total_imgs = 0

    for sid in scan_ids:
        seq_dir = os.path.join(BASE_DIR, sid, SEQ_NAME)
        if not os.path.isdir(seq_dir):
            print(f"[WARN] sequence not found: {seq_dir}")
            continue
        changed, unknown, nimgs = process_one_scan(sid)
        total_changed += changed
        total_unknown += unknown
        total_imgs += nimgs
        print(f"[{sid}] imgs={nimgs}, fixed={changed}, unknown={unknown}")

    print("==== Summary ====")
    print(f"images total: {total_imgs}")
    print(f"fixed total : {total_changed}")
    print(f"unknown     : {total_unknown}")

if __name__ == "__main__":
    main()
