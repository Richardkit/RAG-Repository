import os
from typing import List

BASE_DIR  = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan"
LIST_PATH = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"

def read_scan_ids(list_path: str) -> List[str]:
    with open(list_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def find_flip_file(scan_dir: str) -> str:
    """Prefer flip_list.txt; fall back to filp.txt if it does not exist"""
    p1 = os.path.join(scan_dir, "flip_list.txt")
    p2 = os.path.join(scan_dir, "filp.txt")
    if os.path.isfile(p1):
        return p1
    if os.path.isfile(p2):
        return p2
    return ""

def parse_angle(token: str) -> int:
    try:
        return int(token)
    except Exception:
        return 0

def process_file(fpath: str) -> str:
    """
    Returns 'flip' or 'noflip':
    - More than 5 entries of (90|270) -> set all to 90 -> 'flip'
    - Otherwise set all to 0 -> 'noflip'
    """
    with open(fpath, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # Backup
    with open(fpath + ".bak", "w") as fb:
        fb.write("\n".join(lines) + ("\n" if lines else ""))

    # Count 90/270 entries
    cnt = 0
    parsed = []
    for ln in lines:
        if not ln.strip():
            continue
        # Allow "filename angle" / "angle filename" / angle only (be as robust as possible)
        parts = ln.split()
        if len(parts) == 1:
            # Only one field, try treating it as an angle; use placeholder for filename
            angle = parse_angle(parts[0])
            fname = "UNKNOWN"
        else:
            # Default: last field is the angle, the rest are joined as the filename
            angle = parse_angle(parts[-1])
            fname = " ".join(parts[:-1])
        parsed.append((fname, angle))
        if angle in (90, 270):
            cnt += 1

    # Determine result and write back
    if cnt > 5:
        new_angle = 90
        tag = "flip"
    else:
        new_angle = 0
        tag = "noflip"

    with open(fpath, "w") as f:
        for fname, _ in parsed:
            if fname == "UNKNOWN":
                f.write(f"{new_angle}\n")
            else:
                f.write(f"{fname} {new_angle}\n")

    return tag

def main():
    scan_ids = read_scan_ids(LIST_PATH)
    noflip_scans = []

    for sid in scan_ids:
        scan_dir = os.path.join(BASE_DIR, sid)
        flip_file = find_flip_file(scan_dir)
        if not flip_file:
            print(f"[WARN] flip_list not found for {sid}")
            continue
        result = process_file(flip_file)
        if result == "noflip":
            noflip_scans.append(sid)
        print(f"[OK] {sid}: updated {os.path.basename(flip_file)} -> {result}")

    print("\n=== Scans not needing flip (all set to 0) ===")
    if noflip_scans:
        for sid in noflip_scans:
            print(sid)
    else:
        print("(none)")

if __name__ == "__main__":
    main()
