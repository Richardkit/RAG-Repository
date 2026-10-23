#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Tuple

# ======= Configuration (modify as needed) =======
BASE_DIR  = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan"
LIST_PATH = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
# =================================

def _read_valid_scan_list(list_path: str) -> List[str]:
    with open(list_path, "r") as f:
        scans = [x.strip() for x in f.readlines() if x.strip()]
    return scans

def _first_nonempty_line(path: str) -> str:
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                return s
    return ""

def _need_flip_by_first_line(scan_root: str) -> Tuple[bool, str]:
    """
    Only look at the first valid record in flip_list.txt (first non-empty, non-comment line).
    - If angle in {90, 270} -> True (flip needed)
    - Otherwise -> False (no flip needed)
    Returns: (need_flip, reason)
    """
    flip_file = os.path.join(scan_root, "flip_list.txt")
    if not os.path.isfile(flip_file):
        return (False, "flip_list.txt not found (treat as no-flip)")

    first = _first_nonempty_line(flip_file)
    if not first:
        return (False, "empty flip_list (treat as no-flip)")

    parts = first.split()
    if len(parts) < 2:
        return (False, f"bad line: '{first}' (treat as no-flip)")

    ang = parts[1]
    # Some files may write '90°' and similar; parse as robustly as possible
    ang_clean = "".join([c for c in ang if c.isdigit() or c == "-"])
    try:
        ang_val = int(ang_clean)
    except ValueError:
        return (False, f"angle parse fail: '{ang}' (treat as no-flip)")

    if ang_val in (90, 270):
        return (True, f"first angle = {ang_val}")
    return (False, f"first angle = {ang_val}")

def main():
    scans = _read_valid_scan_list(LIST_PATH)
    no_flip_scans: List[str] = []
    need_flip_scans: List[str] = []
    missing_or_bad: List[str] = []

    for sid in scans:
        scan_root = os.path.join(BASE_DIR, sid)
        if not os.path.isdir(scan_root):
            missing_or_bad.append(f"{sid}  (scan dir not found)")
            continue

        need, why = _need_flip_by_first_line(scan_root)
        if need:
            need_flip_scans.append(sid)
        else:
            no_flip_scans.append(sid)

    # Print results
    print("====== Summary ======")
    print(f"Total valid scans: {len(scans)}")
    print(f"No-flip scans   : {len(no_flip_scans)}")
    print(f"Need-flip scans : {len(need_flip_scans)}")
    print(f"Missing/Bad     : {len(missing_or_bad)}")
    print()

    if no_flip_scans:
        print("Scans that DO NOT need flipping (by first line rule):")
        for s in no_flip_scans:
            print(s)
    else:
        print("No scans found that avoid flipping.")

    # Save no-flip list
    out_path = "no_flip_scans.txt"
    with open(out_path, "w") as f:
        for s in no_flip_scans:
            f.write(s + "\n")
    print(f"\nSaved no-flip list to: {out_path}")

    if missing_or_bad:
        print("\n[Warn] Missing/Bad entries:")
        for m in missing_or_bad[:20]:
            print(" -", m)
        if len(missing_or_bad) > 20:
            print(f" ... and {len(missing_or_bad)-20} more")

if __name__ == "__main__":
    main()
