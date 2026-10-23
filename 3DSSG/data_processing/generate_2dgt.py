# -*- coding: utf-8 -*-
"""
Re-generate <scan_id>.2dgt for ALL scans under 3RScan root.

- Optionally download an official 2dgt.zip first (if DOWNLOAD_URL provided).
- For each scan_id under ROOT_3RSCAN, iterate sequence/ frames, read instance label images,
  compute per-instance bbox and occlution_ratio (mask occupancy inside its bbox),
  and write to OUT_2DGT_DIR/<scan_id>.2dgt

OUTPUT .2dgt format (whitespace-separated):
frame_id  object_id  label  occlution_ratio  x_min  y_min  x_max  y_max

Conventions:
- Instance label image is a single-channel integer PNG (or similar).
- BBox coordinates are exported in a bottom-left origin coordinate system:
    x increases upward, y increases rightward.
  i.e., if image array has shape (H, W) indexed by [row, col] with row=0 at top,
  we output:
      x_min = H-1 - row_max
      x_max = H-1 - row_min
      y_min = col_min
      y_max = col_max

All comments in English.
"""

import os
import io
import re
import sys
import glob
import math
import time
import shutil
import zipfile
import urllib.request
from typing import List, Dict, Tuple, Any

import numpy as np
from PIL import Image


# ----------------- CONFIG: edit these three constants if needed -----------------
ROOT_3RSCAN  = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan"
OUT_2DGT_DIR = "/hkfs/work/workspace/scratch/rg5174-myspace/data/2dgt"

# If you have an official 2dgt.zip URL, put it here; leave empty to skip download.
DOWNLOAD_URL = ""  # e.g., "https://your-server/path/to/2dgt.zip"
# --------------------------------------------------------------------------------


# ---------- Utilities ----------

_FRAME_RE = re.compile(r"frame-(\d+)", re.IGNORECASE)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_instance_images(seq_dir: str) -> List[str]:
    """
    Try several common patterns for instance label images.
    Returns a sorted list by frame index (ascending).
    """
    pats = [
        os.path.join(seq_dir, "frame-*.instance.png"),
        os.path.join(seq_dir, "frame-*.instances.png"),
        os.path.join(seq_dir, "frame-*.label.png"),
        os.path.join(seq_dir, "frame-*.seg.png"),
        os.path.join(seq_dir, "*.instance.png"),
        os.path.join(seq_dir, "*.instances.png"),
        os.path.join(seq_dir, "*.label.png"),
        os.path.join(seq_dir, "*.seg.png"),
    ]
    cands = []
    for p in pats:
        cands.extend(glob.glob(p))
    # unique + sort by extracted frame index, fallback to name
    def _idx(pth: str) -> Tuple[int, str]:
        m = _FRAME_RE.search(os.path.basename(pth))
        return (int(m.group(1)) if m else 1_000_000_000, os.path.basename(pth))
    paths = sorted({p for p in cands if os.path.isfile(p)}, key=_idx)
    return paths

def read_instance_image(path: str) -> np.ndarray:
    """
    Read an instance label image as a 2D numpy array of integers.
    """
    im = Image.open(path)
    im = Image.Image.convert(im, "I")  # 32-bit signed integer pixels
    arr = np.array(im)
    return arr

def frame_id_from_path(path: str) -> int:
    m = _FRAME_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1

def compute_bbox_rc(mask: np.ndarray, inst_id: int) -> Tuple[int, int, int, int]:
    """
    Compute tight bounding box in (row/col) coordinates for inst_id.
    Returns (rmin, rmax, cmin, cmax). If not present, raises ValueError.
    """
    ys, xs = np.where(mask == inst_id)
    if ys.size == 0:
        raise ValueError("instance not found")
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

def rc_bbox_to_2dgt_xy(rmin: int, rmax: int, cmin: int, cmax: int, H: int, W: int) -> Tuple[int, int, int, int]:
    """
    Convert (rmin,rmax,cmin,cmax) to .2dgt coords (x_min,y_min,x_max,y_max) with bottom-left origin.
    In image array, row=0 is top; so x = H-1-row, y = col.
    """
    x_min = (H - 1) - rmax
    x_max = (H - 1) - rmin
    y_min = cmin
    y_max = cmax
    # clip to valid ranges just in case
    x_min = max(0, min(x_min, H - 1))
    x_max = max(0, min(x_max, H - 1))
    y_min = max(0, min(y_min, W - 1))
    y_max = max(0, min(y_max, W - 1))
    if x_max < x_min or y_max < y_min:
        # degenerate -> return zeros in caller if needed
        pass
    return int(x_min), int(y_min), int(x_max), int(y_max)

def compute_occupancy(mask: np.ndarray, inst_id: int, rmin: int, rmax: int, cmin: int, cmax: int) -> float:
    """
    Pixel-accurate occupancy inside the bbox: (#pixels==inst_id within box) / (bbox area).
    """
    crop = mask[rmin:rmax+1, cmin:cmax+1]
    box_area = crop.size
    if box_area <= 0:
        return 0.0
    occ = float(np.sum(crop == inst_id)) / float(box_area)
    return occ

def write_2dgt(out_path: str, rows: List[Tuple[int,int,bytes,float,int,int,int,int]]):
    """
    Write rows to .2dgt (whitespace-separated). 
    Row schema: (frame_id, object_id, label_bytes, occlution_ratio, x_min, y_min, x_max, y_max)
    """
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("frame_id object_id label occlution_ratio x_min y_min x_max y_max\n")
        for (fid, oid, lbl_b, occ, x1, y1, x2, y2) in rows:
            # Keep label as bytes literal representation (b'none'), matching historical format.
            label_repr = repr(lbl_b)  # e.g., "b'none'"
            f.write(f"{fid} {oid} {label_repr} {occ:.6f} {x1} {y1} {x2} {y2}\n")

def download_official_2dgt_zip_if_needed(url: str, out_dir: str) -> str:
    """
    If url is non-empty, download 2dgt.zip to out_dir (if not exists) and extract.
    Returns extracted folder path (or out_dir if skipped).
    """
    ensure_dir(out_dir)
    if not url:
        print("[info] DOWNLOAD_URL is empty; skip downloading 2dgt.zip.")
        return out_dir
    zip_path = os.path.join(out_dir, "2dgt.zip")
    if not os.path.isfile(zip_path):
        print(f"[info] downloading 2dgt.zip from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[ok] downloaded to {zip_path}")
    else:
        print(f"[info] found existing {zip_path}, skip download.")
    # extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"[ok] extracted to {out_dir}")
    return out_dir


# ---------- Main pipeline ----------

def generate_2dgt_for_scan(scan_dir: str, out_dir: str) -> int:
    """
    Generate <scan_id>.2dgt for a single scan.
    Returns number of rows written.
    """
    scan_id = os.path.basename(os.path.normpath(scan_dir))
    seq_dir = os.path.join(scan_dir, "sequence")
    if not os.path.isdir(seq_dir):
        print(f"[warn] no sequence/ under {scan_dir}, skip.")
        return 0

    inst_paths = list_instance_images(seq_dir)
    if not inst_paths:
        print(f"[warn] no instance label images found under {seq_dir}, skip.")
        return 0

    rows_out = []
    for ip in inst_paths:
        try:
            mask = read_instance_image(ip)
        except Exception as e:
            print(f"[warn] failed to read {ip}: {e}")
            continue
        H, W = mask.shape[:2]
        fid = frame_id_from_path(ip)
        # instances: unique labels excluding 0
        inst_ids = np.unique(mask)
        inst_ids = inst_ids[(inst_ids != 0)]
        for iid in inst_ids.tolist():
            try:
                rmin, rmax, cmin, cmax = compute_bbox_rc(mask, iid)
                occ = compute_occupancy(mask, iid, rmin, rmax, cmin, cmax)
                x1, y1, x2, y2 = rc_bbox_to_2dgt_xy(rmin, rmax, cmin, cmax, H, W)
                # Keep label as bytes literal; replace with your mapping if desired.
                rows_out.append((fid, int(iid), b'none', float(occ), x1, y1, x2, y2))
            except Exception:
                # instance might be too small/invalid; skip
                continue

    # sort rows by frame_id then object_id
    rows_out.sort(key=lambda r: (r[0], r[1]))

    out_path = os.path.join(out_dir, f"{scan_id}.2dgt")
    write_2dgt(out_path, rows_out)
    print(f"[ok] wrote {len(rows_out)} rows -> {out_path}")
    return len(rows_out)


def main():
    t0 = time.time()

    # 0) Optional: download/extract official 2dgt.zip into OUT_2DGT_DIR (if URL provided)
    download_official_2dgt_zip_if_needed(DOWNLOAD_URL, OUT_2DGT_DIR)

    # 1) Iterate all scan_id dirs under ROOT_3RSCAN
    if not os.path.isdir(ROOT_3RSCAN):
        print(f"[fatal] ROOT_3RSCAN not found: {ROOT_3RSCAN}")
        sys.exit(1)

    ensure_dir(OUT_2DGT_DIR)

    scan_dirs = sorted([os.path.join(ROOT_3RSCAN, d) for d in os.listdir(ROOT_3RSCAN)
                        if os.path.isdir(os.path.join(ROOT_3RSCAN, d))])

    total_rows = 0
    done = 0
    for sd in scan_dirs:
        rows = generate_2dgt_for_scan(sd, OUT_2DGT_DIR)
        total_rows += rows
        done += 1
        if done % 20 == 0:
            print(f"[progress] {done}/{len(scan_dirs)} scans processed...")

    dt = time.time() - t0
    print(f"[ALL DONE] scans={len(scan_dirs)}, rows={total_rows}, elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
