#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidate per-scan PATCH-LEVEL embeddings into global all_vecs.npy / all_keys.pkl

Expected per scan folder structure:
  <ROOT>/<scan_id>/frame_patch_embeddings.npz
  <ROOT>/<scan_id>/patch_index.json    # contains a list of patches with "key": "frame-XXXXXX|rXcY"

Output:
  <OUTDIR>/all_vecs.npy   # shape = (total_patches, dim)
  <OUTDIR>/all_keys.pkl   # list[str], each like "<scan_id>/frame-XXXXXX|rXcY"

Usage:
  python global_pre_indexing_patches.py \
    --root /path/to/patch_embeddings \
    --outdir /path/to/global_patch_index \
    [--vec-only] [--keys-only] [--dedup]
"""

import os, json, pickle, argparse
from pathlib import Path
import numpy as np

def atomic_replace(tmp: Path, final: Path):
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(tmp), str(final))

def read_patch_keys(index_path: Path, scan_id: str, fallback_count: int, npz_files=None):
    """
    Read patch keys order from patch_index.json.
    If missing or invalid, fallback to npz file keys (sorted) or a generated range.
    Returns a list of keys prefixed with scan_id: "<scan_id>/<bare_key>"
    """
    # 1) Prefer JSON
    if index_path.is_file():
        try:
            meta = json.loads(index_path.read_text(encoding="utf-8"))
            patches = meta.get("patches", [])
            keys = []
            for it in patches:
                k = it.get("key")
                if not k:
                    # allow reconstruct from row/col + frame
                    frm = it.get("frame_id")
                    r = it.get("row"); c = it.get("col")
                    if frm is not None and r is not None and c is not None:
                        k = f"frame-{int(frm):06d}|r{int(r)}c{int(c)}"
                if k:
                    if not str(k).startswith(scan_id + "/"):
                        k = f"{scan_id}/{k}"
                    keys.append(k)
            if keys:
                return keys
        except Exception:
            pass

    # 2) Fallback to npz internal keys order (sorted)
    if npz_files:
        files_sorted = sorted(npz_files)
        return [f"{scan_id}/{name}" for name in files_sorted]

    # 3) Last resort: dummy incremental keys
    return [f"{scan_id}/patch-{i:06d}" for i in range(fallback_count)]

def load_npz_matrix_patch(npz_path: Path, expected_order=None):
    """
    Load patch-level npz.
    We support:
      - Multiple named 1D arrays: { key: vec }
      - Single 2D array: arr_0 [M,D] (rare for patches, but supported)
    If expected_order is provided (bare keys without scan prefix),
    we will pick vectors in that order when possible.

    Returns: (mat [M,D], dim, names_found or None)
    """
    with np.load(npz_path, allow_pickle=False) as z:
        files = list(z.files)
        if not files:
            return np.zeros((0,0), dtype=np.float32), 0, []

        first = files[0]
        # single 2D array case
        if len(files) == 1 and z[first].ndim == 2:
            mat = z[first].astype(np.float32, copy=False)
            return mat, int(mat.shape[1]), None

        # multi named arrays (most common for our patch npz)
        if expected_order is None:
            files_sorted = sorted(files)
            vecs = [z[name].astype(np.float32, copy=False).reshape(-1) for name in files_sorted]
            D = int(vecs[0].shape[0])
            mat = np.vstack(vecs)
            return mat, D, files_sorted
        else:
            picked, names_found = [], []
            for fullkey in expected_order:
                bare = fullkey.split("/", 1)[-1]  # remove scan_id prefix
                if bare in z:
                    v = z[bare].astype(np.float32, copy=False).reshape(-1)
                    picked.append(v)
                    names_found.append(bare)
            if not picked:
                return np.zeros((0,0), dtype=np.float32), 0, []
            D = int(picked[0].shape[0])
            mat = np.vstack(picked)
            return mat, D, names_found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",   default="/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_embeddings_without_validset_siglip2/frame_embeddings",
                    help="Root folder that contains per-scan patch embeddings")
    ap.add_argument("--outdir", default="/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_patch_level_embeddings_without_validset_siglip2",
                    help="Output folder for all_vecs.npy / all_keys.pkl")
    ap.add_argument("--vec-only",  action="store_true", help="Only rebuild all_vecs.npy")
    ap.add_argument("--keys-only", action="store_true", help="Only rebuild all_keys.pkl")
    ap.add_argument("--dedup",     action="store_true", help="Deduplicate duplicate keys (safety)")
    args = ap.parse_args()

    ROOT = Path(args.root)
    OUTDIR = Path(args.outdir)
    VEC_OUT = OUTDIR / "all_vecs.npy"
    KEY_OUT = OUTDIR / "all_keys.pkl"

    scan_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    if not scan_dirs:
        print("No per-scan folders found. Nothing to do.")
        return

    # First pass: collect sizes & dim
    total = 0
    dim = None
    per_scan = []  # (scan_id, npz_path, json_path, count, dim_here, npz_files_list_or_None)
    for sd in scan_dirs:
        scan_id = sd.name
        npz = sd / "frame_patch_embeddings.npz"
        if not npz.is_file():
            continue
        jmeta = sd / "patch_index.json"

        # Peek npz to estimate count & dim
        with np.load(npz, allow_pickle=False) as z:
            files = list(z.files)
            if not files:
                continue
            if len(files) == 1 and z[files[0]].ndim == 2:
                m = int(z[files[0]].shape[0])
                d_here = int(z[files[0]].shape[1])
                npz_files_list = None
            else:
                # multi named arrays
                m = len(files)
                first = files[0]
                d_here = int(z[first].reshape(-1).shape[0])
                npz_files_list = files

        per_scan.append((scan_id, npz, jmeta, m, d_here, npz_files_list))
        if dim is None:
            dim = d_here
        elif dim != d_here:
            raise ValueError(f"Dimension mismatch in {npz}: got {d_here}, expect {dim}")
        total += m

    if total == 0 or dim is None:
        print("No embeddings to consolidate.")
        return

    print(f"Found {len(per_scan)} scans  |  total patches={total}  |  dim={dim}")

    # Prepare outputs
    if not args.vec_only:
        keys = []
    if not args.keys_only:
        tmp_vec = VEC_OUT.with_suffix(".npy.tmp")
        mmap = np.memmap(tmp_vec, mode="w+", dtype=np.float32, shape=(total, dim))
    offset = 0
    seen = set() if args.dedup else None

    # Second pass: write
    for scan_id, npz, jmeta, count, _d, npz_files_list in per_scan:
        # 1) Decide key order
        keys_order = read_patch_keys(jmeta, scan_id, count, npz_files=npz_files_list)

        # 2) Load matrix following key order when possible
        mat, _dim, names_found = load_npz_matrix_patch(npz, expected_order=keys_order)

        if mat.size == 0:
            continue
        m_here = int(mat.shape[0])

        # 3) Keys
        if not args.vec_only:
            if names_found is None:
                ks = keys_order  # already prefixed
            else:
                ks = [f"{scan_id}/{n}" for n in names_found]

            if seen is not None:
                keep = [k not in seen for k in ks]
                if any(keep):
                    ks = [k for k, t in zip(ks, keep) if t]
                    mat = mat[np.nonzero(keep)[0], :]
                    m_here = int(mat.shape[0])
                    for k in ks: seen.add(k)
                else:
                    ks = []
                    m_here = 0
            keys.extend(ks)

        # 4) Write vecs
        if not args.keys_only and m_here > 0:
            mmap[offset:offset+m_here] = mat
            offset += m_here

    # Finalize
    if not args.keys_only:
        del mmap
        atomic_replace(tmp_vec, VEC_OUT)
        print(f"✅ wrote {VEC_OUT}  shape=({total}, {dim})")
    if not args.vec_only:
        tmp_pkl = KEY_OUT.with_suffix(".pkl.tmp")
        with open(tmp_pkl, "wb") as f:
            pickle.dump(keys, f, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_replace(tmp_pkl, KEY_OUT)
        print(f"✅ wrote {KEY_OUT}  count={len(keys)})")

if __name__ == "__main__":
    main()
