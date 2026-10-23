#!/usr/bin/env python3
import os, json, pickle, argparse
from pathlib import Path
import numpy as np

def atomic_replace(tmp: Path, final: Path):
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(tmp), str(final))

def read_index_keys(index_path: Path, scan_id: str, count: int):
    """Read frame key order from index.json; fall back to frame-000000.. if missing or invalid."""
    if not index_path.is_file():
        return [f"{scan_id}/frame-{i:06d}" for i in range(count)]
    try:
        meta = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(meta, dict) and "frames" in meta:
            ks = []
            for i, it in enumerate(meta["frames"]):
                k = it.get("key") or it.get("name") or f"frame-{i:06d}"
                if not str(k).startswith(scan_id + "/"):
                    k = f"{scan_id}/{k}"
                ks.append(k)
            if len(ks) == count:
                return ks
    except Exception:
        pass
    return [f"{scan_id}/frame-{i:06d}" for i in range(count)]

def load_npz_matrix(npz_path: Path, keys_in_order=None):
    """
    Compatible with two per-scan file formats:
      A) Single 2D array: np.savez(npz_path, arr_0=embs) or np.savez(npz_path, embs=...)
      B) Multiple 1D arrays: np.savez(npz_path, **{key: vec}) (one key per frame)
    Returns: (mat, (M,D), names_found or None)
    """
    with np.load(npz_path, allow_pickle=False) as z:
        files = list(z.files)
        if not files:
            return np.zeros((0, 0), dtype=np.float32), None, []

        # Case A: only one entry and it is 2D (regardless of whether the key name is arr_0)
        first = files[0]
        if len(files) == 1 and z[first].ndim == 2:
            mat = z[first].astype(np.float32, copy=False)
            M, D = int(mat.shape[0]), int(mat.shape[1])
            # For a single array, keys are determined by the outer index.json (returning None means “use the order from the outer caller”)
            return mat, (M, D), None

        # Case B: one named array per frame (typically 1D of size 768)
        if keys_in_order is None:
            # When no index order is available, sort by filename
            files_sorted = sorted(files)
            vecs = [z[name].astype(np.float32, copy=False).reshape(-1) for name in files_sorted]
            D = int(vecs[0].shape[0])
            mat = np.vstack(vecs)
            return mat, (mat.shape[0], D), files_sorted
        else:
            # When an index order is available, follow it strictly (keys inside npz have no scan_id prefix)
            bare = [k.split("/", 1)[-1] for k in keys_in_order]
            picked = []
            names_found = []
            for name in bare:
                if name in z:
                    v = z[name].astype(np.float32, copy=False).reshape(-1)
                    picked.append(v)
                    names_found.append(name)
            if not picked:
                return np.zeros((0, 0), dtype=np.float32), None, []
            D = int(picked[0].shape[0])
            mat = np.vstack(picked)
            return mat, (mat.shape[0], D), names_found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset_siglip2/frame_embeddings",
                    help="root directory of per-scan frame_embeddings")
    ap.add_argument("--outdir", default="/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset_siglip2",
                    help="output directory for global all_vecs.npy / all_keys.pkl")
    ap.add_argument("--vec-only", action="store_true", help="only rebuild all_vecs.npy")
    ap.add_argument("--keys-only", action="store_true", help="only rebuild all_keys.pkl")
    ap.add_argument("--dedup", action="store_true", help="deduplicate repeated keys (rare, but safe)")
    args = ap.parse_args()

    ROOT = Path(args.root)
    OUTDIR = Path(args.outdir)
    VEC_OUT = OUTDIR / "all_vecs.npy"
    KEY_OUT = OUTDIR / "all_keys.pkl"

    scan_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    if not scan_dirs:
        print("No per-scan folders found. Nothing to do.")
        return

    # First pass: count total entries and dimension
    total = 0
    dim = None
    per_scan_meta = []  # (scan_id, npz_path, index_path, count)
    for sd in scan_dirs:
        scan_id = sd.name
        npz = sd / "frame_rgb_embeddings.npz"
        if not npz.is_file():
            continue
        idx = sd / "index.json"
        # Pre-fetch key order from index (for named-array npz files)
        keys_order = None
        if idx.is_file():
            try:
                m = len(json.loads(idx.read_text(encoding="utf-8")).get("frames", []))
            except Exception:
                m = 0
        else:
            m = 0
        # Quick dimension check (read one entry)
        with np.load(npz, allow_pickle=True) as z:
            if "arr_0" in z and z["arr_0"].ndim == 2:
                m = int(z["arr_0"].shape[0])
                d_here = int(z["arr_0"].shape[1])
            else:
                # Take the first entry
                first = z.files[0]
                v = z[first].reshape(-1)
                d_here = int(v.shape[0])
                if m == 0:
                    m = len(z.files)
        if m == 0:
            continue
        per_scan_meta.append((scan_id, npz, idx, m, d_here))
        if dim is None:
            dim = d_here
        elif dim != d_here:
            raise ValueError(f"Dimension mismatch in {npz}: expect {dim}, got {d_here}")
        total += m

    if total == 0 or dim is None:
        print("No embeddings to consolidate.")
        return

    print(f"Found {len(per_scan_meta)} scans, total rows={total}, dim={dim}")

    # Second pass: actually write out
    # --- keys ---
    if not args.vec_only:
        keys = []
    # --- vecs ---
    if not args.keys_only:
        tmp_vec = VEC_OUT.with_suffix(".npy.tmp")
        mmap = np.memmap(tmp_vec, mode="w+", dtype=np.float32, shape=(total, dim))
    offset = 0

    seen = set() if args.dedup else None

    for scan_id, npz, idx, m, _d in per_scan_meta:
        # Determine key order first
        keys_order = read_index_keys(idx, scan_id, m)
        # Load matrix (compatible with both npz formats)
        mat, _shape, names_found = load_npz_matrix(npz, keys_in_order=keys_order)
        if mat.size == 0:
            continue
        m_here = int(mat.shape[0])

        # keys list
        if not args.vec_only:
            ks = keys_order if names_found is None else [f"{scan_id}/{n}" for n in names_found]
            if seen is not None:
                # Dedup: filter out keys already seen
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

        # Flush matrix to disk
        if not args.keys_only and m_here > 0:
            mmap[offset:offset+m_here] = mat
            offset += m_here

    # Final flush to disk
    if not args.keys_only:
        del mmap  # flush
        atomic_replace(tmp_vec, VEC_OUT)
        print(f"✅ wrote {VEC_OUT}  shape=({total}, {dim})")
    if not args.vec_only:
        tmp_pkl = KEY_OUT.with_suffix(".pkl.tmp")
        with open(tmp_pkl, "wb") as f:
            pickle.dump(keys, f, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_replace(tmp_pkl, KEY_OUT)
        print(f"✅ wrote {KEY_OUT}  count={len(keys)}")

if __name__ == "__main__":
    main()
