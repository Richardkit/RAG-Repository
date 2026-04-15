# global_pre_indexing_patch.py (example: concatenate all patches into a single global library)
import os, re, glob, pickle
import numpy as np
from tqdm import tqdm

# ========= Paths to modify =========
PATCH_ROOT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_patch_embeddings_without_validset_siglip2"  
OUT_DIR    = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_patch_embeddings_without_validset_siglip2_global"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_X_NPY   = os.path.join(OUT_DIR, "all_patch_embs.npy")   # (N,768) float32
OUT_KEYS    = os.path.join(OUT_DIR, "all_patch_keys.pkl")   # len=N list[str]

D = 768
SAVE_FLOAT16 = False  # set True to save space, but convert back to float32 when building the index

# If the filename contains 'frame', you can parse it with a regex
# e.g.: .../scan_id/sequence/frame-000123.color.jpg -> frame_id=123
FRAME_RE = re.compile(r"frame[-_](\d+)", re.IGNORECASE)

def parse_scan_frame_from_path(p: str):
    # Adjust this according to your actual filename/directory structure
    # Here we assume the npz filename looks like: <scanid>__frame-000123.npz
    base = os.path.basename(p)
    if "__" in base:
        scan_id = base.split("__")[0]
    else:
        # or take it from the parent directory
        scan_id = os.path.basename(os.path.dirname(p))
    m = FRAME_RE.search(base)
    frame_id = int(m.group(1)) if m else -1
    return scan_id, frame_id

def main():
    files = sorted(glob.glob(os.path.join(PATCH_ROOT, "**/*.npz"), recursive=True))
    if len(files) == 0:
        raise RuntimeError(f"No npz files found: {PATCH_ROOT}")

    keys = []
    emb_chunks = []

    total = 0
    for fp in tqdm(files, desc="Load npz"):
        data = np.load(fp, allow_pickle=True)
        emb = data["emb"]   # [m,768]
        if emb.ndim != 2 or emb.shape[1] != D:
            raise RuntimeError(f"Incorrect dimensions: {fp}, emb.shape={emb.shape}")

        scan_id, frame_id = parse_scan_frame_from_path(fp)

        m = emb.shape[0]
        # assign a key to each patch (used for reverse lookup)
        # define the key format yourself, but it must be parseable back to scan/frame/patch
        for pid in range(m):
            keys.append(f"{scan_id}|{frame_id}|p{pid}")

        if SAVE_FLOAT16:
            emb = emb.astype(np.float16)
        else:
            emb = emb.astype(np.float32)

        emb_chunks.append(emb)
        total += m

    # Concatenate
    X = np.concatenate(emb_chunks, axis=0)
    print("[OK] total patches =", X.shape[0], "D =", X.shape[1])

    # Save to disk
    np.save(OUT_X_NPY, X)
    with open(OUT_KEYS, "wb") as f:
        pickle.dump(keys, f)

    print("[SAVE]", OUT_X_NPY)
    print("[SAVE]", OUT_KEYS)

if __name__ == "__main__":
    main()
