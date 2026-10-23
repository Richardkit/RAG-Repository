import os
import sys
import time
import glob
import pickle
import numpy as np
from tqdm import tqdm

# ---- faiss import: some environments use faiss, others use faiss-cpu ----
try:
    import faiss  # type: ignore
except Exception as e:
    print("[FATAL] Cannot import faiss. Make sure faiss is installed in this env.")
    raise

# =========================
# You only need to change the paths here
# =========================

# If your actual structure is .../frame_embeddings/<scan_id>/, point PATCH_ROOT to frame_embeddings
PATCH_ROOT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_patch_embeddings_without_validset_siglip2/frame_embeddings"

# Output directory (index and global keys are saved here)
OUT_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_patch_embeddings_without_validset_siglip2"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_INDEX_PATH = os.path.join(OUT_DIR, "patch_index_002.ivfflat")
OUT_KEYS_PATH  = os.path.join(OUT_DIR, "all_patch_keys_002.pkl")
OUT_LOG_PATH   = os.path.join(OUT_DIR, "build_faiss_patch_002.log")

# =========================
# FAISS / vector parameters
# =========================
D = 768
METRIC = "cosine"   # cosine via inner product + L2 normalize
INDEX_TYPE = "ivfflat"  # this is the one you want

KB_FRACTION = 1
KB_MODE = "random"   # or "top"
KB_SEED = 0

# IVF parameters (no need to change; but increase NLIST if the library is very large)
NLIST = 4096
NPROBE = 16

# Upper limit of training samples (too large slows training, too small makes it unstable)
TRAIN_MAX = 100_000
# Maximum number of entries sampled per scan into the training pool (prevents a single large scan from overwhelming the pool)
TRAIN_PER_SCAN_MAX = 4000

# Chunk size for add (avoids excessive memory usage from adding too many at once)
ADD_CHUNK = 200_000

# Enable fast key validation: keys must be strings
STRICT_KEY_TYPE = True

# =========================
# Utility functions
# =========================
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(OUT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x: float32 [N,D]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def find_scan_dirs(root: str):
    """
    Find all directories containing frame_patch_embeddings.npy + patch_keys.pkl
    """
    npy_files = glob.glob(os.path.join(root, "**", "frame_patch_embeddings.npy"), recursive=True)
    scan_dirs = []
    for npy in npy_files:
        d = os.path.dirname(npy)
        pkl = os.path.join(d, "patch_keys.pkl")
        if os.path.exists(pkl):
            scan_dirs.append(d)
    scan_dirs = sorted(list(set(scan_dirs)))
    return scan_dirs

def load_one_scan(scan_dir: str):
    npy_path = os.path.join(scan_dir, "frame_patch_embeddings.npy")
    pkl_path = os.path.join(scan_dir, "patch_keys.pkl")

    if not os.path.exists(npy_path):
        raise FileNotFoundError(npy_path)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)

    # read with mmap to save memory
    X = np.load(npy_path, mmap_mode="r")  # may be float16/float32
    with open(pkl_path, "rb") as f:
        keys = pickle.load(f)

    # ---- Hard checks ----
    if not isinstance(X, np.ndarray):
        raise RuntimeError(f"X is not ndarray: {npy_path}")

    if X.ndim != 2:
        raise RuntimeError(f"Bad X.ndim={X.ndim} in {npy_path}, expected 2")

    if X.shape[1] != D:
        raise RuntimeError(f"Bad D in {npy_path}: got {X.shape[1]}, expected {D}")

    if not isinstance(keys, (list, tuple)):
        raise RuntimeError(f"keys is not list/tuple in {pkl_path}: {type(keys)}")

    if len(keys) != X.shape[0]:
        raise RuntimeError(f"Key count mismatch in {scan_dir}: X={X.shape[0]} vs keys={len(keys)}")

    if STRICT_KEY_TYPE:
        # only check a few entries at the front and back to avoid O(N) slowness
        for idx in [0, min(1, len(keys)-1), max(0, len(keys)-2), len(keys)-1]:
            if len(keys) == 0:
                break
            if not isinstance(keys[idx], str):
                raise RuntimeError(f"Key is not str in {pkl_path}, keys[{idx}]={type(keys[idx])}")
            
    import re
    UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

    def infer_scan_id(scan_dir: str) -> str:
        cur = os.path.abspath(scan_dir)
        for _ in range(4):  # search up at most 4 levels, enough to cover .../<uuid>/sequence/...
            base = os.path.basename(cur)
            if UUID_RE.match(base):
                return base.lower()
            cur = os.path.dirname(cur)
        # fallback: return the original basename at minimum (will trigger subsequent checks)
        return os.path.basename(scan_dir)

    # used inside load_one_scan():
    scan_id = infer_scan_id(scan_dir)
    if not UUID_RE.match(scan_id):
        raise RuntimeError(f"Cannot infer UUID scan_id from scan_dir={scan_dir}, got scan_id={scan_id}")


    return X, keys, scan_id


def make_index():
    if METRIC == "cosine":
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, D, NLIST,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT
        )

    else:
        raise ValueError("Only cosine is supported in this script.")
    return index

# =========================
# Main pipeline
# =========================
def main():
    # Clear/create log
    with open(OUT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    log("==== Build FAISS (one-pass) started ====")
    log(f"PATCH_ROOT = {PATCH_ROOT}")
    log(f"OUT_DIR    = {OUT_DIR}")
    log(f"D={D}, METRIC={METRIC}, INDEX_TYPE={INDEX_TYPE}, NLIST={NLIST}, NPROBE={NPROBE}")
    log(f"TRAIN_MAX={TRAIN_MAX}, TRAIN_PER_SCAN_MAX={TRAIN_PER_SCAN_MAX}, ADD_CHUNK={ADD_CHUNK}")

    scan_dirs = find_scan_dirs(PATCH_ROOT)
    num = len(scan_dirs)
    keep_n = max(1, int(round(num * KB_FRACTION)))

    rng = np.random.default_rng(KB_SEED)
    idx = rng.choice(num, size=keep_n, replace=False)
    scan_dirs = [scan_dirs[i] for i in sorted(idx)]
    if len(scan_dirs) == 0:
        log("[FATAL] No scan dirs found. Check PATCH_ROOT.")
        sys.exit(1)

    log(f"Found scan dirs: {len(scan_dirs)}")

    # 1) Collect training samples
    train_buf = []
    train_count = 0
    total_vectors_est = 0

    log("Collecting training samples...")
    for d in tqdm(scan_dirs, desc="train-scan"):
        X_mmap, _keys, _sid = load_one_scan(d)
        n = int(X_mmap.shape[0])
        total_vectors_est += n

        if train_count >= TRAIN_MAX:
            continue

        take = min(TRAIN_PER_SCAN_MAX, n, TRAIN_MAX - train_count)
        if take <= 0:
            continue

        # sampling (random if larger than take, otherwise take all)
        if n > take:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=take, replace=False)
            Xs = np.array(X_mmap[idx])  # copy to RAM
        else:
            Xs = np.array(X_mmap)       # copy to RAM

        # to float32 + normalize
        Xs = Xs.astype(np.float32, copy=False)
        Xs = l2_normalize(Xs)

        train_buf.append(Xs)
        train_count += Xs.shape[0]

    if train_count == 0:
        log("[FATAL] No training samples collected. Check your embedding files.")
        sys.exit(1)

    X_train = np.concatenate(train_buf, axis=0)
    # Final hard check
    if X_train.ndim != 2 or X_train.shape[1] != D:
        log(f"[FATAL] Bad X_train shape: {X_train.shape}")
        sys.exit(1)

    log(f"Training samples: {X_train.shape}")

    # 2) Build index + train
    index = make_index()
    log("Training FAISS index...")
    t0 = time.time()
    index.train(X_train)
    t1 = time.time()
    log(f"FAISS train done in {t1 - t0:.1f}s. is_trained={index.is_trained}")

    if not index.is_trained:
        log("[FATAL] Index training failed (index.is_trained=False).")
        sys.exit(1)

    # 3) Add scan by scan (streaming, without assembling a large global matrix)
    all_keys = []
    added = 0

    log("Adding vectors to FAISS index (streaming)...")
    for d in tqdm(scan_dirs, desc="add-scan"):
        X_mmap, keys, sid = load_one_scan(d)
        keys2 = [f"{sid}|{k}" for k in keys] 
        n = int(X_mmap.shape[0])
        if n == 0:
            continue

        # add in chunks to avoid excessive memory usage at once
        start = 0
        while start < n:
            end = min(n, start + ADD_CHUNK)
            X = np.array(X_mmap[start:end])              # load chunk
            X = X.astype(np.float32, copy=False)
            X = l2_normalize(X)
            index.add(X)
            start = end

        all_keys.extend(keys2)
        added += n
        if added == n:  # first scan
            log(f"Sample key after prefix: {keys2[0]}")

        # Critical consistency check: index.ntotal must stay in sync with keys
        if index.ntotal != len(all_keys):
            log(f"[FATAL] ntotal mismatch after add scan {d}: ntotal={index.ntotal}, keys={len(all_keys)}")
            sys.exit(1)

    log(f"Add done. index.ntotal={index.ntotal}, keys={len(all_keys)}, estimated_total={total_vectors_est}")

    if index.ntotal == 0:
        log("[FATAL] index.ntotal is 0, nothing added.")
        sys.exit(1)

    # 4) Set nprobe and save
    index.nprobe = NPROBE

    log(f"Saving FAISS index -> {OUT_INDEX_PATH}")
    faiss.write_index(index, OUT_INDEX_PATH)

    log(f"Saving keys -> {OUT_KEYS_PATH}")
    with open(OUT_KEYS_PATH, "wb") as f:
        pickle.dump(all_keys, f)

    # 5) Read back after saving for a minimal sanity check (guards against corrupt writes)
    log("Verifying saved index...")
    idx2 = faiss.read_index(OUT_INDEX_PATH)
    if idx2.ntotal != len(all_keys):
        log(f"[FATAL] Saved index ntotal mismatch: {idx2.ntotal} vs keys {len(all_keys)}")
        sys.exit(1)

    log("Quick search sanity check...")
    # Take one vector for self-check: the nearest neighbor should have a very high similarity
    q = np.array(np.load(os.path.join(scan_dirs[0], "frame_patch_embeddings.npy"), mmap_mode="r")[0:1]).astype(np.float32)
    q = l2_normalize(q)
    Dists, Ids = idx2.search(q, 5)
    log(f"Top5 IDs: {Ids[0].tolist()}")
    log(f"Top5 sims: {Dists[0].tolist()}")

    log("==== Build FAISS finished OK ====")

if __name__ == "__main__":
    main()
