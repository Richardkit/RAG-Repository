# building_IVF_index_patch.py (example: build IVF index)
import os, pickle
import numpy as np
import faiss
from tqdm import tqdm

# ========= Paths to modify =========
IN_DIR  = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_patch_embeddings_without_validset_siglip2_global"
OUT_DIR = IN_DIR

X_NPY   = os.path.join(IN_DIR, "all_patch_embs.npy")
KEYS_PKL= os.path.join(IN_DIR, "all_patch_keys.pkl")

OUT_INDEX = os.path.join(OUT_DIR, "patch_index.ivfflat")
# If you want to keep a flat backup as well:
OUT_INDEX_FLAT = os.path.join(OUT_DIR, "patch_index.flatip")

D = 768

# IVF parameters: adjust according to data size
NLIST = 4096     # number of cluster centroids (commonly 1k~16k)
NPROBE = 16      # number of inverted lists to probe at query time
TRAIN_MAX = 300000  # maximum number of patches to sample for training

def l2_normalize(x: np.ndarray, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def main():
    X = np.load(X_NPY)  # (N,D) float32/float16
    if X.ndim != 2 or X.shape[1] != D:
        raise RuntimeError(f"X shape is incorrect: {X.shape}")

    # FAISS recommends float32
    X = X.astype(np.float32)

    # Use IP for cosine similarity: L2 normalize first
    X = l2_normalize(X)

    N = X.shape[0]
    print("[INFO] N =", N, "D =", D)

    # -------- Optional: build a FlatIP first (most stable, but slow for large libraries) ---------
    # flat = faiss.IndexFlatIP(D)
    # flat.add(X)
    # faiss.write_index(flat, OUT_INDEX_FLAT)
    # print("[SAVE]", OUT_INDEX_FLAT)

    # -------- IVF Flat (IP) ----------
    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFFlat(quantizer, D, NLIST, faiss.METRIC_INNER_PRODUCT)

    # train
    if not index.is_trained:
        if N > TRAIN_MAX:
            rnd = np.random.RandomState(0)
            idx = rnd.choice(N, size=TRAIN_MAX, replace=False)
            X_train = X[idx]
        else:
            X_train = X
        print("[TRAIN] X_train =", X_train.shape)
        index.train(X_train)

    # add
    print("[ADD] ...")
    index.add(X)

    # set nprobe (can also be set again at query time)
    index.nprobe = NPROBE

    faiss.write_index(index, OUT_INDEX)
    print("[SAVE]", OUT_INDEX)

    # the keys file is only referenced here to verify that the lengths match
    with open(KEYS_PKL, "rb") as f:
        keys = pickle.load(f)
    assert len(keys) == N, f"keys({len(keys)}) != N({N})"
    print("[OK] keys length matched.")

if __name__ == "__main__":
    main()
