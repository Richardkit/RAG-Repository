import os
import sys
import time
import glob
import pickle
import numpy as np
from tqdm import tqdm

# ---- faiss import：有的环境是 faiss，有的是 faiss-cpu ----
try:
    import faiss  # type: ignore
except Exception as e:
    print("[FATAL] Cannot import faiss. Make sure faiss is installed in this env.")
    raise

# =========================
# 你只需要改这里的路径
# =========================

# 如果你实际是 .../frame_embeddings/<scan_id>/，就把 PATCH_ROOT 指到 frame_embeddings
PATCH_ROOT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_patch_embeddings_without_validset_siglip2/frame_embeddings"

# 输出目录（索引与全局 keys 保存到这里）
OUT_DIR = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan_patch_embeddings_without_validset_siglip2"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_INDEX_PATH = os.path.join(OUT_DIR, "patch_index_50.ivfflat")
OUT_KEYS_PATH  = os.path.join(OUT_DIR, "all_patch_keys_50.pkl")
OUT_LOG_PATH   = os.path.join(OUT_DIR, "build_faiss_patch_50.log")

# =========================
# FAISS / 向量参数
# =========================
D = 768
METRIC = "cosine"   # cosine via inner product + L2 normalize
INDEX_TYPE = "ivfflat"  # 你要的就是它

KB_FRACTION = 0.5
KB_MODE = "random"   # or "top"
KB_SEED = 0

# IVF 参数（你可以不改；但如果库很大可加大 NLIST）
NLIST = 4096
NPROBE = 16

# 训练样本上限（太大训练慢，太小训练不稳）
TRAIN_MAX = 100_000
# 每个 scan 最多采多少条进训练池（防止某个 scan 特别大把训练池挤爆）
TRAIN_PER_SCAN_MAX = 4000

# add 的分块大小（避免一次 add 太大占内存）
ADD_CHUNK = 200_000

# 允许 key 的快速检查：必须是字符串
STRICT_KEY_TYPE = True

# =========================
# 工具函数
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
    找到所有包含 frame_patch_embeddings.npy + patch_keys.pkl 的目录
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

    # mmap 读取，省内存
    X = np.load(npy_path, mmap_mode="r")  # may be float16/float32
    with open(pkl_path, "rb") as f:
        keys = pickle.load(f)

    # ---- 硬检查 ----
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
        # 只检查前后几个，避免 O(N) 很慢
        for idx in [0, min(1, len(keys)-1), max(0, len(keys)-2), len(keys)-1]:
            if len(keys) == 0:
                break
            if not isinstance(keys[idx], str):
                raise RuntimeError(f"Key is not str in {pkl_path}, keys[{idx}]={type(keys[idx])}")
            
    import re
    UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

    def infer_scan_id(scan_dir: str) -> str:
        cur = os.path.abspath(scan_dir)
        for _ in range(4):  # 向上最多找4层，足够覆盖 .../<uuid>/sequence/...
            base = os.path.basename(cur)
            if UUID_RE.match(base):
                return base.lower()
            cur = os.path.dirname(cur)
        # fallback：至少返回原basename（但会触发后面的检查）
        return os.path.basename(scan_dir)

    # load_one_scan() 里用：
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
# 主流程
# =========================
def main():
    # 清理/新建 log
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

    # 1) 收集训练样本
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

        # 采样（大于 take 就随机，否则全取）
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
    # 最终硬检查
    if X_train.ndim != 2 or X_train.shape[1] != D:
        log(f"[FATAL] Bad X_train shape: {X_train.shape}")
        sys.exit(1)

    log(f"Training samples: {X_train.shape}")

    # 2) 建 index + train
    index = make_index()
    log("Training FAISS index...")
    t0 = time.time()
    index.train(X_train)
    t1 = time.time()
    log(f"FAISS train done in {t1 - t0:.1f}s. is_trained={index.is_trained}")

    if not index.is_trained:
        log("[FATAL] Index training failed (index.is_trained=False).")
        sys.exit(1)

    # 3) 逐 scan add（流式，不拼全局大矩阵）
    all_keys = []
    added = 0

    log("Adding vectors to FAISS index (streaming)...")
    for d in tqdm(scan_dirs, desc="add-scan"):
        X_mmap, keys, sid = load_one_scan(d)
        keys2 = [f"{sid}|{k}" for k in keys] 
        n = int(X_mmap.shape[0])
        if n == 0:
            continue

        # 分块 add，避免一次性占太多内存
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
        if added == n:  # 第一个scan
            log(f"Sample key after prefix: {keys2[0]}")

        # 关键一致性检查：index.ntotal 必须与 keys 同步
        if index.ntotal != len(all_keys):
            log(f"[FATAL] ntotal mismatch after add scan {d}: ntotal={index.ntotal}, keys={len(all_keys)}")
            sys.exit(1)

    log(f"Add done. index.ntotal={index.ntotal}, keys={len(all_keys)}, estimated_total={total_vectors_est}")

    if index.ntotal == 0:
        log("[FATAL] index.ntotal is 0, nothing added.")
        sys.exit(1)

    # 4) 设置 nprobe 并保存
    index.nprobe = NPROBE

    log(f"Saving FAISS index -> {OUT_INDEX_PATH}")
    faiss.write_index(index, OUT_INDEX_PATH)

    log(f"Saving keys -> {OUT_KEYS_PATH}")
    with open(OUT_KEYS_PATH, "wb") as f:
        pickle.dump(all_keys, f)

    # 5) 保存后再读回做一次最小验证（防止写坏）
    log("Verifying saved index...")
    idx2 = faiss.read_index(OUT_INDEX_PATH)
    if idx2.ntotal != len(all_keys):
        log(f"[FATAL] Saved index ntotal mismatch: {idx2.ntotal} vs keys {len(all_keys)}")
        sys.exit(1)

    log("Quick search sanity check...")
    # 取一个向量做自检：最近邻应该有很高相似度
    q = np.array(np.load(os.path.join(scan_dirs[0], "frame_patch_embeddings.npy"), mmap_mode="r")[0:1]).astype(np.float32)
    q = l2_normalize(q)
    Dists, Ids = idx2.search(q, 5)
    log(f"Top5 IDs: {Ids[0].tolist()}")
    log(f"Top5 sims: {Dists[0].tolist()}")

    log("==== Build FAISS finished OK ====")

if __name__ == "__main__":
    main()
