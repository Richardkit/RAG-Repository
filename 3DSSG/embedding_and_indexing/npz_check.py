import numpy as np
import pickle

# === path ===
npy_path = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan_embeddings/all_vecs.npy"
keys_path = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan_embeddings/all_keys.pkl"

# === load keys ===
with open(keys_path, "rb") as f:
    rgb_keys = pickle.load(f)
N = len(rgb_keys)
print(f"🔍 Loaded {N} keys")

# === 使用 memmap 读取 ===
try:
    rgb_vectors = np.memmap(npy_path, dtype=np.float32, mode="r", shape=(N, 768))
    print(f"✅ Loaded .npy file with shape: {rgb_vectors.shape}")
except Exception as e:
    print(f"❌ Failed to load .npy: {e}")
    exit(1)

# === 检查维度正确性 ===
if rgb_vectors.shape != (N, 768):
    print(f"❌ Shape mismatch: got {rgb_vectors.shape}, expected ({N}, 768)")
else:
    print("✅ Shape correct")

# === 检查是否存在 NaN / Inf ===
subset = rgb_vectors[:1000]  # 检查前 1000 个
if np.isnan(subset).any():
    print("❌ Detected NaN values")
elif np.isinf(subset).any():
    print("❌ Detected Inf values")
else:
    print("✅ No NaN or Inf in sample")

# === 打印一条向量看看 ===
print("Sample vector:")
print(rgb_vectors[0])
print("Corresponding key:")
print(rgb_keys[0])
