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

# === Read using memmap ===
try:
    rgb_vectors = np.memmap(npy_path, dtype=np.float32, mode="r", shape=(N, 768))
    print(f"✅ Loaded .npy file with shape: {rgb_vectors.shape}")
except Exception as e:
    print(f"❌ Failed to load .npy: {e}")
    exit(1)

# === Check dimension correctness ===
if rgb_vectors.shape != (N, 768):
    print(f"❌ Shape mismatch: got {rgb_vectors.shape}, expected ({N}, 768)")
else:
    print("✅ Shape correct")

# === Check for NaN / Inf values ===
subset = rgb_vectors[:1000]  # check the first 1000 entries
if np.isnan(subset).any():
    print("❌ Detected NaN values")
elif np.isinf(subset).any():
    print("❌ Detected Inf values")
else:
    print("✅ No NaN or Inf in sample")

# === Print one vector as a sanity check ===
print("Sample vector:")
print(rgb_vectors[0])
print("Corresponding key:")
print(rgb_keys[0])
