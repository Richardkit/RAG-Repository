import faiss
import pickle

index_path = "/lsdf/users/zwang/Embedding_Combination/rgb_index.ivfflat"
keys_path = "/lsdf/users/zwang/Embedding_Combination/rgb_keys.pkl"

# === 加载 index ===
try:
    index = faiss.read_index(index_path)
    print("✅ Successfully loaded FAISS index.")
except Exception as e:
    print(f"❌ Failed to load index: {e}")
    exit(1)

# === 打印 index 基本信息 ===
print(f"📦 Index type: {type(index)}")
print(f"🧠 Dimension: {index.d}")
print(f"📈 Number of vectors: {index.ntotal}")
print(f"🧭 IVF nlist: {index.nlist}")
print(f"🔍 Current nprobe: {index.nprobe}")

# === 加载 key 并检查数量是否一致 ===
with open(keys_path, "rb") as f:
    keys = pickle.load(f)

if len(keys) != index.ntotal:
    print(f"❌ Mismatch: index has {index.ntotal} vectors, but {len(keys)} keys found.")
else:
    print("✅ Key count matches vector count.")
