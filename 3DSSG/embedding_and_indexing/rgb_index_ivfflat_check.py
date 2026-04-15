import faiss
import pickle

index_path = "/lsdf/users/zwang/Embedding_Combination/rgb_index.ivfflat"
keys_path = "/lsdf/users/zwang/Embedding_Combination/rgb_keys.pkl"

# === Load index ===
try:
    index = faiss.read_index(index_path)
    print("✅ Successfully loaded FAISS index.")
except Exception as e:
    print(f"❌ Failed to load index: {e}")
    exit(1)

# === Print basic index information ===
print(f"📦 Index type: {type(index)}")
print(f"🧠 Dimension: {index.d}")
print(f"📈 Number of vectors: {index.ntotal}")
print(f"🧭 IVF nlist: {index.nlist}")
print(f"🔍 Current nprobe: {index.nprobe}")

# === Load keys and check whether the count matches ===
with open(keys_path, "rb") as f:
    keys = pickle.load(f)

if len(keys) != index.ntotal:
    print(f"❌ Mismatch: index has {index.ntotal} vectors, but {len(keys)} keys found.")
else:
    print("✅ Key count matches vector count.")
