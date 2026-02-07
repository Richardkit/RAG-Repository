import faiss
import numpy as np
import pickle

# === 加载数据 ===
with open("/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset_siglip2/all_keys.pkl", "rb") as f:
    rgb_keys = pickle.load(f)

rgb_vectors = np.memmap(
    "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset_siglip2/all_vecs.npy",
    dtype=np.float32,
    mode='r',
    shape=(len(rgb_keys), 768)
)

# === 构建 IndexIVFFlat ===
d = 768
nlist = 529  # coarse centroid 数量
quantizer = faiss.IndexFlatL2(d)  # coarse quantizer

index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# === 训练 ===
print("🔧 Training IVFFlat index...")
index.train(rgb_vectors[:50000])  # 训练阶段只需部分数据
print("✅ Training done.")

# === 添加向量 ===
index.add(rgb_vectors)
print(f"✅ Indexed {index.ntotal} vectors.")

# === 保存 ===
faiss.write_index(index, "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset_siglip2/rgb_index.ivfflat")
