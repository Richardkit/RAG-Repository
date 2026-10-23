import faiss
import numpy as np
import pickle

# === Load data ===
with open("/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset_siglip2/all_keys.pkl", "rb") as f:
    rgb_keys = pickle.load(f)

rgb_vectors = np.memmap(
    "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset_siglip2/all_vecs.npy",
    dtype=np.float32,
    mode='r',
    shape=(len(rgb_keys), 768)
)

# === Build IndexIVFFlat ===
d = 768
nlist = 529  # number of coarse centroids
quantizer = faiss.IndexFlatL2(d)  # coarse quantizer

index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# === Train ===
print("🔧 Training IVFFlat index...")
index.train(rgb_vectors[:50000])  # only a subset of data is needed during the training phase
print("✅ Training done.")

# === Add vectors ===
index.add(rgb_vectors)
print(f"✅ Indexed {index.ntotal} vectors.")

# === Save ===
faiss.write_index(index, "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan_embeddings_without_validset_siglip2/rgb_index.ivfflat")
