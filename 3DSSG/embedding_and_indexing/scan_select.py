# Load the validation set, sample 20% at random (deterministic), and ensure the given ID is included.
import os, math, random, pathlib

src_path = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
dst_path = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_validation_subset_20pct.txt"
must_keep = "09582205-e2c2-2de1-9475-1cdac7639e60"

# Read source IDs
with open(src_path, "r") as f:
    ids = [line.strip() for line in f if line.strip()]

total = len(ids)
k = max(1, math.ceil(total * 0.20))

# Deterministic sampling
random.seed(42)
sampled = set(random.sample(ids, k if k < total else total))

# Ensure must_keep is included
if must_keep in ids:
    sampled.add(must_keep)
else:
    # If not present in src, still add it and note this condition later
    sampled.add(must_keep)

# Write to file
with open(dst_path, "w") as f:
    f.write("\n".join(sorted(sampled)) + "\n")
print(f"Written {len(sampled)} IDs to {dst_path}.")
