import os
import random

data_dir = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan"
output_txt = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
sample_ratio = 0.1  # 10%

scan_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

sample_size = max(1, int(len(scan_ids) * sample_ratio))
test_scan_ids = random.sample(scan_ids, sample_size)

with open(output_txt,"w") as f:
    for scan_id in test_scan_ids:
        f.write(scan_id + "\n")

print(f"Sampled {sample_size} scan_ids for the test set, saved to {output_txt}")