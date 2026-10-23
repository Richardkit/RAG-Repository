import h5py

REL_H5 = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data/3RScan_ScanNet20_InSeg/relationships.h5"
SCAN_ID = "f62fd5fd-9a3f-2f44-883a-1e5cf819608e"

with h5py.File(REL_H5, 'r') as f:
    rel_bytes = f[SCAN_ID][0]  # extract the single string
    rel_str = rel_bytes.decode("utf-8")
    print("📄 Raw string content:")
    print(rel_str)
