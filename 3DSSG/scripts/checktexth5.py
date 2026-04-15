import h5py

PROP_H5 = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data/3RScan_ScanNet20_InSeg/relationships.h5"
SCAN_ID = "02b33df9-be2b-2d54-9062-1253be3ce186"

with h5py.File(PROP_H5, 'r') as f:
    node_keys = list(f[SCAN_ID]["nodes"].keys())
    print("📦 Keys under ['nodes']:", node_keys)
    print("📦 The first element of nodes", f[SCAN_ID]['nodes']["0"])
