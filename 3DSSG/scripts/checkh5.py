import h5py

REL_H5 = "data/3RScan_ScanNet20_InSeg/relationships.h5"
PROP_H5 = "data/3RScan_ScanNet20_InSeg/proposals.h5"

with h5py.File(REL_H5, "r") as f:
    print("📁 relationships.h5 keys:", list(f.keys()))

with h5py.File(PROP_H5, "r") as f:
    print("📁 proposals.h5 keys:", list(f.keys()))
