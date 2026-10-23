import h5py

REL_H5 = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data/3RScan_ScanNet20_InSeg/relationships.h5"

def print_structure(name, obj):
    indent = '  ' * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}📂 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 Dataset: {name} | shape: {obj.shape} | dtype: {obj.dtype}")

with h5py.File(REL_H5, 'r') as f:
    print("🔍 HDF5 file structure:")
    f.visititems(print_structure)
