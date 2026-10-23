import h5py
import ast
import tqdm

# Path to your relationships.h5 file
h5_path = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data_GT/3RScan_3RScan160/relationships.h5"

# Choose a scan_id to inspect (we'll just take the first one)
with h5py.File(h5_path, "r") as f:
    scan_ids = list(f.keys())
    print(len(scan_ids), "scan_ids found.")
    second_scan = scan_ids[1]
    print(f"Available scan_ids: {scan_ids[:5]} ...")
    print(f"Inspecting scan_id: {second_scan}")

    second_scan = "0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca"
    # Read and parse the stored dict
    raw_data = f[second_scan][0].decode()
    data = ast.literal_eval(raw_data)

# for i in range(len(scan_ids)):

#     with h5py.File(h5_path, "r") as f:
#         scan_ids = list(f.keys())   
#         scan_to_test = scan_ids[i]

#         # Read and parse the stored dict
#         raw_data = f[second_scan][0].decode()
#         data = ast.literal_eval(raw_data)
#         nodes = data.get("nodes", {})
#         for i, (oid, meta) in enumerate(nodes.items()):
#         # print(f"oid={oid}:", meta)

#             if oid != meta.get("instance_id"):
#                 print(f"Warning: oid {oid} does not match instance_id {meta.get('instance_id')} of {scan_to_test}")

#     i += 1


# Show the structure
data_keys = list(data.keys())
print(f"Top-level keys: {data_keys}")

# Print nodes info (first few)
nodes = data.get("nodes", {})
print("\n=== NODES ===")
for i, (oid, meta) in enumerate(nodes.items()):
    print(f"oid={oid}:", meta)

    # if oid != meta.get("instance_id"):
    #     print(f"Warning: oid {oid} does not match instance_id {meta.get('instance_id')}")
    #     # there is no difference between oid and instance id, it is a error during our early coding

    if i >= 5:
        break

# Print relationships info (first few)
rels = data.get("relationships", [])
print("\n=== RELATIONSHIPS ===")
for r in rels[:5]:
    print(r)                     