"""
this code is used to find out if the name of the dataset from a roi_image folder of a certain scan id, z.B.
/hkfs/work/workspace/scratch/rg5174-myspace/data/data_extra/roi_images.h5/roi_images/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca.h5
is a object id or not.

partially output of it as following:
  [Dataset] 1 shape=(17, 3, 256, 256) dtype=float32
  [Dataset] 101 shape=(2, 3, 256, 256) dtype=float32
  [Dataset] 114 shape=(6, 3, 256, 256) dtype=float32
  [Dataset] 12 shape=(4, 3, 256, 256) dtype=float32
  [Dataset] 129 shape=(1, 3, 256, 256) dtype=float32
  [Dataset] 130 shape=(3, 3, 256, 256) dtype=float32
  [Dataset] 131 shape=(1, 3, 256, 256) dtype=float32
  [Dataset] 133 shape=(1, 3, 256, 256) dtype=float32
  [Dataset] 134 shape=(3, 3, 256, 256) dtype=float32
  ......

  I guess 1, 101, 114 and so on are object id in the scan id, and the first dimension of the shape is the number of frames, where the object is visible.
"""


import os, glob, h5py, json
from ssg import define  # if not available, manually write out the two format strings below

scan_id = "0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca"
oid = "1"
root = "/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan"# cfg.data.path_3rscan_data
roi_link_h5 = "/hkfs/work/workspace/scratch/rg5174-myspace/data/data_dense/roi_img/roi_images.h5" # cfg.data.roi_img_path/roi_images.h5
proposals_h5 = "/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/data/3RScan_ScanNet20_InSeg/proposals.h5" # can be obtained from the visibility map generation stage
seq_dir = os.path.join(root, scan_id, "sequence")

# 1) Total number of RGB frames in the Sequence folder
rgb_frames = sorted(glob.glob(os.path.join(seq_dir, "frame-*.color.jpg")))
print("Total RGB frames:", len(rgb_frames))

# 2) Number of frames in which this object “appears” in proposals (has a bbox)
with h5py.File(proposals_h5, "r") as fp:
    kfs = fp[scan_id]["kfs"]
    count_has_bbox = 0
    for k in kfs.keys():               # each key is a frame id (string)
        seg2idx = dict(kfs[k].attrs["seg2idx"])
        if int(oid) in seg2idx:        # this frame contains the bbox of the object
            count_has_bbox += 1
print("Frames containing the object's bbox:", count_has_bbox)

# 3) Actual number of ROI images written (after filtering such as viewpoint deduplication)
with h5py.File(roi_link_h5, "r") as link:
    # It is an ExternalLink to a sub h5: roi_images/<scan_id>.h5
    with h5py.File(link[scan_id].file.filename, "r") as roi:
        n_roi = roi[oid].shape[0]      # the first dimension is the number of ROIs for this object
        print("ROI count (final retained):", n_roi)

"""
the former code is to ensure the quantitative relationship of frames for one scan id, frames that contain the chosen object's bbx 
and frames the finally are contained, so the final result is: total RGB frames >= frames containing the object's bbox >= ROI count (final retained)
"""