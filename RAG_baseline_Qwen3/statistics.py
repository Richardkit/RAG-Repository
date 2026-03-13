"""
how many images are under f"/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan/{scan_id}/sequence", 
scan_id should be in "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
"""
from pathlib import Path
from collections import Counter

VALID_LIST = Path("/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/embedding_and_indexing/3RScan_validation_set.txt")
SEQUENCE_ROOT = Path("/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan")

def main():
    scan_ids = [line.strip() for line in VALID_LIST.read_text().splitlines() if line.strip()]
    counts = {}
    missing = []

    for scan_id in scan_ids:
        seq_dir = SEQUENCE_ROOT / scan_id / "sequence"
        if not seq_dir.is_dir():
            missing.append(scan_id)
            continue
        counts[scan_id] = sum(1 for p in seq_dir.glob("*.color.jpg"))

    for scan_id, num in sorted(counts.items()):
        print(f"{scan_id}: {num}")

    if counts:
        stats = Counter(counts.values())
        total_frames = sum(counts.values())
        avg = total_frames / len(counts)
        print(f"\nscans counted: {len(counts)}")
        print(f"total frames: {total_frames}")
        print(f"min frames: {min(counts.values())}")
        print(f"max frames: {max(counts.values())}")
        print(f"mean frames: {avg:.2f}")
        print("\nframe-count distribution:")
        for frame_count, freq in sorted(stats.items()):
            print(f"{frame_count}: {freq}")

    if missing:
        print("\nmissing sequence dirs:")
        for scan_id in missing:
            print(scan_id)

if __name__ == "__main__":
    main()