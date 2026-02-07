cd /hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing

python - <<'PY'
from pathlib import Path
root = Path("/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan")
scan_ids = sorted([p.name for p in root.iterdir() if p.is_dir()])
Path("scans.txt").write_text("\n".join(scan_ids), encoding="utf-8")
print("✅ wrote scans.txt:", len(scan_ids))
PY
