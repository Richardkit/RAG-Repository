#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import Any, List

# =======================
# 配置：你只需要确认这三个
# =======================
VALID_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/3DSSG/embedding_and_indexing/3RScan_validation_set.txt"
DATA_ROOT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/3RScan"

OLD = "rg5174-ragspace2"
NEW = "rg5174-ragspace3"

MAKE_BACKUP = True
DRY_RUN = False
# =======================


def read_scan_ids(txt_path: str) -> List[str]:
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(txt_path)
    out = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s:
                out.append(s)
    return out


def replace_in_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return obj.replace(OLD, NEW)
    if isinstance(obj, list):
        return [replace_in_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: replace_in_obj(v) for k, v in obj.items()}
    return obj


def atomic_write_json(path: str, data: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    scan_ids = read_scan_ids(VALID_TXT)
    print(f"[Load] valid scans = {len(scan_ids)} from {VALID_TXT}")

    changed_files = 0
    changed_strings_total = 0
    missing_paths = 0
    missing_siglip2_val = 0
    load_failed = 0

    for sid in scan_ids:
        scan_dir = os.path.join(DATA_ROOT, sid)
        siglip2_val_dir = os.path.join(scan_dir, "siglip2_val")
        paths_json = os.path.join(siglip2_val_dir, "paths.json")

        if not os.path.isdir(scan_dir):
            print(f"[Miss] scan_dir not found: {scan_dir}")
            missing_paths += 1
            continue
        if not os.path.isdir(siglip2_val_dir):
            print(f"[Miss] siglip2_val not found: {siglip2_val_dir}")
            missing_siglip2_val += 1
            continue
        if not os.path.isfile(paths_json):
            print(f"[Miss] paths.json not found: {paths_json}")
            missing_paths += 1
            continue

        try:
            with open(paths_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[Skip] load failed: {paths_json} err={e}")
            load_failed += 1
            continue

        before = json.dumps(obj, ensure_ascii=False)
        n_before = before.count(OLD)
        if n_before == 0:
            continue

        obj2 = replace_in_obj(obj)
        after = json.dumps(obj2, ensure_ascii=False)
        n_after = after.count(OLD)
        replaced = n_before - n_after

        if replaced <= 0:
            continue

        print(f"[Fix] {sid} replaced={replaced} file={paths_json}")
        changed_files += 1
        changed_strings_total += replaced

        if DRY_RUN:
            continue

        if MAKE_BACKUP and not os.path.exists(paths_json + ".bak"):
            try:
                with open(paths_json + ".bak", "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Warn] backup failed: {paths_json}.bak err={e}")

        try:
            atomic_write_json(paths_json, obj2)
        except Exception as e:
            print(f"[Err] write failed: {paths_json} err={e}")

    print(
        f"[Done] changed_files={changed_files} changed_strings_total={changed_strings_total} "
        f"missing_paths_json={missing_paths} missing_siglip2_val={missing_siglip2_val} load_failed={load_failed}"
    )
    if DRY_RUN:
        print("[Done] DRY_RUN=True, no files were written.")


if __name__ == "__main__":
    main()
