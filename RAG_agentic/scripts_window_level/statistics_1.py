import os
import re
import glob
import ast
from collections import defaultdict

ROOT = "/hkfs/work/workspace/scratch/rg5174-ragspace3/data/RAG_Qwen3/RAG_end2end_CoT_agentic_0_1"

# 只匹配你关心的两种
RE_KEPT = re.compile(r"kept_query_imgs\s*=\s*(\[[^\]]*\])", re.IGNORECASE)
RE_SKIPPED = re.compile(r"\[JUDGE\]\s*skipped.*?query_imgs\s*=\s*(\[[^\]]*\])", re.IGNORECASE)

def safe_parse_list(list_text: str):
    """
    list_text looks like: ['path1', 'path2', ...]
    """
    try:
        x = ast.literal_eval(list_text)
        if isinstance(x, list):
            return [str(i) for i in x]
    except Exception:
        pass
    return None

def main():
    log_paths = sorted(glob.glob(os.path.join(ROOT, "**", "*.log"), recursive=True))
    if not log_paths:
        print(f"[ERR] No .log found under {ROOT}")
        return

    # per scan
    scan_unique = defaultdict(set)     # scan_id -> set(frame_path)
    scan_total = defaultdict(int)      # scan_id -> total occurrences (no dedup)
    scan_hits = defaultdict(int)       # scan_id -> number of matched lines
    no_match_logs = []

    for lp in log_paths:
        scan_id = os.path.basename(os.path.dirname(lp))  # folder name is scan id (UUID)
        matched_any = False

        with open(lp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = RE_KEPT.search(line)
                if m:
                    lst = safe_parse_list(m.group(1))
                    if lst is not None:
                        matched_any = True
                        scan_hits[scan_id] += 1
                        scan_total[scan_id] += len(lst)
                        scan_unique[scan_id].update(lst)
                    continue

                m = RE_SKIPPED.search(line)
                if m:
                    lst = safe_parse_list(m.group(1))
                    if lst is not None:
                        matched_any = True
                        scan_hits[scan_id] += 1
                        scan_total[scan_id] += len(lst)
                        scan_unique[scan_id].update(lst)
                    continue

        if not matched_any:
            no_match_logs.append(lp)

    # 输出
    print("=== Per-scan kept frames (ColQwen kept_query_imgs + JUDGE skipped query_imgs) ===")
    total_unique_sum = 0
    total_occ_sum = 0

    for sid in sorted(scan_unique.keys()):
        uniq_n = len(scan_unique[sid])
        occ_n = scan_total[sid]
        hits = scan_hits[sid]
        total_unique_sum += uniq_n
        total_occ_sum += occ_n
        print(f"{sid}\tunique={uniq_n}\tocc={occ_n}\thit_lines={hits}")

    print("\n=== TOTAL ===")
    print(f"TOTAL_UNIQUE_SUM={total_unique_sum}")
    print(f"TOTAL_OCC_SUM={total_occ_sum}")

    if no_match_logs:
        print("\n=== NO_MATCH_LOGS (no kept_query_imgs / no [JUDGE] skipped query_imgs) ===")
        for lp in no_match_logs[:50]:
            print(lp)
        if len(no_match_logs) > 50:
            print(f"... ({len(no_match_logs)-50} more)")

if __name__ == "__main__":
    main()
