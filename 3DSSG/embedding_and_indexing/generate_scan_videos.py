#!/usr/bin/env python3
"""Generate MP4 videos for 3RScan sequences listed in a validation file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create videos for 3RScan sequences.")
    parser.add_argument(
        "validation_file",
        type=Path,
        help="Path to the text file containing scan IDs (one per line).",
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Path to the data/3RScan directory containing scan folders.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame rate for the output videos (default: 10).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild videos even when the output file already exists.",
    )
    return parser.parse_args()


def read_scan_ids(validation_file: Path) -> List[str]:
    if not validation_file.is_file():
        raise FileNotFoundError(f"Validation file not found: {validation_file}")

    scan_ids: List[str] = []
    for raw_line in validation_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        scan_ids.append(line)
    if not scan_ids:
        raise ValueError(f"No scan IDs found in {validation_file}")
    return scan_ids


def natural_sort_key(path: Path) -> Iterable:
    stem = path.stem
    token = ''.join(ch if ch.isdigit() else ' ' for ch in stem)
    numeric_parts = [int(part) for part in token.split() if part]
    return (*numeric_parts, path.name)


def gather_frames(sequence_dir: Path) -> List[Path]:
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory missing: {sequence_dir}")

    frames = sorted(sequence_dir.glob("*.jpg"), key=natural_sort_key)
    if not frames:
        raise FileNotFoundError(f"No JPG frames found in {sequence_dir}")
    return frames


def init_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    return writer


def write_video(frames: List[Path], output_path: Path, fps: float) -> None:
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")

    height, width = first_frame.shape[:2]
    writer = init_writer(output_path, fps, (width, height))

    try:
        writer.write(first_frame)
        for frame_path in frames[1:]:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()



def build_video_for_scan(scan_id: str, data_root: Path, fps: float, force: bool) -> None:
    scan_dir = data_root / scan_id
    sequence_dir = scan_dir / "sequence"
    output_path = scan_dir / f"{scan_id}.mp4"

    if output_path.exists() and not force:
        print(f"[skip] {scan_id}: video already exists at {output_path}")
        return

    frames = gather_frames(sequence_dir)
    print(f"[info] {scan_id}: writing {len(frames)} frames to {output_path}")
    write_video(frames, output_path, fps)
    print(f"[done] {scan_id}: saved video to {output_path}")


def main() -> int:
    args = parse_args()
    try:
        scan_ids = read_scan_ids(args.validation_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1

    failures = 0
    for scan_id in scan_ids:
        try:
            build_video_for_scan(scan_id, args.data_root, args.fps, args.force)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[fail] {scan_id}: {exc}")

    if failures:
        print(f"[warn] Completed with {failures} failure(s).")
        return 1

    print("[ok] All videos generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
