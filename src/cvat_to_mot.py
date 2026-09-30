#!/usr/bin/env python3
"""
Convert CVAT XML annotations to MOT ground truth format.

CVAT XML structure:
    <track id="5" label="player">
      <box frame="0" xtl="100" ytl="200" xbr="150" ybr="300" outside="0" occluded="0"/>
      <box frame="1" ... outside="1"/>  <!-- outside=1 → object not in frame, skip -->
    </track>

Output directory layout (MOT-compatible):
    <output>/
        gt.txt        ← one line per visible box
        seqinfo.ini   ← sequence metadata (fps, resolution, length)

MOT line format:
    frame_id, track_id, x, y, w, h, conf, class_id, visibility
    - frame_id: 1-indexed
    - conf: always 1 for ground truth
    - visibility: 1.0 if not occluded, 0.5 if occluded

Class mapping (CVAT label → integer):
    player   → 0
    referee  → 1
    ball     → 2

Usage:
    python src/cvat_to_mot.py \\
        --annotations /cluster/.../RBK-AALESUND/annotations.xml

    # Output goes to data/tracking/gt/rbk-aalesund/ automatically.
    # Override root with --output-dir if needed.
"""

import argparse
import configparser
import xml.etree.ElementTree as ET
from pathlib import Path


LABEL_MAP = {
    "player": 0,
    "ball": 1,
    # referee is annotated as player in this dataset
}


def parse_cvat_xml(xml_path: Path) -> tuple[list[tuple], int]:
    """
    Parse CVAT XML and return (rows, max_frame).

    Each row is: (frame_id, track_id, x, y, w, h, conf, class_id, visibility)
    frame_id is 1-indexed (MOT convention).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    max_frame = 0

    for track in root.findall(".//track"):
        track_id = int(track.get("id"))
        label = track.get("label", "player").lower()
        class_id = LABEL_MAP.get(label, 0)

        for box in track.findall("box"):
            outside = int(box.get("outside", "0"))
            if outside == 1:
                continue  # object not visible in this frame

            frame = int(box.get("frame"))
            occluded = int(box.get("occluded", "0"))

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            x = xtl
            y = ytl
            w = xbr - xtl
            h = ybr - ytl

            visibility = 0.5 if occluded else 1.0
            frame_id = frame + 1  # MOT is 1-indexed

            rows.append((frame_id, track_id, x, y, w, h, 1, class_id, visibility))
            max_frame = max(max_frame, frame_id)

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, max_frame


def write_gt(rows: list[tuple], output_dir: Path) -> None:
    (output_dir / "gt").mkdir(parents=True, exist_ok=True)
    gt_path = output_dir / "gt" / "gt.txt"
    with open(gt_path, "w") as f:
        for row in rows:
            frame_id, track_id, x, y, w, h, conf, class_id, visibility = row
            f.write(
                f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f}"
                f",{conf},{class_id},{visibility:.1f}\n"
            )
    print(f"Wrote {gt_path} ({len(rows)} rows)")


def write_seqinfo(output_dir: Path, seq_name: str, fps: int,
                  width: int, height: int, seq_length: int) -> None:
    ini_path = output_dir / "seqinfo.ini"
    config = configparser.ConfigParser()
    config["Sequence"] = {
        "name": seq_name,
        "imDir": "img1",
        "frameRate": str(fps),
        "seqLength": str(seq_length),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".jpg",
    }
    with open(ini_path, "w") as f:
        config.write(f)
    print(f"Wrote {ini_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CVAT XML to MOT ground truth")
    parser.add_argument("--annotations", required=True, help="Path to CVAT annotations.xml")
    parser.add_argument("--output-dir", default="data/tracking/gt",
                        help="Root output directory (default: data/tracking/gt). "
                             "A subdirectory named after the game folder is created inside.")
    parser.add_argument("--fps", type=int, default=25, help="Video frame rate (default: 25)")
    parser.add_argument("--width", type=int, default=1920, help="Frame width in pixels (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Frame height in pixels (default: 1080)")
    args = parser.parse_args()

    xml_path = Path(args.annotations).resolve()

    if not xml_path.exists():
        raise SystemExit(f"Annotations not found: {xml_path}")

    seq_name = xml_path.parent.name.lower()
    output_dir = (Path(args.output_dir) / seq_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {xml_path} ...")
    rows, max_frame = parse_cvat_xml(xml_path)
    print(f"  {len(rows)} visible boxes across {max_frame} frames")

    write_gt(rows, output_dir)
    write_seqinfo(output_dir, seq_name, args.fps, args.width, args.height, max_frame)


if __name__ == "__main__":
    main()
