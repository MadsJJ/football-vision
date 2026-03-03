#!/usr/bin/env python3
"""
Build a combined data.yaml for the Football2025 dataset.

The dataset has N game folders (e.g. RBK-AALESUND, RBK-BRANN, …),
each structured like:
    <game>/
        data.yaml          # per-game yaml (only has 'train' split)
        train.txt          # lists image paths relative to <game>/
        data/images/train/
        labels/train/

This script assigns whole games to train/val/test splits and writes:
    <output_dir>/football.yaml   # combined yaml for Ultralytics
    <output_dir>/train.txt
    <output_dir>/val.txt
    <output_dir>/test.txt        # omitted if no test games

The dataset root is read-only (shared course data); all output goes to --output-dir.

Usage:
    # On IDUN (dataset is read-only):
    python src/prepare_dataset.py \
        --dataset-root /cluster/projects/vc/courses/TDT17/other/Football2025 \
        --output-dir data/

    # Local sample:
    python src/prepare_dataset.py --dataset-root data/RBK-AALESUND --output-dir data/

Split assignment (4 games): 2 train, 1 val, 1 test
You can override with --train --val --test by game folder name.
"""

import argparse
import yaml
from pathlib import Path


def find_game_folders(dataset_root: Path) -> list[Path]:
    """Return subdirectories that look like game folders (contain a labels/ dir)."""
    games = sorted(
        d for d in dataset_root.iterdir()
        if d.is_dir() and (d / "labels").exists()
    )
    if not games:
        raise RuntimeError(f"No game folders found in {dataset_root}")
    return games


def default_split(games: list[Path]) -> dict[str, list[Path]]:
    """Assign games to splits. With 5 games: 3 train, 1 val, 1 test."""
    n = len(games)
    if n == 1:
        # Local sample: use the single game for all splits
        return {"train": games, "val": games, "test": []}
    if n == 2:
        return {"train": games[:1], "val": games[1:], "test": []}

    n_test = 1
    n_val = 1
    n_train = n - n_val - n_test
    return {
        "train": games[:n_train],
        "val":   games[n_train:n_train + n_val],
        "test":  games[n_train + n_val:],
    }


def collect_image_paths(games: list[Path]) -> list[str]:
    """Collect absolute image paths from all games' train.txt files."""
    paths = []
    for game in games:
        txt = game / "train.txt"
        if not txt.exists():
            print(f"  WARNING: {txt} not found, skipping")
            continue
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            img = Path(line)
            if not img.is_absolute():
                img = (game / img).resolve()
            paths.append(str(img))
    return paths


def read_class_names(game: Path) -> dict:
    """Read class names from a game's data.yaml."""
    data_yaml = game / "data.yaml"
    if not data_yaml.exists():
        return {0: "player", 1: "ball", 2: "referee"}
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names


def main():
    parser = argparse.ArgumentParser(description="Prepare Football2025 data.yaml")
    parser.add_argument("--dataset-root", required=True, help="Path to Football2025 folder (read-only is fine)")
    parser.add_argument("--output-dir", default="data", help="Writable directory for output files (default: data/)")
    parser.add_argument("--train", nargs="*", help="Game folder names for train split")
    parser.add_argument("--val",   nargs="*", help="Game folder names for val split")
    parser.add_argument("--test",  nargs="*", help="Game folder names for test split")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_games = find_game_folders(root)
    print(f"Found {len(all_games)} game folder(s):")
    for g in all_games:
        print(f"  {g.name}")

    # Resolve split assignment
    if args.train or args.val or args.test:
        by_name = {g.name: g for g in all_games}
        splits = {
            "train": [by_name[n] for n in (args.train or [])],
            "val":   [by_name[n] for n in (args.val   or [])],
            "test":  [by_name[n] for n in (args.test  or [])],
        }
    else:
        splits = default_split(all_games)

    print("\nSplit assignment:")
    for split, games in splits.items():
        print(f"  {split:5s}: {[g.name for g in games]}")

    # Collect image paths and write .txt files
    txt_keys = {}
    for split, games in splits.items():
        if not games:
            continue
        paths = collect_image_paths(games)
        out_txt = out_dir / f"{split}.txt"
        out_txt.write_text("\n".join(paths) + "\n")
        print(f"\n{split}.txt: {len(paths)} images → {out_txt}")
        txt_keys[split] = str(out_txt)

    # Read class names from the first game
    names = read_class_names(all_games[0])
    print(f"\nClasses: {names}")

    # Write combined football.yaml — paths in txt_keys are absolute so 'path' is not needed
    out_yaml = out_dir / "football.yaml"
    data_cfg = {"names": names}
    for split in ["train", "val", "test"]:
        if split in txt_keys:
            data_cfg[split] = txt_keys[split]

    with open(out_yaml, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote {out_yaml}")
    print(f"\nUpdate config.yaml dataset_path to:\n  {out_yaml}")


if __name__ == "__main__":
    main()
