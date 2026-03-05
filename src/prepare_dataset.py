#!/usr/bin/env python3
"""
Build a combined dataset layout for Football2025 and write football.yaml.

Each game folder is structured like:
    <game>/
        data/images/train/   <- images
        labels/train/        <- labels (NOT under data/)
        train.txt
        data.yaml

Ultralytics derives label paths from image paths by replacing 'images' with
'labels'. Since images live at data/images/ but labels at labels/ (different
depth), Ultralytics finds nothing.

Fix: create symlinks in --output-dir so images and labels sit at the same
depth, then point football.yaml at that structure:
    <output_dir>/
        images/train/  <- symlinks named <game>__<frame>.png -> real image
        images/val/
        images/test/
        labels/train/  <- symlinks named <game>__<frame>.txt -> real label
        labels/val/
        labels/test/
        football.yaml

Usage:
    python src/prepare_dataset.py \\
        --dataset-root /cluster/projects/vc/courses/TDT17/other/Football2025 \\
        --output-dir data/

Split assignment (4 games): 2 train, 1 val, 1 test
Override with --train/--val/--test by game folder name.
"""

import argparse
import os
import yaml
from pathlib import Path


def find_game_folders(dataset_root: Path) -> list[Path]:
    """Return subdirectories that look like game folders (have labels/ and data/images/)."""
    games = sorted(
        d for d in dataset_root.iterdir()
        if d.is_dir()
        and (d / "labels").exists()
        and (d / "data" / "images").exists()
    )
    if not games:
        raise RuntimeError(f"No game folders found in {dataset_root}")
    return games


def default_split(games: list[Path]) -> dict[str, list[Path]]:
    n = len(games)
    if n == 1:
        return {"train": games, "val": games, "test": []}
    if n == 2:
        return {"train": games[:1], "val": games[1:], "test": []}
    return {
        "train": games[:n - 2],
        "val":   games[n - 2:n - 1],
        "test":  games[n - 1:],
    }


def symlink_split(games: list[Path], img_dir: Path, lbl_dir: Path) -> tuple[int, int]:
    """Symlink all images and labels from the given games into img_dir/lbl_dir."""
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_imgs = n_lbls = 0
    for game in games:
        game_img_dir = game / "data" / "images" / "train"
        game_lbl_dir = game / "labels" / "train"

        if not game_img_dir.exists():
            print(f"  WARNING: image dir not found: {game_img_dir}")
        else:
            for img in sorted(game_img_dir.iterdir()):
                link = img_dir / f"{game.name}__{img.name}"
                if not link.exists():
                    os.symlink(img.resolve(), link)
                n_imgs += 1

        if not game_lbl_dir.exists():
            print(f"  WARNING: label dir not found: {game_lbl_dir}")
        else:
            for lbl in sorted(game_lbl_dir.iterdir()):
                link = lbl_dir / f"{game.name}__{lbl.name}"
                if not link.exists():
                    os.symlink(lbl.resolve(), link)
                n_lbls += 1

    return n_imgs, n_lbls


def read_class_names(game: Path) -> dict:
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
    parser = argparse.ArgumentParser(description="Prepare Football2025 dataset layout")
    parser.add_argument("--dataset-root", required=True, help="Path to Football2025 (read-only ok)")
    parser.add_argument("--output-dir", default="data", help="Writable output directory (default: data/)")
    parser.add_argument("--train", nargs="*", help="Game names for train split")
    parser.add_argument("--val",   nargs="*", help="Game names for val split")
    parser.add_argument("--test",  nargs="*", help="Game names for test split")
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

    print()
    active_splits = {}
    for split, games in splits.items():
        if not games:
            continue
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        n_imgs, n_lbls = symlink_split(games, img_dir, lbl_dir)
        print(f"{split:5s}: {n_imgs} image symlinks, {n_lbls} label symlinks")
        active_splits[split] = f"images/{split}"

    names = read_class_names(all_games[0])
    print(f"\nClasses: {names}")

    out_yaml = out_dir / "football.yaml"
    data_cfg = {"path": str(out_dir), "names": names}
    for split in ["train", "val", "test"]:
        if split in active_splits:
            data_cfg[split] = active_splits[split]

    with open(out_yaml, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote {out_yaml}")
    print(f"Update config.yaml dataset_path to:\n  {out_yaml}")


if __name__ == "__main__":
    main()
