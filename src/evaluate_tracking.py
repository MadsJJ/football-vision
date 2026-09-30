#!/usr/bin/env python3
"""
Evaluate tracking predictions against ground truth using TrackEval (HOTA, MOTA, MOTP).

Expected layout:
    data/tracking/
        gt/
            <seq>/
                gt/gt.txt
                seqinfo.ini
        trackers/
            bytetrack/
                <seq>.txt
            botsort/
                <seq>.txt

Usage:
    uv run src/evaluate_tracking.py
    uv run src/evaluate_tracking.py --tracker bytetrack
    uv run src/evaluate_tracking.py --seq rbk-aalesund

Install:
    uv pip install trackeval
"""

import argparse
from pathlib import Path


def find_trackers(trackers_dir: Path) -> list[str]:
    trackers = [
        p.name for p in trackers_dir.iterdir()
        if p.is_dir() and list(p.glob("*.txt"))
    ]
    if not trackers:
        raise SystemExit(f"No tracker result directories found in {trackers_dir}")
    return sorted(trackers)


def find_sequences(gt_dir: Path) -> list[str]:
    seqs = [
        p.name for p in gt_dir.iterdir()
        if p.is_dir() and (p / "gt" / "gt.txt").exists()
    ]
    if not seqs:
        raise SystemExit(f"No sequences with gt/gt.txt found in {gt_dir}")
    return sorted(seqs)


def run_trackeval(gt_dir: Path, trackers_dir: Path,
                  trackers: list[str], sequences: list[str]) -> None:
    try:
        import trackeval
    except ImportError:
        raise SystemExit("trackeval is not installed. Run:\n  uv pip install trackeval")

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_dir),
        "TRACKERS_FOLDER": str(trackers_dir),
        "TRACKERS_TO_EVAL": trackers,
        "TRACKER_SUB_FOLDER": "",
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SEQMAP_FILE": None,
        "SEQ_INFO": {seq: None for seq in sequences},
        "SPLIT_TO_EVAL": "",
        "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": False,  # our class IDs don't follow MOT Challenge convention
        "PRINT_CONFIG": False,
    })

    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "PRINT_CONFIG": False}

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"DISPLAY_LESS_PROGRESS": True, "PRINT_CONFIG": False})

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]

    results, _ = evaluator.evaluate(dataset_list, metrics_list)

    print("\n" + "=" * 60)
    print("TRACKING EVALUATION RESULTS")
    print("=" * 60)

    for tracker in trackers:
        print(f"\nTracker: {tracker}")
        for seq in sequences:
            try:
                seq_results = results["MotChallenge2DBox"][tracker]["COMBINED_SEQ"]["pedestrian"]
                hota = seq_results["HOTA"]["HOTA"].mean() * 100
                mota = seq_results["CLEAR"]["MOTA"] * 100
                motp = seq_results["CLEAR"]["MOTP"] * 100
                idf1 = seq_results["Identity"]["IDF1"] * 100
                print(f"  Sequence: {seq}")
                print(f"    HOTA:  {hota:.2f}%")
                print(f"    MOTA:  {mota:.2f}%")
                print(f"    MOTP:  {motp:.2f}%")
                print(f"    IDF1:  {idf1:.2f}%")
            except KeyError:
                print(f"  Sequence: {seq} — could not extract results (check GT/pred format)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-object tracking with HOTA")
    parser.add_argument("--gt", default="data/tracking/gt",
                        help="Ground truth directory (default: data/tracking/gt)")
    parser.add_argument("--trackers", default="data/tracking/trackers",
                        help="Trackers directory (default: data/tracking/trackers)")
    parser.add_argument("--tracker", nargs="*",
                        help="Tracker name(s) to evaluate (default: all with results)")
    parser.add_argument("--seq", nargs="*",
                        help="Sequence name(s) to evaluate (default: all)")
    args = parser.parse_args()

    gt_dir = Path(args.gt).resolve()
    trackers_dir = Path(args.trackers).resolve()

    if not gt_dir.exists():
        raise SystemExit(f"GT directory not found: {gt_dir}")
    if not trackers_dir.exists():
        raise SystemExit(f"Trackers directory not found: {trackers_dir}")

    trackers = args.tracker if args.tracker else find_trackers(trackers_dir)
    sequences = args.seq if args.seq else find_sequences(gt_dir)

    print(f"Trackers:  {trackers}")
    print(f"Sequences: {sequences}")

    run_trackeval(gt_dir, trackers_dir, trackers, sequences)


if __name__ == "__main__":
    main()
