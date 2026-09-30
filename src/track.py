#!/usr/bin/env python3
"""
Run RT-DETR tracking on a video and save predictions in MOT format.

All parameters live in tracking_config.yaml — only --mode and --config
are accepted as CLI arguments. Use --duration to override the video
length for a quick local test without editing the config.

Usage:
    python src/track.py                          # uses src/tracking_config.yaml
    python src/track.py --mode slurm             # submit to IDUN
    python src/track.py --duration 30            # process only first 30 seconds
    python src/track.py --config my_config.yaml  # use a different config file
"""

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from ultralytics import RTDETR


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackingConfig:
    model: str
    video: str
    tracker: str
    conf: float
    output: str
    name: str
    duration: Optional[float]   # seconds; None = full video
    save_video: bool = False


@dataclass
class SlurmConfig:
    account: str
    partition: str
    gpus: int
    gpu_type: str
    constraint: str
    cpus: int
    mem: str
    time: str
    nodes: int
    ntasks: int
    requirements: str


def load_config(config_path: Path) -> tuple[str, TrackingConfig, SlurmConfig]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    mode = raw["run"]["mode"]

    t = raw["tracking"]
    tracking = TrackingConfig(
        model=t["model"],
        video=t["video"],
        tracker=t["tracker"],
        conf=t["conf"],
        output=t["output"],
        name=t["name"],
        duration=t.get("duration"),
        save_video=t.get("save_video", False),
    )

    s = raw["slurm"]
    slurm = SlurmConfig(
        account=s["account"],
        partition=s["partition"],
        gpus=s["gpus"],
        gpu_type=s["gpu_type"],
        constraint=s["constraint"],
        cpus=s["cpus"],
        mem=s["mem"],
        time=s["time"],
        nodes=s["nodes"],
        ntasks=s["ntasks"],
        requirements=s["requirements"],
    )

    return mode, tracking, slurm


# ---------------------------------------------------------------------------
# Local tracking
# ---------------------------------------------------------------------------

def run_local(cfg: TrackingConfig) -> None:
    model_path = Path(cfg.model)
    video_path = Path(cfg.video)
    output_dir = Path(cfg.output)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    source = str(video_path)

    # Trim video to --duration seconds using ffmpeg into a temp file
    temp_clip = None
    if cfg.duration is not None:
        suffix = video_path.suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_clip = Path(tmp.name)
        tmp.close()
        print(f"Trimming video to {cfg.duration}s → {temp_clip}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-t", str(cfg.duration), "-c", "copy", str(temp_clip)],
            check=True, capture_output=True,
        )
        source = str(temp_clip)

    model = RTDETR(str(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{cfg.name}.txt"

    print(f"Tracking {source} with {cfg.tracker} (conf={cfg.conf}) ...")

    try:
        results = model.track(
            source=source,
            tracker=cfg.tracker,
            conf=cfg.conf,
            save=cfg.save_video,
            stream=True,
        )

        with open(out_file, "w") as f:
            for frame_idx, result in enumerate(results):
                frame_id = frame_idx + 1  # MOT is 1-indexed

                if result.boxes is None or result.boxes.id is None:
                    continue

                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box_xyxy, track_id, det_conf, cls in zip(
                    boxes_xyxy, track_ids, confs, classes
                ):
                    x1, y1, x2, y2 = box_xyxy
                    w = x2 - x1
                    h = y2 - y1
                    f.write(
                        f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f}"
                        f",{det_conf:.4f},{cls},-1\n"
                    )

    finally:
        if temp_clip is not None:
            temp_clip.unlink(missing_ok=True)

    print(f"Saved {out_file}")


# ---------------------------------------------------------------------------
# Slurm submission
# ---------------------------------------------------------------------------

TEMPLATE_PATH = Path(__file__).parent.parent / "models_training" / "job_template.slurm"


def submit_slurm(slurm: SlurmConfig, config_path: Path) -> None:
    Path("logs").mkdir(exist_ok=True)

    template = TEMPLATE_PATH.read_text()
    script_content = template.format(
        job_name="tracking",
        account=slurm.account,
        partition=slurm.partition,
        time=slurm.time,
        nodes=slurm.nodes,
        ntasks=slurm.ntasks,
        cpus=slurm.cpus,
        gpu_type=slurm.gpu_type,
        gpus=slurm.gpus,
        constraint=slurm.constraint,
        mem=slurm.mem,
        requirements=slurm.requirements,
        train_script=f"{__file__} --mode local --config {config_path}",
        config=config_path,
    )

    job_file = Path("track_job.slurm")
    job_file.write_text(script_content)
    subprocess.run(["sbatch", str(job_file)], check=True)
    print(f"Submitted {job_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    default_config = Path(__file__).parent / "tracking_config.yaml"

    parser = argparse.ArgumentParser(description="RT-DETR tracking pipeline")
    parser.add_argument("--mode", choices=["local", "slurm"],
                        help="Override run.mode from config")
    parser.add_argument("--config", default=str(default_config),
                        help=f"Config file (default: {default_config})")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override: process only first N seconds of video")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    mode, tracking, slurm = load_config(config_path)

    if args.mode:
        mode = args.mode
    if args.duration is not None:
        tracking.duration = args.duration

    print(f"Mode:    {mode}")
    print(f"Model:   {tracking.model}")
    print(f"Video:   {tracking.video}")
    print(f"Tracker: {tracking.tracker}")
    print(f"Output:  {tracking.output}/{tracking.name}.txt")
    if tracking.duration:
        print(f"Duration: {tracking.duration}s (trimmed)")

    if mode == "local":
        run_local(tracking)
    else:
        submit_slurm(slurm, config_path)


if __name__ == "__main__":
    main()
