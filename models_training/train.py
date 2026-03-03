#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import RTDETR


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(config_path):
    config = load_config(config_path)

    task = config["run"]["task"]
    model_cfg = config["models"][task]

    data_yaml = model_cfg["dataset_path"]
    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Check dataset_path in config.yaml. "
            "On IDUN: /cluster/projects/vc/courses/TDT17/other/Football2025/data.yaml\n"
            "On Cybele: /datasets/tdt4265/Football2025/data.yaml"
        )

    model = RTDETR(model_cfg["model"])
    device = get_device()

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_name = f"{task}-{timestamp}"

    model.train(
        data=str(Path(data_yaml).resolve()),
        epochs=model_cfg["epochs"],
        imgsz=model_cfg["imgsz"],
        batch=model_cfg["batch"],
        patience=model_cfg["patience"],
        device=device,
        project=config["run"]["output_dir"],
        name=run_name,
        **({"cls": model_cfg["cls_weights"]} if model_cfg.get("cls_weights") else {}),
    )

    model.val()

    export_formats = model_cfg.get("export", [])
    if export_formats:
        print(f"Exporting model to formats: {export_formats}")
        for fmt in export_formats:
            model.export(format=fmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    train(args.config)
