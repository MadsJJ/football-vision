# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TDT4265 mini-project (Option 3). Deep learning pipeline for analyzing RBK football match footage — detecting and tracking players, the referee, and the ball. Deadline: April 21st.

## Repository Layout

```
models_training/    # Training pipeline (Ultralytics, local + Slurm)
src/                # Evaluation scripts, utilities
notebooks/          # EDA and result analysis
configs/            # Additional model/experiment configs
data/               # Dataset (gitignored)
models/             # Saved checkpoints (gitignored)
```

## Dataset

Located on NTNU compute infrastructure — not downloaded from Roboflow:
- IDUN: `/cluster/projects/vc/courses/TDT17/other/Football2025`
- Cybele: `/datasets/tdt4265/Football2025`

3 classes: players, referee, ball. Annotated with 2D bounding boxes and consistent tracking IDs (including `not_visible` property for occluded objects).

## Architecture Decisions

- **Detection model:** RT-DETR (via Ultralytics) — `RTDETR("rtdetr-l.pt")`. Chosen over YOLO for academic depth; API is identical to YOLO so the existing pipeline requires minimal changes.
- **Training:** Fine-tune from COCO pretrained weights.
- **Ball detection:** Single unified model with class-weighted loss and augmentation to compensate for class imbalance.
- **Tracking (optional):** ByteTrack post-detection, evaluated with HOTA.

## Training Pipeline (`models_training/`)

Entry point: `python run.py [--mode local|slurm] [--task detection]`

- `run.py` — reads `config.yaml`, installs deps, runs locally or submits Slurm job
- `train.py` — Ultralytics training script; uses `RTDETR` instead of `YOLO`
- `config.yaml` — all settings (model, epochs, batch, Slurm resources, dataset path)
- `job_template.slurm` — Slurm job template for IDUN

Dataset path is passed directly in `config.yaml` (no Roboflow download). Slurm account: NTNU student account (not `studiegrupper-vortex`).

## Performance Metrics

Detection: Precision, Recall, mAP@50, mAP@0.5:0.95
Tracking: HOTA

## Implementation Plan

See `.claude/plans/adaptive-tinkering-wadler.md` for the full phased plan. Check the checkboxes there to track progress as phases are completed:
- Phase 1: Adapt `models_training/` for football (RT-DETR, local dataset path)
- Phase 2: EDA notebook
- Phase 3: Evaluation script
- Phase 4: Tracking with ByteTrack + HOTA (optional)
