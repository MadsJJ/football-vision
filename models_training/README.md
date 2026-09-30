# Training pipeline

Fine-tunes RT-DETR on the Football2025 dataset, either locally or as a Slurm job on NTNU's IDUN cluster. All settings live in [`config.yaml`](config.yaml).

## Usage

Prepare the dataset first (from the repo root, see the main [README](../README.md#dataset)). Then, from this folder:

```bash
python run.py --mode local    # train on this machine
python run.py --mode slurm    # submit a job to IDUN
```

`run.py` reads `config.yaml`, installs `requirements.txt`, and either runs `train.py` directly or renders `job_template.slurm` and submits it with `sbatch`.

## Configuration

| Key | Value | Why |
|---|---|---|
| `model` | `rtdetr-l.pt` | COCO-pretrained RT-DETR-L |
| `imgsz` | `1280` | The ball is < 1% of frame width; at 640 px it is only a few pixels |
| `cls` | `2.0` | Upweights classification loss to counter the 26:1 player/ball imbalance |
| `epochs` / `patience` | `100` / `20` | Early stopping ended the final run at epoch 37 |
| `cos_lr` | `true` | Cosine learning-rate schedule |
| `close_mosaic` | `20` | Turns off mosaic augmentation for the last epochs |
| `classes` | `[0, 1]` | Player and ball only; `event_labels` is excluded |

Slurm resources are set under `slurm:`. RT-DETR-L at 1280 px needs a 40 GB+ A100 (`constraint: "gpu40g|gpu80g"`).

## Monitoring Slurm jobs

```bash
export SLURM_JOB=<job id printed by run.py>

squeue -u $USER                     # queued / running jobs
tail -f *$SLURM_JOB*.out            # live log
sacct -j $SLURM_JOB --format=JobID,State,Elapsed,MaxRSS,AllocGRES
scancel $SLURM_JOB
```
