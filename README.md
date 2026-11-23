# SRPA-Net

Official PyTorch implementation of **SRPA-Net** for hyperspectral oil spill binary segmentation (Global Spectral Reliability + Spectral Detail Compensation built upon a frozen pretrained SPRA-Net backbone).

## 1. Repository Structure

```text
SRPA-Net/
├── train.py                                # Main training, validation, and test script
├── run_experiments.py                      # Experiment preparation and sequential runner
├── infer_scene.py                          # Full-scene hyperspectral tiling inference and stitching
├── configs/
│   ├── selected_spectral_detail.json       # Main experiment config (Global Spectral Reliability + Spectral Detail)
│   ├── sensor_metadata_224.json            # 224-band center wavelengths, FWHM, and sensor metadata
│   └── text_embeddings.pt                  # Pre-cached SPRA-Net attribute text embeddings
├── environment/
│   ├── environment.yml                     # Conda environment specification
│   └── requirements.txt                    # Python package requirements
├── scripts/
│   └── train_selected.sh                   # Convenience launch script for the main experiment
├── src/
│   ├── __init__.py
│   ├── model.py                            # SRPANet base class, spectral reliability estimator, and SPRA-Net wrapper
│   ├── innovations.py                      # Spectral detail compensation branch (SpectralDetail) and full model
│   ├── data.py                             # 224-band oil spill tile dataset loader and metadata utilities
│   ├── runtime.py                          # GhostBatchNorm, mixed-precision fallback, and fused optimizer controls
│   └── experiment_controls.py              # Training schedules, loss functions, and optimizer parameter grouping
└── vendor/
    ├── SPRA-Net_README.md                 # Upstream SPRA-Net documentation
    ├── SPRA-Net/                          # Pretrained SPRA-Net encoder and UPerNet decoder
    └── benchmark/                          # Dataset cache reader and spatial augmentation utilities
```

## 2. Environment Setup

The code has been verified in the following environment (an NVIDIA GPU with CUDA support is required):

| Dependency | Version |
| :--- | :--- |
| Python | `3.11.11` |
| PyTorch / TorchVision | `2.5.1` / `0.20.1` (CUDA) |
| torchdata | `0.11.0` |
| timm | `1.0.15` |
| transformers / sentence-transformers | `4.51.3` / `4.1.0` |
| kornia | `0.8.0` |
| GDAL / NumPy | `3.6.2` / `< 2.0` |

### Option 1: Create via `environment.yml` (Recommended)

```bash
conda env create -f environment/environment.yml
conda activate SPRA-Net
```

### Option 2: Manual Conda + Pip Installation

```bash
conda create -n SPRA-Net python=3.11.11 -y
conda activate SPRA-Net

# Install geospatial and scientific dependencies (GDAL 3.6.2 requires the NumPy 1.x ABI)
conda install -y gdal=3.6.2 "numpy<2" rasterio netcdf4 pandas scipy

# Install PyTorch (adjust the CUDA index URL to match your local CUDA version; CUDA 12.1 example below)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python packages
pip install -r environment/requirements.txt
```

## 3. Pretrained Weights & Data Preparation

### 3.1 Download Pretrained SPRA-Net Weights

Download `SPRA-Net_Base_model.pth` from the [official SPRA-Net link](https://r2-proxy.3sobs.top/f6UhnB) (see [vendor/SPRA-Net_README.md](vendor/SPRA-Net_README.md)) and place it at:

```text
SRPA-Net/
└── checkpoints/
    └── SPRA-Net_Base_model.pth
```

### 3.2 Prepare the 224-Band Tile Dataset Cache

Place the preprocessed 224-band hyperspectral oil spill tiles (`.npy` format) under `data/224_fullbands/`:

```text
SRPA-Net/
└── data/
    └── 224_fullbands/
        ├── tr_images.npy       # shape: (N_tr, 224, 224, 224), float32, per-tile per-band min-max normalized
        ├── tr_labels.npy       # shape: (N_tr, 224, 224), uint8, 0=background, 1=oil, 255=ignore
        ├── val_images.npy      # shape: (N_val, 224, 224, 224), float32
        ├── val_labels.npy      # shape: (N_val, 224, 224), uint8
        ├── ts_images.npy       # shape: (N_ts, 224, 224, 224), float32
        └── ts_labels.npy       # shape: (N_ts, 224, 224), uint8
```

> To use custom paths, update `pretrained_checkpoint`, `cache_root`, and `scenes_root` in `configs/selected_spectral_detail.json`.

## 4. Reproducing the Main Experiment

The main experiment freezes the pretrained SPRA-Net encoder and trains the global spectral reliability estimator, full-resolution spectral detail compensation branch, and UPerNet segmentation head (224 bands, seed 42, effective batch size 8, AdamW with `lr=1e-4`, 20,000 updates, validation every 500 steps with best-checkpoint selection by validation mIoU).

### Training and Evaluation

```bash
conda activate SPRA-Net
python run_experiments.py \
  --config configs/selected_spectral_detail.json \
  --run-dir runs/srpa_net
```

Or run the launch script:

```bash
bash scripts/train_selected.sh runs/srpa_net
```

- **Resuming**: If training is interrupted, re-running the same command with the same `--run-dir` automatically resumes from `last.ckpt` with model, optimizer, AMP scaler, and RNG states preserved.
- **Tile-only evaluation**: Pass `--skip-scene-predictions` if full `.mat` test scenes are not placed under `data/scenes/`.

### Outputs

After training and testing complete, all outputs are saved to `runs/srpa_net/models/spectral_detail/`:

- `best.ckpt`: Best checkpoint selected by validation mIoU
- `last.ckpt`: Latest resumable checkpoint (model, optimizer, scaler, and RNG states)
- `result.json`: Full test metrics for both best-validation and final checkpoints
- `PR_mIoU_IoU_percent.csv`: Summary table of Precision, Recall, F1, Oil IoU, and mIoU (%)

## 5. Full-Scene Hyperspectral Inference

`infer_scene.py` supports `.mat` (default key `img`, shape `H×W×C`), `.npy` (shape `C×H×W`), and GDAL raster files. Scenes are tiled into non-overlapping `224×224` patches with edge padding, inferred, and stitched back to original dimensions:

```bash
python infer_scene.py \
  --run-dir runs/srpa_net \
  --variant spectral_detail \
  --scene /path/to/scene.mat \
  --output outputs/prediction \
  --threshold 0.5
```

Inference outputs:

- `outputs/prediction.probability.npy`: Oil probability map (`float32`, `H×W`)
- `outputs/prediction.mask.npy`: Binary segmentation mask (`uint8`, `H×W`)
- `outputs/prediction.tiles.jsonl`: Per-tile spectral reliability weight diagnostics
- `outputs/prediction.metadata.json`: Inference settings and scene metadata

## 6. Acknowledgements

The backbone encoder and decoder are built upon [SPRA-Net](https://www.sciencedirect.com/science/article/pii/S0924271626000754). We thank the authors for releasing their pretrained weights and code.
