# README.md

Flood Model Repository

This repository provides tools for flood prediction using ConvLSTM models.

## Installation
If you want to train the model, please download dataset file ```https://pics1.obs.myhuaweicloud.com/GraduateGW/flood_dataset/precip.pt``` and copy it under ```/data```. (The file exceeds Github's file size limit)

If you just want to test the model, you can run the code below directly.
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

- **Training**: `flood-train --epochs 10 --batch-size 32 --lr 1e-4`
- **Validation**: `flood-validate`

## Project Structure

- `flood_model/`: Package source code
- `data/`: Input precipitation, DEM, gage, and runoff data
- `checkpoint/`: Saved model weights
- `lossPic/`: Training loss curves
- `validation_result/`: Validation CSV outputs
- `lossPic/`: Training loss figure