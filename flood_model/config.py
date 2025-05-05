"""
Configuration constants for the flood prediction package.
"""
from pathlib import Path
from datetime import datetime

# Base project directory
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Data directories and file paths
DATA_DIR: Path = BASE_DIR / "data"
PRECIP_PT: Path = DATA_DIR / "precip.pt"
DEM_PATH: Path = DATA_DIR / "dem_resampled.tif"
GAGE_CSV: Path = DATA_DIR / "gageheight_ffill.csv"
RUNOFF_CSV: Path = DATA_DIR / "runoff_hourly.csv"

# Output directories
CHECKPOINT_DIR: Path = BASE_DIR / "checkpoint"
PIC_DIR: Path = BASE_DIR / "lossPic"
VALIDATION_DIR: Path = BASE_DIR / "validation_result"

# Validation checkpoint
VALID_CKPT: Path = CHECKPOINT_DIR / "tw12_24h_epoch200_1e_4.pth"

# Model and training hyperparameters
T_STEPS: int = 24
PREDICT_WINDOW: int = 12
BATCH_SIZE: int = 32
LR: float = 1e-4
EPOCHS: int = 2
STATION_ID: str = "2301738"
READING_THREAD: int = 0

# Global time range for training
ORIG_TIME = datetime(2020, 1, 1, 0, 0)

TRAIN_START      = datetime(2021, 1, 1, 0, 0)
TRAIN_END        = datetime(2024, 12, 30, 23, 0)
VALID_START      = datetime(2020, 1, 1, 0, 0)
VALID_END        = datetime(2021, 1, 1, 0, 0)

# BASE_TIME: datetime = datetime(2020, 1, 1, 0, 0)
# END_TIME: datetime = datetime(2024, 12, 30, 23, 0)