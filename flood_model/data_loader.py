"""
Data loading utilities for precipitation and DEM data.
"""
from typing import Tuple

import rasterio
import torch

from .config import PRECIP_PT, DEM_PATH


def direct_preload(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load preprocessed precipitation tensor and DEM into memory.

    Args:
        device: Torch device to map tensors to.

    Returns:
        precip_all: Tensor of shape (T, H, W) with precipitation data.
        dem: Tensor of shape (H, W) with DEM data.
    """
    precip_all = torch.load(PRECIP_PT, map_location=device)
    with rasterio.open(DEM_PATH) as src:
        dem_array = src.read(1)
    dem = torch.from_numpy(dem_array).float().to(device)
    return precip_all, dem
