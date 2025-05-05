"""
Dataset class for flood prediction tasks.
"""
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import GAGE_CSV, RUNOFF_CSV, STATION_ID, ORIG_TIME, T_STEPS, PREDICT_WINDOW


class FloodDataset(Dataset):
    """
    PyTorch Dataset that yields spatial-temporal inputs and runoff targets.
    """
    def __init__(
        self,
        start_times: List[datetime],
        precip_all: torch.Tensor,
        dem: torch.Tensor,
        station_id: str = STATION_ID,
        base_time: datetime = ORIG_TIME,
        t_steps: int = T_STEPS,
    ) -> None:
        """
        Args:
            start_times: List of datetime objects for window starts.
            precip_all: Tensor with all precipitation data (T_total, H, W).
            dem: Tensor with DEM data (H, W).
            station_id: Station identifier as string.
        """
        super().__init__()
        self.start_times = start_times
        self.precip_all = precip_all
        self.dem = dem
        self.station_id = station_id
        self.base_time = base_time
        self.t_steps = t_steps

        # Preload CSVs for efficiency
        self.gage_df = pd.read_csv(GAGE_CSV, parse_dates=["datetime"]).set_index("datetime")
        self.runoff_df = pd.read_csv(RUNOFF_CSV, parse_dates=["datetime"]).set_index("datetime")

    def __len__(self) -> int:
        return len(self.start_times)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_time = self.start_times[idx]
        delta_hours = int((start_time - self.base_time).total_seconds() // 3600)

        precip_seq = self.precip_all[delta_hours : delta_hours + T_STEPS]
        dem_seq = self.dem.unsqueeze(0).repeat(T_STEPS, 1, 1)
        spatial = torch.stack([precip_seq, dem_seq], dim=1)

        target_time = start_time + timedelta(hours=T_STEPS)
        prev_date = (target_time.date() - timedelta(days=1))
        gage_value = float(self.gage_df.at[pd.to_datetime(prev_date), self.station_id])
        gage_tensor = torch.tensor(gage_value).float()

        future_idx = pd.date_range(start=target_time, periods=PREDICT_WINDOW, freq="h")
        runoff_vals = self.runoff_df.loc[future_idx, self.station_id].astype(float).values
        avg_runoff = float(runoff_vals.mean())
        avg_runoff_tensor = torch.tensor(avg_runoff).float()

        prev_time = target_time - timedelta(hours=1)
        try:
            prev_runoff = float(self.runoff_df.at[prev_time, self.station_id])
        except KeyError:
            prev_runoff = 0.0
        prev_runoff_tensor = torch.tensor(prev_runoff).float()

        return spatial, gage_tensor, prev_runoff_tensor, avg_runoff_tensor