"""
Validation script to check model performance over non-overlapping windows.
"""
from datetime import timedelta

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import VALID_END, VALID_START, VALIDATION_DIR, VALID_CKPT, T_STEPS, STATION_ID
from .data_loader import direct_preload
from .dataset import FloodDataset
from .model import FloodPredictor


def verify_non_overlapping_windows_to_csv() -> None:
    """
    Run validation over non-overlapping windows and save metrics to CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precip_all, dem = direct_preload(device)

    model = FloodPredictor().to(device)
    ckpt = torch.load(VALID_CKPT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    window_delta = timedelta(hours=T_STEPS)
    n_windows = int((VALID_END - VALID_START) / window_delta)
    
    records = []
    for i in range(n_windows):
        start_time = VALID_START + i * window_delta
        ds = FloodDataset(
            [start_time], 
            precip_all, 
            dem, 
            STATION_ID,
            base_time=VALID_START,
            t_steps=T_STEPS
        )
        loader = DataLoader(ds, batch_size=1)

        with torch.no_grad():
            for spatial, gage, prev_runoff, true_runoff in loader:
                spatial = spatial.to(device)
                gage = gage.to(device)
                prev_runoff = prev_runoff.to(device)
                true_runoff = true_runoff.to(device)

                pred = model(spatial, gage, prev_runoff)
                error = pred - true_runoff
                mse = (error ** 2).item()
                mae = error.abs().item()

        records.append({
            "window_idx": i + 1,
            "start_time": start_time.isoformat(),
            "predicted": pred.item(),
            "ground_truth": true_runoff.item(),
            "mse": mse,
            "mae": mae,
        })

    df = pd.DataFrame(records)
    output = VALIDATION_DIR / 'predictions_non_overlap.csv'
    df.to_csv(output, index=False, encoding='utf-8-sig')
    print(f"Saved validation results to {output}") 
    
    df = pd.DataFrame(records)
    output = VALIDATION_DIR / 'predictions_non_overlap.csv'
    df.to_csv(output, index=False, encoding='utf-8-sig')
    print(f"Saved validation results to {output}")

    # print MSE、RMSE、MAE、NMAE
    avg_mse = df["mse"].mean()
    avg_rmse = avg_mse ** 0.5
    avg_mae = df["mae"].mean()
    mean_truth = df["ground_truth"].mean()
    nmae = avg_mae / mean_truth if mean_truth != 0 else float('nan')
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"NMAE: {nmae:.4f}")



def main() -> None:
    verify_non_overlapping_windows_to_csv()


if __name__ == '__main__':
    main()