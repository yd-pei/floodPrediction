import os
from datetime import datetime, timedelta
import re

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ===================== 超参数 & 路径 =====================
data_dir = "./data/"
model_dir = "./model/"
picPath = "./lossPic/"
PRECIP_DIR  = os.path.join(data_dir, "Precipitation")
DEM_PATH    = os.path.join(data_dir, "dem_resampled.tif")
GAGE_CSV    = os.path.join(data_dir, "gageheight_ffill.csv")
RUNOFF_CSV  = os.path.join(data_dir, "runoff_hourly.csv")

T_STEPS     = 72      # 输入时间步 (小时)
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 20
STATION_ID  = 2301738

# ===================== 数据加载函数 =====================

def load_precipitation_sequence(start_time: datetime, t_steps: int, precip_dir: str):
    """读取连续 t_steps 小时的降水 tif -> (T, H, W)"""
    seq = []
    for i in range(t_steps):
        ts       = start_time + timedelta(hours=i)
        ts_str   = ts.strftime("%Y%m%d-%H%M%S")
        tif_name = f"MultiSensor_QPE_01H_Pass2_00.00_{ts_str}.grib2_clipped.tif"
        tif_path = os.path.join(precip_dir, tif_name)
        with rasterio.open(tif_path) as src:
            seq.append(src.read(1))
    return np.stack(seq, axis=0)  # (T, H, W)


def load_dem(dem_path: str):
    with rasterio.open(dem_path) as src:
        return src.read(1)  # (H, W)


def load_gage_runoff(station_id: int, gage_csv: str, runoff_csv: str, t_steps: int, target_time: datetime):
    """返回 (前一天 gage_height 标量, 目标 runoff 值)"""
    gage_df   = pd.read_csv(gage_csv, parse_dates=["datetime"]).set_index("datetime")
    runoff_df = pd.read_csv(runoff_csv, parse_dates=["datetime"]).set_index("datetime")

    col = str(station_id)
    prev_date   = (target_time.date() - timedelta(days=1))
    gage_scalar = gage_df.at[pd.to_datetime(prev_date), col]
    runoff_val  = runoff_df.at[target_time, col]
    return gage_scalar, runoff_val

# ===================== Dataset =====================

class FloodDataset(Dataset):
    def __init__(self, start_times, t_steps, precip_dir, dem_path, gage_csv, runoff_csv, station_id):
        self.start_times = start_times
        self.t_steps     = t_steps
        self.precip_dir  = precip_dir
        self.dem         = load_dem(dem_path)
        self.gage_csv    = gage_csv
        self.runoff_csv  = runoff_csv
        self.station_id  = station_id

    def __len__(self):
        return len(self.start_times)

    def __getitem__(self, idx):
        start_time   = self.start_times[idx]
        precip_seq   = load_precipitation_sequence(start_time, self.t_steps, self.precip_dir)  # (T, H, W)
        dem_stack    = np.stack([self.dem] * self.t_steps, axis=0)
        spatial      = np.stack([precip_seq, dem_stack], axis=1)  # (T, 2, H, W)
        spatial      = torch.from_numpy(spatial).float()

        target_time          = start_time + timedelta(hours=self.t_steps)
        gage_scalar, runoff  = load_gage_runoff(self.station_id, self.gage_csv, self.runoff_csv, self.t_steps, target_time)
        gage_scalar          = torch.tensor(gage_scalar).float()
        runoff               = torch.tensor(runoff).float()
        return spatial, gage_scalar, runoff

# ===================== 模型定义 =====================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding   = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv  = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, input_seq):
        B, T, C, H, W = input_seq.shape
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=input_seq.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(input_seq[:, t], h, c)
        return h


class FloodPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, kernel_size=3):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, spatial, gage_scalar):
        h       = self.convlstm(spatial)
        h_pool  = self.pool(h).flatten(1)
        x       = torch.cat([h_pool, gage_scalar.view(-1, 1)], dim=1)
        return self.fc(x).squeeze(1)
# test
def verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = datetime(2020, 1, 1, 0)
    end   = datetime(2021, 1, 1, 0)
    total_hours  = int((end - start).total_seconds() // 3600)
    start_times  = [start + timedelta(hours=i) for i in range(total_hours - T_STEPS + 1)]

    dataset     = FloodDataset(start_times, T_STEPS, PRECIP_DIR, DEM_PATH, GAGE_CSV, RUNOFF_CSV, STATION_ID)
    dataloader  = DataLoader(dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=(device.type == "cuda"),
                             persistent_workers=True)
    model       = FloodPredictor().to(device)
    checkpoints = []
    checkpoint_dir = model_dir
    pattern = re.compile(r"FloodPredictor_epoch(\d+)\.pth")
    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, fname))

    if checkpoints:
        latest_epoch, latest_file = max(checkpoints, key=lambda x: x[0])
        checkpoint_path = os.path.join(checkpoint_dir, latest_file)
        print(f"加载最新模型：{latest_file}（epoch {latest_epoch}）")
        checkpoint = torch.load(checkpoint_path,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
    else:
        print("No checkpoint.")
        return
    
    model.eval()
    
    # —— 2. 测试代码（禁用梯度计算） ——
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for spatial_batch, gage_batch, runoff_batch in dataloader:
            spatial_batch = spatial_batch.to(device)
            gage_batch    = gage_batch.to(device)
            runoff_batch  = runoff_batch.to(device)

            preds = model(spatial_batch, gage_batch)

            preds_list.append(preds.cpu())
            labels_list.append(runoff_batch.cpu())

    # —— 3. 整合输出 ——
    preds_all  = torch.cat(preds_list).numpy()
    labels_all = torch.cat(labels_list).numpy()

    # —— 4. 可视化结果 ——
    plt.figure(figsize=(10, 5))
    plt.plot(labels_all, label="True Runoff")
    plt.plot(preds_all,  label="Predicted Runoff")
    plt.xlabel("Sample Index")
    plt.ylabel("Runoff")
    plt.title("Model Predictions vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===================== 主训练流程 =====================

def main():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"预测站点 ID: {STATION_ID}")

    # 生成时间窗口起点
    start = datetime(2021, 1, 1, 0)
    end   = datetime(2024, 1, 1, 0)
    total_hours  = int((end - start).total_seconds() // 3600)
    start_times  = [start + timedelta(hours=i) for i in range(total_hours - T_STEPS + 1)]

    dataset     = FloodDataset(start_times, T_STEPS, PRECIP_DIR, DEM_PATH, GAGE_CSV, RUNOFF_CSV, STATION_ID)
    dataloader  = DataLoader(dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=(device.type == "cuda"),
                             persistent_workers=True)

    model       = FloodPredictor().to(device)
    criterion   = nn.MSELoss()
    optimizer   = optim.Adam(model.parameters(), lr=LR)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    checkpoint_dir = model_dir
    pattern = re.compile(r"FloodPredictor_epoch(\d+)\.pth")
    start_epoch = 1
    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, fname))

    if checkpoints:
        latest_epoch, latest_file = max(checkpoints, key=lambda x: x[0])
        checkpoint_path = os.path.join(checkpoint_dir, latest_file)
        print(f"加载最新模型：{latest_file}（epoch {latest_epoch}）")
        checkpoint = torch.load(checkpoint_path,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        model.to(device)        

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        batch_loss_history = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for spatial, gage, runoff in pbar:
            spatial, gage, runoff = spatial.to(device), gage.to(device), runoff.to(device)
            pred   = model(spatial, gage)
            loss   = criterion(pred, runoff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_loss_history.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # clear_output(wait=True)
        plt.figure(figsize=(8, 4))
        plt.plot(batch_loss_history, marker='o', linestyle='-')
        plt.xlabel('Batch Index')
        plt.ylabel('Loss')
        plt.title(f'Epoch {epoch} Batch Loss')
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(picPath,f"loss_epoch{epoch}.png")
        plt.savefig(save_path)
        # plt.show()
            

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
        scheduler.step(avg_loss)

        file_name = f"FloodPredictor_epoch{epoch}.pth"
        path_name = os.path.join(model_dir,file_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path_name)
        print(f"模型已保存: FloodPredictor_epoch{epoch}.pth")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
