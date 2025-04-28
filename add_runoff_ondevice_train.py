import matplotlib
matplotlib.use('Agg')
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
from concurrent.futures import ThreadPoolExecutor

# ===================== 超参数 =====================
data_dir    = "./data/"
model_dir   = "./checkpoint/"
picPath     = "./lossPic/"
checkpoint_dir = "./checkpoint/"
PRECIP_DIR  = os.path.join(data_dir, "Precipitation")
PRECIP_PT   = os.path.join(data_dir, "precip.pt")
DEM_PATH    = os.path.join(data_dir, "dem_resampled.tif")
GAGE_CSV    = os.path.join(data_dir, "gageheight_ffill.csv")
RUNOFF_CSV  = os.path.join(data_dir, "runoff_hourly.csv")
VALID_CKP   = os.path.join(checkpoint_dir, "lstm36min_epoch200_1e_3.pth")

T_STEPS     = 36      # 滑窗长度
BATCH_SIZE  = 32
LR_POWER = -3
LR          = 1*(10 ** LR_POWER)
EPOCHS      = 2
STATION_ID  = 2301738

READING_THREAD = 0

def direct_preload(device):
    precip_all = torch.load(PRECIP_PT, map_location=device) 
    with rasterio.open(DEM_PATH) as src:
        dem_np = src.read(1)
    dem = torch.from_numpy(dem_np).float().to(device)

    return precip_all, dem

def _load_one_precip(path):
    """Helper：打开单个 tif，返回 (H, W) numpy array"""
    with rasterio.open(path) as src:
        return src.read(1)

def parallel_preload_data(device):
    # 1. 列出所有降水文件，按名称排序保证时间顺序
    files = sorted(os.listdir(PRECIP_DIR))
    paths = [os.path.join(PRECIP_DIR, f) for f in files]

    # 2. 并行加载
    with ThreadPoolExecutor(max_workers=READING_THREAD) as exe:
        # executor.map 会保持输入顺序
        arrays = list(tqdm(
            exe.map(_load_one_precip, paths),
            total=len(paths),
            desc="并行加载降水栅格"
        ))

    # 3. 拼成 (T_total, H, W) 并一次性搬到 GPU
    precip_np  = np.stack(arrays, axis=0)
    precip_all = torch.from_numpy(precip_np).float().to(device)

    # 4. DEM 同样搬到 GPU
    with rasterio.open(DEM_PATH) as src:
        dem_np = src.read(1)
    dem = torch.from_numpy(dem_np).float().to(device)

    return precip_all, dem

# ================ 先在 GPU 上预加载所有光栅 ================
def preload_data(device):
    # 1. 全量降水
    files = sorted(os.listdir(PRECIP_DIR))
    precip_list = []
    for fname in tqdm(files, desc="加载降水栅格"):
        path = os.path.join(PRECIP_DIR, fname)
        with rasterio.open(path) as src:
            precip_list.append(src.read(1))
    precip_np = np.stack(precip_list, axis=0)    # (T_total, H, W)
    precip_all = torch.from_numpy(precip_np).float().to(device)

    # 2. DEM
    with rasterio.open(DEM_PATH) as src:
        dem_np = src.read(1)                       # (H, W)
    dem = torch.from_numpy(dem_np).float().to(device)

    return precip_all, dem

# ================ 新版 Dataset，直接在 GPU 上切片 ================
class FloodDataset(Dataset):
    def __init__(self, start_times, precip_all, dem, gage_csv, runoff_csv, station_id, base_time, t_steps):
        self.start_times  = start_times
        self.precip_all   = precip_all      # 全量降水张量 (T_total, H, W) on GPU
        self.dem          = dem             # DEM (H, W) on GPU
        self.gage_csv     = gage_csv
        self.runoff_csv   = runoff_csv
        self.station_id   = station_id
        self.base_time    = base_time       # 与 precip_all 第 0 步对应的时间点
        self.t_steps      = t_steps

        # 读一次 pandas 表，加速 __getitem__
        self.gage_df   = pd.read_csv(gage_csv, parse_dates=["datetime"]).set_index("datetime")
        self.runoff_df= pd.read_csv(runoff_csv, parse_dates=["datetime"]).set_index("datetime")

    def __len__(self):
        return len(self.start_times)

    def __getitem__(self, idx):
        start_time = self.start_times[idx]
        # 计算在 precip_all 中的起始索引
        delta_hours = int((start_time - self.base_time).total_seconds() // 3600)
        # GPU 上直接切片
        precip_seq = self.precip_all[delta_hours : delta_hours + self.t_steps]         # (T, H, W)
        dem_stack  = self.dem.unsqueeze(0).repeat(self.t_steps, 1, 1)                   # (T, H, W)
        spatial    = torch.stack([precip_seq, dem_stack], dim=1)                       # (T, 2, H, W)

        # gage & runoff 读自 CPU pandas，再转 GPU
        target_time   = start_time + timedelta(hours=self.t_steps)
        prev_date     = (target_time.date() - timedelta(days=1))
        gage_scalar   = torch.tensor(self.gage_df.at[pd.to_datetime(prev_date), str(self.station_id)]).float().to(self.precip_all.device)
        runoff_scalar = torch.tensor(self.runoff_df.at[target_time, str(self.station_id)]).float().to(self.precip_all.device)
        
        prev_time   = target_time - timedelta(hours=1)
        # 新增：前一时刻 runoff
        try:
            prev_runoff_scalar = torch.tensor(
                self.runoff_df.at[prev_time, str(self.station_id)]
            ).float().to(self.precip_all.device)
        except KeyError:
            # 如果缺失，可以填 0 或者其它策略
            print(f"Warning: Missing runoff data for {prev_time}. Filling with 0.")
            prev_runoff_scalar = torch.tensor(0.0).float().to(self.precip_all.device)


        return spatial, gage_scalar,prev_runoff_scalar , runoff_scalar

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


# ================ 修改 FloodPredictor =================
class FloodPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, kernel_size=3):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        # 这里把输入维度从 hidden_dim+1 → hidden_dim+2
        self.fc       = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, spatial, gage_scalar, prev_runoff):
        # convLSTM + 池化
        h      = self.convlstm(spatial)
        h_pool = self.pool(h).flatten(1)        # (B, hidden_dim)
        # 拼接 gage 和前一时刻 runoff
        x = torch.cat([
            h_pool,
            gage_scalar.view(-1, 1),
            prev_runoff.view(-1, 1)
        ], dim=1)                              # (B, hidden_dim+2)
        return self.fc(x).squeeze(1)

def verify_non_overlapping_windows_to_csv():
    """
    非重叠 36 小时窗口验证，并把结果输出到 CSV。
    """
    # 1. 设备 & 预加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precip_all, dem = direct_preload(device)

    # 2. 加载模型
    model = FloodPredictor().to(device)
    ckpt = torch.load(VALID_CKP, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. 定义验证窗口
    val_start     = datetime(2020, 1, 1, 0, 0)
    val_end       = datetime(2021, 1, 1, 0, 0)
    window_delta  = timedelta(hours=T_STEPS)
    total_windows = (val_end - val_start) // window_delta

    # 4. 循环验证，收集结果
    records = []
    for i in range(total_windows):
        start_time = val_start + i * window_delta
        end_time   = start_time + window_delta

        ds     = FloodDataset(
            start_times=[start_time],
            precip_all=precip_all,
            dem=dem,
            gage_csv=GAGE_CSV,
            runoff_csv=RUNOFF_CSV,
            station_id=STATION_ID,
            base_time=val_start,
            t_steps=T_STEPS
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        with torch.no_grad():
            # 解包时多一个 prev_runoff
            for spatial, gage, prev_runoff, true_runoff in loader:
                spatial      = spatial.to(device)
                gage         = gage.to(device)
                prev_runoff  = prev_runoff.to(device)
                true_runoff  = true_runoff.to(device)

                # 把 prev_runoff 也传进去
                pred = model(spatial, gage, prev_runoff)

                pred_val = pred.item()
                true_val = true_runoff.item()
                err      = pred - true_runoff
                mse      = float((err ** 2).item())
                mae      = float(err.abs().item())

        records.append({
            "window_idx":   i + 1,
            "start_time":   start_time.isoformat(),
            "end_time":     end_time.isoformat(),
            "predicted":    pred_val,
            "ground_truth": true_val,
            "mse":          mse,
            "mae":          mae
        })
        print(f"[窗口 {i+1}] {start_time} → {end_time}  MSE={mse:.4f}  MAE={mae:.4f}")

    # 5. 保存到 CSV
    df       = pd.DataFrame(records)
    csv_path = os.path.join(checkpoint_dir, "predictions_non_overlap.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n已将所有窗口的预测结果保存到: {csv_path}")


# ================ 主流程 ================================
def main():
    # 1. 选设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 预加载全部光栅到 GPU
    precip_all, dem = direct_preload(device)

    # 3. 构造时间列表
    base_time   = datetime(2021, 1, 1, 0)
    end         = datetime(2024, 1, 1, 0)
    total_hours = int((end - base_time).total_seconds() // 3600)
    start_times = [base_time + timedelta(hours=i) for i in range(total_hours - T_STEPS + 1)]

    # 4. Dataset & DataLoader
    dataset = FloodDataset(
        start_times,
        precip_all,
        dem,
        GAGE_CSV,
        RUNOFF_CSV,
        STATION_ID,
        base_time,
        T_STEPS
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=READING_THREAD,
        pin_memory=False,
        persistent_workers=False
    )

    # 5. 模型、优化器、损失、调度
    model     = FloodPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 6. （可选）加载 checkpoint
    pattern = re.compile(r"FloodPredictor_epoch(\d+)\.pth")
    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, fname))
    if checkpoints:
        latest_epoch, latest_file = max(checkpoints, key=lambda x: x[0])
        ckpt = torch.load(os.path.join(checkpoint_dir, latest_file), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = latest_epoch + 1
        print(f"Resuming from epoch {latest_epoch}")
    else:
        start_epoch = 1

    loss_history = []

    # 7. 训练循环
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for spatial, gage, prev_runoff, runoff in pbar:
            # data 已经在 GPU 上
            # forward
            pred = model(spatial, gage, prev_runoff)
            loss = criterion(pred, runoff)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: Avg Loss={avg_loss:.4f}")

        # 调度
        scheduler.step(avg_loss)
        loss_history.append(avg_loss)

        # 保存 checkpoint & 画图
        if epoch == EPOCHS:
            save_name = f"Newlstm{T_STEPS}h_epoch{EPOCHS}_1e_{abs(LR_POWER)}.pth"
            save_path = os.path.join(checkpoint_dir, save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, save_path)

            plt.figure(figsize=(8, 4))
            plt.plot(loss_history, marker='o')
            plt.title("Training Loss Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            pic_name = f"Newloss{T_STEPS}h_1e_{abs(LR_POWER)}_curve.png"
            plt.savefig(os.path.join(picPath, pic_name))
            print("Loss curve saved.")
 

if __name__ == "__main__":
    main()
    # verify_non_overlapping_windows_to_csv()
