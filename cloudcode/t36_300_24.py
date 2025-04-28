import matplotlib
matplotlib.use('Agg')
import os
from datetime import datetime, timedelta
import re
import numpy as np
import pandas as pd
import rasterio
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
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

T_STEPS     = 36      # 滑窗长度
BATCH_SIZE  = 32
LR_POWER = -3
LR          = 1*(10 ** LR_POWER)
EPOCHS      = 200
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

        return spatial, gage_scalar, runoff_scalar

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
                             num_workers=READING_THREAD,
                             pin_memory=(device.type == "cuda"),
                             persistent_workers=True)
    model       = FloodPredictor().to(device)
    checkpoints = []
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

# ================ 主流程 ================================
def main():
    # 1. 选设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}",flush = True)

    # 2. 预加载全部光栅到 GPU
    # precip_all, dem = parallel_preload_data(device)
    precip_all, dem = direct_preload(device)

    # 3. 构造时间列表
    base_time   = datetime(2021, 1, 1, 0)
    end         = datetime(2024, 1, 1, 0)
    total_hours = int((end - base_time).total_seconds() // 3600)
    start_times = [base_time + timedelta(hours=i) for i in range(total_hours - T_STEPS + 1)]

    # 4. Dataset & DataLoader
    dataset  = FloodDataset(start_times, precip_all, dem, GAGE_CSV, RUNOFF_CSV, STATION_ID, base_time, T_STEPS)
    loader   = DataLoader(dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=READING_THREAD,
                          pin_memory=False,
                          persistent_workers=False
    )

    # 5. 模型、优化器、训练循环照旧
    model     = FloodPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # …（加载 checkpoint 逻辑保持不变）…
    pattern = re.compile(r"FloodPredictor_epoch(\d+)\.pth")
    start_epoch = 1
    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, fname))

    loss_history = []

    for epoch in range(start_epoch, EPOCHS+1):
        model.train()
        total_loss = 0.0
        nan_n = 0
        # pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for spatial, gage, runoff in loader:
            # spatial, gage, runoff 已经都在 GPU 上，无需 .to(device) 拷贝
            pred = model(spatial, gage)
            loss = criterion(pred, runoff)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_n += 1
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            # pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}: Avg Loss={avg:.4f}")
        print("the num of nan:" + str(nan_n),flush = True)
        scheduler.step(avg)
        loss_history.append(avg)
        # …（保存 checkpoint、画图等逻辑保持不变）…

        if epoch == EPOCHS:
            save_path = "lstm{inputw}min_epoch{nepoch}_1e_{thelr}.pth".format(
                nepoch = EPOCHS,
                inputw = T_STEPS,
                thelr = abs(LR_POWER)
            )
            save_path = checkpoint_dir + save_path
            torch.save({
                'epoch': EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg
            }, save_path)
            plt.figure(figsize=(8, 4))
            plt.plot(loss_history, marker='o')
            plt.title("Training Loss Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            pic_name = "loss{inputw}min_1e_{thelr}_curve.png".format(
                inputw = T_STEPS,
                thelr = abs(LR_POWER)
                )
            pic_name = picPath + pic_name
            plt.savefig(pic_name)
            print("Loss curve saved.")            



if __name__ == "__main__":
    main()
