"""
Training script for the flood prediction model.
"""
import argparse
import os
import re
from datetime import timedelta

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import (
    CHECKPOINT_DIR, PIC_DIR, T_STEPS, BATCH_SIZE,
    LR, EPOCHS, TRAIN_END, TRAIN_START, STATION_ID
)
from .data_loader import direct_preload
from .dataset import FloodDataset
from .model import FloodPredictor


def train_model(args: argparse.Namespace) -> None:
    """
    Train the flood prediction model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precip_all, dem = direct_preload(device)

    n_hours = int((TRAIN_END - TRAIN_START).total_seconds() // 3600) - T_STEPS + 1
    start_times = [TRAIN_START + timedelta(hours=i) for i in range(n_hours)]

    dataset = FloodDataset(
        start_times,
        precip_all, 
        dem, 
        STATION_ID, 
        base_time=TRAIN_START, 
        t_steps=T_STEPS
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model = FloodPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Resume if checkpoint exists
    pattern = re.compile(r"FloodPredictor_epoch(\d+)\.pth")
    ckpts = []
    for f in os.listdir(CHECKPOINT_DIR):
        m = pattern.match(f)
        if m:
            ckpts.append((int(m.group(1)), f))
    if ckpts:
        latest = max(ckpts, key=lambda x: x[0])
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, latest[1]), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = latest[0] + 1
    else:
        start_epoch = 1

    loss_history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for spatial, gage, prev_runoff, runoff in pbar:
            spatial = spatial.to(device)
            gage = gage.to(device)
            prev_runoff = prev_runoff.to(device)
            runoff = runoff.to(device)

            pred = model(spatial, gage, prev_runoff)
            loss = criterion(pred, runoff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch}: Avg Loss={avg_loss:.4f}")

        if epoch == args.epochs:
            save_name = f"FloodPredictor_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(CHECKPOINT_DIR, save_name))

            plt.figure(figsize=(8, 4))
            plt.plot(loss_history, marker='o')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(PIC_DIR, 'loss_curve.png'))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train flood prediction model"
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=BATCH_SIZE,
        dest='batch_size',
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=LR,
        help='Learning rate'
    )
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()