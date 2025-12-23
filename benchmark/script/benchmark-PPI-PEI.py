#!/usr/bin/env python
# ============================================================
# Benchmark: PPI / PEI (Refactored, logic-preserving)
# ============================================================

import os
import sys
import time
import glob
import copy
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    recall_score, precision_score, roc_curve,
    matthews_corrcoef, precision_recall_curve
)
from sklearn import metrics
from prefetch_generator import BackgroundGenerator

# ============================================================
# Config
# ============================================================

class Config:
    batch_size = 128
    epochs = 80
    lr = 1e-3
    patience = 10
    num_workers = 32
    prefetch_factor = 2

cfg = Config()

cfg.model_name = sys.argv[1]
cfg.data_dir = sys.argv[2]
cfg.layer = int(sys.argv[3])
cfg.seed = int(sys.argv[4])
cfg.output_dim = int(sys.argv[5])

device = torch.device("cuda")

# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)

def numeric_sort_key(s):
    return int(s.split('-split-')[1].split('-')[0])

# ============================================================
# Dataset
# ============================================================

class SingleFileDataset(Dataset):
    def __init__(self, fp1, fp2, labels):
        self.fp1 = fp1
        self.fp2 = fp2
        self.labels = labels
        with h5py.File(fp1, "r") as f:
            self.length = len(f["embedding"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.fp1, "r") as f1, h5py.File(self.fp2, "r") as f2:
            x1 = f1["embedding"][idx]
            x2 = f2["embedding"][idx]
        y = self.labels[idx]
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

class InterleavedDataset(Dataset):
    def __init__(self, fps1, fps2, label_blocks):
        self.datasets = [
            SingleFileDataset(f1, f2, lb)
            for f1, f2, lb in zip(fps1, fps2, label_blocks)
        ]
        self.lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        offset = 0
        for ds, l in zip(self.datasets, self.lengths):
            if idx < offset + l:
                return ds[idx - offset]
            offset += l
        raise IndexError

class PrefetchDataLoader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# ============================================================
# Load data
# ============================================================

def build_train_dataset():
    data_type = "train"
    fp1 = sorted(
        glob.glob(f"{cfg.data_dir}/{cfg.model_name}/{data_type}_data_0-split-***-embedding-layer_{cfg.layer}.h5"),
        key=numeric_sort_key
    )
    fp2 = sorted(
        glob.glob(f"{cfg.data_dir}/{cfg.model_name}/{data_type}_data_1-split-***-embedding-layer_{cfg.layer}.h5"),
        key=numeric_sort_key
    )
    labels = np.loadtxt(f"{cfg.data_dir}/{data_type}_label.txt", dtype=int).reshape(-1, 1)

    group_size = 5000
    blocks = [
        labels[i:i+group_size]
        for i in range(0, len(labels), group_size)
    ]
    return InterleavedDataset(fp1, fp2, blocks)

class EmbeddingDataset(Dataset):
    def __init__(self, split):
        p1 = sorted(
            glob.glob(f"{cfg.data_dir}/{cfg.model_name}/{split}_data_0-split-***-embedding-layer_{cfg.layer}.h5"),
            key=numeric_sort_key
        )
        p2 = sorted(
            glob.glob(f"{cfg.data_dir}/{cfg.model_name}/{split}_data_1-split-***-embedding-layer_{cfg.layer}.h5"),
            key=numeric_sort_key
        )
        self.labels = np.loadtxt(f"{cfg.data_dir}/{split}_label.txt", dtype=int)
        self.fp1, self.fp2 = p1, p2

        self.indices = []
        for i, fp in enumerate(p1):
            with h5py.File(fp, "r") as f:
                for j in range(len(f["embedding"])):
                    self.indices.append((i, j))

        with h5py.File(p1[0], "r") as f:
            self.data_shape = f["embedding"].shape

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fid, j = self.indices[idx]
        with h5py.File(self.fp1[fid], "r") as f1, h5py.File(self.fp2[fid], "r") as f2:
            x1 = f1["embedding"][j]
            x2 = f2["embedding"][j]
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        )

# ============================================================
# Model
# ============================================================

class DownstreamModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 7, padding="same")

        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv3 = nn.Conv1d(in_channels, 256, 1)
        self.conv4 = nn.Conv1d(256, 256, 7, padding="same")

        self.act = nn.GELU()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, cfg.output_dim)
        self.drop = nn.Dropout(0.2)

    def forward(self, x1, x2):
        x1 = self.act(self.conv2(self.conv1(self.bn1(x1.permute(0,2,1)))))
        x2 = self.act(self.conv4(self.conv3(self.bn2(x2.permute(0,2,1)))))

        x1 = torch.max(x1, dim=2).values
        x2 = torch.max(x2, dim=2).values

        x = torch.cat([x1, x2], dim=1)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.fc3(x)
        return torch.sigmoid(x)

# ============================================================
# Train / Eval
# ============================================================

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total = 0
    for x1, x2, y in loader:
        x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
        pred = model(x1, x2)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total = 0
    for x1, x2, y in loader:
        x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
        total += loss_fn(model(x1, x2), y).item()
    return total / len(loader)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for x1, x2, _ in loader:
        out.append(model(x1.cuda(), x2.cuda()).cpu().numpy())
    return np.concatenate(out).ravel()

# ============================================================
# Main
# ============================================================

def main():
    save_dir = f"{cfg.data_dir}/{cfg.model_name}/checkpoint"
    os.makedirs(save_dir, exist_ok=True)

    train_ds = build_train_dataset()
    valid_ds = EmbeddingDataset("dev")
    test_ds  = EmbeddingDataset("test")

    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers,
                              prefetch_factor=cfg.prefetch_factor)
    valid_loader = PrefetchDataLoader(valid_ds, cfg.batch_size, shuffle=True,
                                      num_workers=cfg.num_workers)
    test_loader  = PrefetchDataLoader(test_ds, cfg.batch_size, shuffle=False,
                                      num_workers=cfg.num_workers)

    model = DownstreamModel(valid_ds.data_shape[2]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCELoss()

    best, counter, best_state = 1e9, 0, None

    for epoch in range(cfg.epochs):
        tr = train_epoch(model, train_loader, optimizer, loss_fn)
        va = evaluate(model, valid_loader, loss_fn)
        print(f"Epoch {epoch} | train {tr:.4f} | valid {va:.4f}")

        if va < best:
            best, counter = va, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter == cfg.patience:
                break

    model.load_state_dict(best_state)
    y_pred = predict(model, test_loader)
    y_true = test_ds.labels

    auc = roc_auc_score(y_true, y_pred)
    print("AUC:", auc)

    np.save(f"{save_dir}/test-{cfg.model_name}_{cfg.layer}_{cfg.seed}_pred.npy", y_pred)
    torch.save(best_state, f"{save_dir}/checkpoint-{cfg.model_name}_{cfg.layer}_{cfg.seed}.pt")

if __name__ == "__main__":
    main()