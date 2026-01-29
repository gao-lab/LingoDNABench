import os
import sys
import copy
import random
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.backends import cudnn

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    recall_score, precision_score, roc_curve,
    matthews_corrcoef, precision_recall_curve
)
from sklearn import metrics


# =========================
# Environment & Seed
# =========================

def setup_env():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Argument Parsing
# =========================

model_name = sys.argv[1]
data_dir = sys.argv[2]
layer = int(sys.argv[3])
random_seed = int(sys.argv[4])
output_dim = int(sys.argv[5])

regression = sys.argv[6] == "True"
num_targets = output_dim

setup_env()
set_seed(random_seed)

device = torch.device("cuda")


# =========================
# IO Path
# =========================

save_dir = f"{data_dir}/{model_name}/checkpoint"
os.makedirs(save_dir, exist_ok=True)


# =========================
# Data Loading
# =========================

def load_labels(data_dir, regression):
    dtype = float if regression else int
    train = torch.as_tensor(np.loadtxt(f"{data_dir}/train_label.txt", dtype=dtype))
    valid = torch.as_tensor(np.loadtxt(f"{data_dir}/dev_label.txt", dtype=dtype))
    test  = torch.as_tensor(np.loadtxt(f"{data_dir}/test_label.txt", dtype=dtype))
    return train, valid, test


def load_embeddings(data_dir, model_name, layer):
    def _load(split):
        return torch.as_tensor(
            np.load(
                f"{data_dir}/{model_name}/{split}_data_0-embedding-layer_{layer}.npy"
            )
        )
    return _load("train"), _load("dev"), _load("test")


def build_dataset(data, labels):
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)
    return TensorDataset(data, labels)


train_label, valid_label, test_label = load_labels(data_dir, regression)
train_data, valid_data, test_data = load_embeddings(data_dir, model_name, layer)

train_dataset = build_dataset(train_data, train_label)
valid_dataset = build_dataset(valid_data, valid_label)
test_dataset  = build_dataset(test_data, test_label)


# =========================
# Dataloader
# =========================

batch_size = 64

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)
test_dataloader  = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# =========================
# Model
# =========================

class DownstreamModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 7, padding="same")
        self.act = nn.GELU()
        self.linear1 = nn.Linear(256, 512)
        self.drop = nn.Dropout(0.2)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.batchnorm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = torch.nn.functional.max_pool1d(x, kernel_size=x.shape[2])
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear3(x)
        if output_dim == 1 and not regression:
            x = torch.sigmoid(x)
        return x


model = DownstreamModel(valid_data.shape[2]).to(device)


# =========================
# Loss & Optimizer
# =========================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if regression:
    loss_fn = nn.MSELoss()
else:
    loss_fn = nn.BCELoss() if output_dim == 1 else nn.CrossEntropyLoss()


# =========================
# Helpers
# =========================

def preprocess_targets(targets):
    targets = targets.to(torch.float32)
    if output_dim != 1:
        targets = targets.to(torch.int32)
        targets = torch.tensor(targets.squeeze(1), dtype=torch.long)
    return targets


def run_epoch(dataloader, model, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for DNA, targets in dataloader:
            DNA = DNA.to(device, dtype=torch.float32)
            targets = preprocess_targets(targets.to(device))
            preds = model(DNA).to(torch.float32)
            loss = loss_fn(preds, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)


def predict(dataloader, model):
    model.eval()
    preds = np.empty((len(dataloader.dataset), output_dim))
    offset = 0

    with torch.no_grad():
        for DNA, _ in dataloader:
            DNA = DNA.to(device, dtype=torch.float32)
            out = model(DNA)
            if output_dim != 1:
                out = nn.Softmax(dim=1)(out)
            n = DNA.size(0)
            preds[offset:offset+n] = out.cpu().numpy()
            offset += n

    return preds


# =========================
# Training Loop
# =========================

epochs = 80
patience = 10
count = 0
min_valid_loss = float("inf")
final_data = {}

for epoch in range(epochs):
    train_loss = run_epoch(train_dataloader, model, loss_fn, optimizer)
    valid_loss = run_epoch(valid_dataloader, model, loss_fn)

    print(f"train_loss\t{train_loss}")
    print(f"valid_loss\t{valid_loss}")

    state = {
        "epoch": epoch,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
    }

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        final_data = copy.deepcopy(state)
        count = 0
    else:
        count += 1
        if count == patience:
            break


# =========================
# Testing & Metrics
# =========================

test_model = DownstreamModel(valid_data.shape[2]).to(device)
test_model.load_state_dict(final_data["model_state_dict"])

y_pred = predict(test_dataloader, test_model)

if regression:
    y_test = test_label.cpu().numpy()
    import scipy.stats
    pearson = scipy.stats.pearsonr(y_test.reshape(-1), y_pred.reshape(-1))[0]
    spearman = scipy.stats.spearmanr(y_test.reshape(-1), y_pred.reshape(-1))[0]
    with open(f"{save_dir}/metrics-{layer}.txt", "a") as f:
        f.write(f"{model_name}\t{random_seed}\t{spearman}\t{pearson}\n")

else:
    if output_dim == 1:
        y_test = test_label.flatten().cpu().numpy()
        y_pred = y_pred.reshape(-1)

        auc = roc_auc_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc = metrics.auc(recall, precision)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        cutoff = thresholds[np.argmax(tpr - fpr)]

        y_bin = (y_pred >= cutoff).astype(int)
        acc = accuracy_score(y_test, y_bin)
        prec = precision_score(y_test, y_bin)
        rec = recall_score(y_test, y_bin)
        f1 = f1_score(y_test, y_bin)
        mcc = matthews_corrcoef(y_test, y_bin)

        with open(f"{save_dir}/metrics-{layer}.txt", "a") as f:
            f.write(
                f"{model_name}\t{random_seed}\t{cutoff}\t"
                f"{acc}\t{prec}\t{rec}\t{f1}\t{auc}\t{pr_auc}\t{mcc}\n"
            )
    else:
        y_test = test_label.cpu().numpy()
        auc = roc_auc_score(y_test, y_pred, multi_class="ovo")
        with open(f"{save_dir}/metrics-{layer}.txt", "a") as f:
            f.write(f"{model_name}\t{random_seed}\t{auc}\n")


# =========================
# Save
# =========================

np.save(
    f"{save_dir}/test-{model_name}_{layer}_{random_seed}_pred.npy",
    y_pred
)
torch.save(
    final_data,
    f"{save_dir}/checkpoint-{model_name}_{layer}_{random_seed}.pt"
)
