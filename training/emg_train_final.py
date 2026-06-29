"""
EMG Final Model — Train & Save
================================
Trains the Residual CNN on the merged dataset (4-class: Closed, Hook, Pencil, Rest)
and saves all files needed for live inference.

Dataset:
    merged_closed.csv  — 30 reps 
    merged_hook.csv    — 30 reps
    merged_pencil.csv  — 30 reps
    Rest extracted from rest intervals of the above three files.

Saves
    residual_cnn_final.pth   — model weights
    scaler_mean.npy          — StandardScaler mean  (90 values)
    scaler_scale.npy         — StandardScaler scale (90 values)
    mvc_rms.npy              — MVC calibration dict per gesture
    label_names.txt          — class index to name mapping

Requirements:
    pip install torch scikit-learn numpy pandas scipy matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_FILES = {
    "closed": "closed.csv",
    "hook":   "hook.csv",
    "pencil": "pencil.csv",
}
DATA_DIR   = r"YOUR_DATASET_DIRECTORY"  # Path to the folder containing the CSV files
OUTPUT_DIR = r"YOUR_OUTPUT_DIRECTORY"  # Path to the folder where results will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS          = 300
WINDOW_MS   = 300
OVERLAP_MS  = 250
WINDOW_SAMP = int(FS * WINDOW_MS  / 1000)   # 90 samples
STEP_SAMP   = int(FS * (WINDOW_MS - OVERLAP_MS) / 1000)   # 15 samples

EPOCHS      = 100
BATCH_SIZE  = 64
LR          = 1e-3
PATIENCE    = 20
N_FOLDS     = 5
SEED        = 42

AUG_NOISE_STD   = 0.05
AUG_SCALE_RANGE = (0.8, 1.2)
AUG_SHIFT_MAX   = 5
AUG_COPIES      = 2

# 4-class: Closed=0, Hook=1, Pencil=2, Rest=3
LABEL_NAMES = ["Closed", "Hook", "Pencil", "Rest"]
N_CLASSES   = 4
GRIP_LABELS = {"closed": 0, "hook": 1, "pencil": 2}
REST_LABEL  = 3

# MVC RMS values (S1 channel, top-10% method)
MVC_RMS = {
    "Closed": 113.20,
    "Hook":   17.43,
    "Pencil": 32.62,
}

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Window : {WINDOW_MS} ms ({WINDOW_SAMP} samples) | Step: {STEP_SAMP} samples")


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════
def load_and_segment():
    all_X, all_y = [], []
    for name, fname in DATA_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        df   = pd.read_csv(path)
        ts   = df["Timestamp_sec"].values
        sig  = df["Filtered"].values
        grip_label = GRIP_LABELS[name]
        # Use all complete 10-second cycles present in the file
        n_reps = int(ts[-1] // 10)
        for rep in range(n_reps):
            for t_start, t_end, label in [
                (rep*10,   rep*10+5,  REST_LABEL),
                (rep*10+5, rep*10+10, grip_label),
            ]:
                seg = sig[(ts >= t_start) & (ts < t_end)]
                if len(seg) < WINDOW_SAMP:
                    continue
                for start in range(0, len(seg) - WINDOW_SAMP, STEP_SAMP):
                    all_X.append(seg[start:start+WINDOW_SAMP])
                    all_y.append(label)
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    print(f"\n[Data] {len(X)} total windows")
    for i, n in enumerate(LABEL_NAMES):
        print(f"  {n:10s}: {(y == i).sum()} windows")
    return X, y


def augment_batch(X, y, rng):
    aug_X, aug_y = [], []
    for _ in range(AUG_COPIES):
        Xa = X.copy()
        stds = Xa.std(axis=1, keepdims=True) + 1e-8
        Xa  += rng.normal(0, AUG_NOISE_STD, Xa.shape).astype(np.float32) * stds
        scales = rng.uniform(*AUG_SCALE_RANGE, size=(len(Xa), 1)).astype(np.float32)
        Xa    *= scales
        shifts = rng.integers(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1, size=len(Xa))
        for i, sh in enumerate(shifts):
            if sh > 0:
                Xa[i] = np.concatenate([np.zeros(sh, dtype=np.float32), Xa[i, :-sh]])
            elif sh < 0:
                Xa[i] = np.concatenate([Xa[i, -sh:], np.zeros(-sh, dtype=np.float32)])
        aug_X.append(Xa); aug_y.append(y)
    return np.concatenate([X]+aug_X), np.concatenate([y]+aug_y)


# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1    = nn.Conv1d(in_ch,  out_ch, kernel_size, stride=stride, padding=pad)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.relu     = nn.ReLU()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride),
            nn.BatchNorm1d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResidualCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.stages = nn.Sequential(
            ResBlock(32,  64,  stride=2),
            ResBlock(64,  128, stride=2),
            ResBlock(128, 256, stride=1),
            ResBlock(256, 256, stride=1),
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.classifier(self.pool(x).squeeze(-1))


# ═══════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════
def make_criterion(y_tr):
    counts  = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
    weights = torch.tensor(
        counts.sum() / (N_CLASSES * counts + 1e-10), dtype=torch.float32
    ).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        correct += (model(xb).argmax(1) == yb).sum().item()
        total   += len(yb)
    return correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    correct = total = 0
    preds_all, true_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds  = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total   += len(yb)
        preds_all.extend(preds.cpu().numpy())
        true_all.extend(yb.cpu().numpy())
    return correct / total, np.array(preds_all), np.array(true_all)


# ═══════════════════════════════════════════════════════════════════
# CROSS-VALIDATION  (evaluation only — not used for saving)
# ═══════════════════════════════════════════════════════════════════
def run_cross_validation(X, y):
    print("\n[Cross-Validation] 5-fold stratified...")
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rng  = np.random.default_rng(SEED)
    fold_accs, all_preds, all_true = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_n = scaler.fit_transform(X_tr).astype(np.float32)
        X_val_n = scaler.transform(X_val).astype(np.float32)

        X_tr_aug, y_tr_aug = augment_batch(X_tr_n, y_tr, rng)

        tr_loader  = DataLoader(
            TensorDataset(torch.tensor(X_tr_aug).unsqueeze(1), torch.tensor(y_tr_aug)),
            batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val_n).unsqueeze(1), torch.tensor(y_val)),
            batch_size=BATCH_SIZE)

        criterion = make_criterion(y_tr_aug)
        model     = ResidualCNN(N_CLASSES).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_acc, best_p, best_t, patience_ctr = 0, None, None, 0
        for epoch in range(EPOCHS):
            train_epoch(model, tr_loader, optimizer, criterion)
            val_acc, preds, true = eval_epoch(model, val_loader, criterion)
            scheduler.step()
            if val_acc > best_acc:
                best_acc, best_p, best_t, patience_ctr = val_acc, preds.copy(), true.copy(), 0
            else:
                patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

        fold_accs.append(best_acc)
        all_preds.extend(best_p)
        all_true.extend(best_t)
        print(f"  Fold {fold+1}: {best_acc*100:.2f}%")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n  CV Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=LABEL_NAMES))

    # Save confusion matrix
    cm = confusion_matrix(all_true, all_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Final Residual CNN — {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cm_final_cv.png"), dpi=150)
    plt.close()
    print("  Saved: cm_final_cv.png")
    return mean_acc, std_acc


# ═══════════════════════════════════════════════════════════════════
# FULL-DATASET TRAINING & SAVE
# ═══════════════════════════════════════════════════════════════════
def train_and_save(X, y):
    print("\n[Training on full dataset for deployment...]")
    rng    = np.random.default_rng(SEED)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X).astype(np.float32)

    # Save scaler
    np.save(os.path.join(OUTPUT_DIR, "scaler_mean.npy"),  scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, "scaler_scale.npy"), scaler.scale_)

    X_aug, y_aug = augment_batch(X_norm, y, rng)
    print(f"  After augmentation: {len(X_aug)} windows")

    loader    = DataLoader(
        TensorDataset(torch.tensor(X_aug).unsqueeze(1), torch.tensor(y_aug)),
        batch_size=BATCH_SIZE, shuffle=True)
    criterion = make_criterion(y_aug)
    model     = ResidualCNN(N_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        acc = train_epoch(model, loader, optimizer, criterion)
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: train_acc={acc:.4f}")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "residual_cnn_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  Model  saved → {model_path}")

    # Save MVC
    mvc_path = os.path.join(OUTPUT_DIR, "mvc_rms.npy")
    np.save(mvc_path, MVC_RMS)
    print(f"  MVC    saved → {mvc_path}")

    # Save label names
    label_path = os.path.join(OUTPUT_DIR, "label_names.txt")
    with open(label_path, "w") as f:
        for i, n in enumerate(LABEL_NAMES):
            f.write(f"{i},{n}\n")
    print(f"  Labels saved → {label_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  EMG FINAL MODEL — TRAIN & SAVE")
    print("  4-class | Merged dataset | Residual CNN")
    print("=" * 60)

    X, y = load_and_segment()

    # Step 1: cross-validation for honest accuracy estimate
    # run_cross_validation(X, y)

    # Step 2: retrain on full data and save
    train_and_save(X, y)

    print("\n" + "=" * 60)
    print("  DONE — all files saved to:")
    print(f"  {OUTPUT_DIR}")
    print("\n  Now run: master_esp32_final.ino on master ESP32")
    print("            slave_esp32_final.ino  on slave  ESP32")
    print("            emg_live_final.py      on PC")
    print("=" * 60)


if __name__ == "__main__":
    main()
