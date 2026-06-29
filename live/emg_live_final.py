"""
EMG Live Inference — Final System
===================================
Full pipeline: EMG acquisition → Residual CNN classification →
MVC-normalised proportional control → servo command over ESP-NOW.

Architecture:
    PC (this script)
      ├─ Reads filtered EMG from Master ESP32 over USB serial (COM13)
      ├─ Classifies gesture with Residual CNN every 50 ms
      ├─ Accumulates 2-second majority vote
      ├─ Computes MVC-normalised effort from raw window RMS
      └─ Sends "G<idx>,E<effort_pct>\n" back to Master ESP32

    Master ESP32 (master_esp32_final.ino)
      └─ Forwards command to Slave ESP32 over ESP-NOW

    Slave ESP32 (slave_esp32_final.ino)
      └─ Drives 5× MG996R servos via PCA9685

Command format sent to Master:
    "G<gesture_idx>,E<effort_0_to_100>\n"
    e.g. "G0,E78\n" = Closed at 78% MVC effort

Gesture indices:
    0 = Closed   1 = Hook   2 = Pencil   3 = Rest

MVC-normalised effort:
    effort = clip(RMS(window) / MVC_RMS[gesture], 0, 1)
    → modulates servo step size on slave ESP32
    → Rest always sends effort 0 (no actuation)

Prerequisites (run once):
    python emg_train_final.py
    → produces emg_results_final/ with model + scaler + mvc files

Requirements:
    pip install torch numpy pyserial
"""

import os
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import serial
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
COM_PORT   = "COMXX" # Your ESP COM PORT
BAUD_RATE  = 230400
MODEL_DIR  = r"YOUR_MODEL_DIRECTORY"

FS            = 300
WINDOW_MS     = 300
WINDOW_SAMP   = int(FS * WINDOW_MS / 1000)   # 90 samples
STEP_SAMP     = int(FS * 50 / 1000)          # 15 samples → classify every 50 ms

PREDICT_EVERY_S      = 2.0    # seconds per majority-vote window
CONFIDENCE_THRESHOLD = 0.55   # ignore predictions below this confidence

LABEL_NAMES = ["Closed", "Hook", "Pencil", "Rest"]
N_CLASSES   = 4

COLORS = {
    "Closed":    "\033[94m",
    "Hook":      "\033[92m",
    "Pencil":    "\033[95m",
    "Rest":      "\033[93m",
    "uncertain": "\033[90m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"

device = torch.device("cpu")


# ─────────────────────────────────────────────────────────────
# MODEL  (identical to training)
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────────────────────
def load_assets():
    model = ResidualCNN(n_classes=N_CLASSES).to(device)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "residual_cnn_final.pth"),
                   map_location=device)
    )
    model.eval()

    s_mean  = np.load(os.path.join(MODEL_DIR, "scaler_mean.npy"))
    s_scale = np.load(os.path.join(MODEL_DIR, "scaler_scale.npy"))

    mvc_raw = np.load(os.path.join(MODEL_DIR, "mvc_rms.npy"),
                      allow_pickle=True).item()
    # Map gesture index → MVC RMS value
    # LABEL_NAMES: Closed=0, Hook=1, Pencil=2, Rest=3
    mvc_rms = {
        0: mvc_raw.get("Closed", 1.0),
        1: mvc_raw.get("Hook",   1.0),
        2: mvc_raw.get("Pencil", 1.0),
        3: 1.0,   # Rest — effort unused
    }

    print("[OK] Model, scaler, and MVC values loaded.")
    print("     MVC RMS (S1 channel):")
    for idx, name in enumerate(LABEL_NAMES):
        if idx < 3:
            print(f"       {name:10s}: {mvc_rms[idx]:.2f}")
    return model, s_mean, s_scale, mvc_rms


# ─────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────
def normalize_window(window, mean, scale):
    return ((window - mean) / (scale + 1e-8)).astype(np.float32)


def compute_effort(window_raw, gesture_idx, mvc_rms):
    """
    MVC-normalised effort from raw (un-normalised) window.
    Uses the raw signal RMS so electrode gain is consistent
    with the MVC calibration recording.
    Rest always returns 0.
    """
    if gesture_idx == 3:
        return 0.0
    rms    = float(np.sqrt(np.mean(window_raw ** 2)))
    effort = rms / mvc_rms[gesture_idx]
    return float(np.clip(effort, 0.0, 1.0))


@torch.no_grad()
def predict(model, window_raw, s_mean, s_scale, mvc_rms):
    """
    Full inference:
      1. Normalise for model input
      2. Residual CNN → gesture + confidence
      3. Compute effort from raw RMS / MVC_RMS
    """
    x      = normalize_window(window_raw, s_mean, s_scale)
    x      = torch.tensor(x).unsqueeze(0).unsqueeze(0)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])
    effort = compute_effort(window_raw, idx, mvc_rms)
    return idx, LABEL_NAMES[idx], conf, effort, probs


# ─────────────────────────────────────────────────────────────
# SERIAL
# ─────────────────────────────────────────────────────────────
def parse_emg_line(line):
    try:
        return float(line.strip())
    except ValueError:
        return None


def send_command(ser, gesture_idx, effort):
    """
    "G<idx>,E<effort_pct>\n"
    e.g. "G0,E78\n"
    """
    cmd = f"G{gesture_idx},E{int(effort * 100)}\n"
    ser.write(cmd.encode("utf-8"))


# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────
def effort_bar(effort, width=15):
    filled = int(effort * width)
    return f"[{'█'*filled}{'░'*(width-filled)}] {effort*100:.0f}%"


def conf_bar(conf, width=10):
    filled = int(conf * width)
    return f"[{'█'*filled}{'░'*(width-filled)}] {conf*100:.0f}%"


def countdown_bar(elapsed, total, width=18):
    filled = int(min(elapsed / total, 1.0) * width)
    return f"[{'█'*filled}{'░'*(width-filled)}] {max(0, total-elapsed):.1f}s"


def print_status(name, conf, effort, elapsed):
    color  = COLORS.get(name, "")
    status = name if conf >= CONFIDENCE_THRESHOLD else "uncertain"
    e_str  = effort_bar(effort) if name != "Rest" else "      [REST — no actuation]     "
    print(f"\r{BOLD}{color}{status:10s}{RESET}  "
          f"conf {conf_bar(conf)}  "
          f"effort {e_str}  "
          f"next {countdown_bar(elapsed, PREDICT_EVERY_S)}   ",
          end="", flush=True)


def print_decision(name, conf, effort, vote_counts, total_confident, last):
    color   = COLORS.get(name, "")
    changed = name != last
    print(f"\n{'═'*60}")
    print(f"  Decision   : {BOLD}{color}{name}{RESET}"
          f"  {'→ CHANGED' if changed else '  (same)'}")
    print(f"  Confidence : {conf*100:.1f}%  "
          f"({'SENT' if conf >= CONFIDENCE_THRESHOLD else 'below gate — REST sent'})")
    if name != "Rest":
        print(f"  Effort     : {effort*100:.1f}% of MVC")
    print(f"  Votes      : ", end="")
    for i, n in enumerate(LABEL_NAMES):
        pct = vote_counts[i] / max(total_confident, 1) * 100
        print(f"{n}={vote_counts[i]}({pct:.0f}%)  ", end="")
    print(f"\n{'═'*60}\n")


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  EMG LIVE INFERENCE — FINAL SYSTEM")
    print(f"  Port        : {COM_PORT} @ {BAUD_RATE} baud")
    print(f"  Sample rate : {FS} Hz | Window: {WINDOW_MS} ms ({WINDOW_SAMP} samples)")
    print(f"  Classes     : {LABEL_NAMES}")
    print(f"  Vote window : {PREDICT_EVERY_S:.0f} s | Conf gate: >{CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Control     : MVC-normalised effort → servo speed")
    print("=" * 60)

    model, s_mean, s_scale, mvc_rms = load_assets()

    buffer             = collections.deque(maxlen=WINDOW_SAMP)
    samples_seen       = 0
    samples_since_pred = 0

    vote_counts     = [0] * N_CLASSES
    total_confident = 0
    interval_start  = None
    last_gesture    = "Rest"
    last_idx        = 3

    latest_name   = "Rest"
    latest_conf   = 0.0
    latest_effort = 0.0

    print(f"\nOpening {COM_PORT}... (Ctrl+C to quit)\n")

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"[OK] Connected to {COM_PORT}")
        print(f"     Filling buffer ({WINDOW_SAMP} samples = {WINDOW_MS} ms)...\n")

        while True:
            raw_line = ser.readline().decode("utf-8", errors="ignore")
            value    = parse_emg_line(raw_line)
            if value is None:
                continue

            buffer.append(value)
            samples_seen       += 1
            samples_since_pred += 1

            if len(buffer) < WINDOW_SAMP:
                continue

            if interval_start is None:
                interval_start = time.time()

            # Classify every STEP_SAMP new samples (~50 ms)
            if samples_since_pred >= STEP_SAMP:
                samples_since_pred = 0
                window_raw = np.array(buffer, dtype=np.float32)
                idx, name, conf, effort, _ = predict(
                    model, window_raw, s_mean, s_scale, mvc_rms
                )
                latest_name, latest_conf, latest_effort = name, conf, effort

                if conf >= CONFIDENCE_THRESHOLD:
                    vote_counts[idx] += 1
                    total_confident  += 1

            elapsed = time.time() - interval_start
            print_status(latest_name, latest_conf, latest_effort, elapsed)

            # Every 5 seconds — majority vote → send command
            if (time.time() - interval_start) >= PREDICT_EVERY_S:

                if total_confident == 0:
                    winner_idx  = 3   # default Rest
                    winner_name = "Rest"
                    winner_conf = 0.0
                    winner_effort = 0.0
                else:
                    winner_idx   = int(np.argmax(vote_counts))
                    winner_name  = LABEL_NAMES[winner_idx]
                    winner_conf  = vote_counts[winner_idx] / total_confident
                    # Compute effort from the last raw window for the winning gesture
                    winner_effort = compute_effort(
                        np.array(buffer, dtype=np.float32),
                        winner_idx, mvc_rms
                    )

                # Send command to master ESP32
                # If below confidence gate, send Rest (safe default)
                if winner_conf >= CONFIDENCE_THRESHOLD:
                    send_command(ser, winner_idx, winner_effort)
                else:
                    send_command(ser, 3, 0.0)   # Rest

                print_decision(winner_name, winner_conf, winner_effort,
                               vote_counts, total_confident, last_gesture)

                last_gesture = winner_name
                last_idx     = winner_idx
                vote_counts     = [0] * N_CLASSES
                total_confident = 0
                interval_start  = time.time()

    except serial.SerialException as e:
        print(f"\n[ERROR] {e}")
        print("  → Is the master ESP32 plugged in to COM13?")
        print("  → Is Arduino IDE Serial Monitor closed?")
    except KeyboardInterrupt:
        print(f"\n\n[STOPPED] {samples_seen} samples received.")
        try:
            send_command(ser, 3, 0.0)   # send Rest on exit
        except Exception:
            pass
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    run()
