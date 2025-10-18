"""
phone_oximetry_pipeline.py
端到端示例：
  1) 从 mp4 提取每帧 RGB 平均 -> 得到 time-series (R,G,B)
  2) 划 3s 窗口 (centered on each ground-truth sample) 形成样本 (3 x 90)
  3) 标准化、构造 PyTorch Dataset
  4) 简单 CNN (3 conv + 2 fc) 训练 / 评估（演示用）
注意：用真实科研/临床数据时，需按论文做更严格的 LOOCV、超参搜索与 QC。
参考：Hoffman et al., "Smartphone camera oximetry..." (npj Digital Medicine, 2022). :contentReference[oaicite:6]{index=6}
"""

import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------------------
# 1) 从视频提取 per-frame RGB mean
# -----------------------------
def extract_rgb_means_from_video(video_path, downsize=(176,144), show_progress=False):
    """
    返回:
      rgb_means: ndarray shape (n_frames, 3)  按列为 R,G,B mean
      fps: frames per second
      frame_timestamps: ndarray (n_frames,) 时间戳(秒) 从 0 开始
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rgb_list = []
    timestamps = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if downsize is not None:
            frame = cv2.resize(frame, downsize)
        # OpenCV 默认 BGR -> 转成 RGB
        r_mean = frame[:,:,2].mean()
        g_mean = frame[:,:,1].mean()
        b_mean = frame[:,:,0].mean()
        rgb_list.append([r_mean, g_mean, b_mean])
        timestamps.append(frame_idx / fps)
        frame_idx += 1
    cap.release()
    rgb_means = np.array(rgb_list)  # shape (n,3)
    frame_timestamps = np.array(timestamps)
    return rgb_means, fps, frame_timestamps

# -----------------------------
# 2) 将 RGB 序列划窗成 3s (论文中用 3s @30fps => 90 帧)
#    并与 ground truth 对齐（ground_truth: DataFrame with columns ['time','spO2']）
# -----------------------------
def make_samples_from_rgb(rgb_means, frame_timestamps, gt_df, sample_duration_s=3.0, fps=30.0):
    """
    输入：
      rgb_means: (n_frames, 3)
      frame_timestamps: (n_frames,)
      gt_df: pandas.DataFrame with 'time' (seconds) and 'spo2' columns. times should be aligned with video times.
    输出:
      X: ndarray (n_samples, 3, T)  T = int(sample_duration_s*fps)
      y: ndarray (n_samples,)   ground truth SPO2 at sample center
      sample_times: center times
    说明：
      对每个 GT 时间 t, 取以 t 为中心的 sample_duration_s 长度窗口；如果边界溢出，则跳过该样本。
    """
    T = int(round(sample_duration_s * fps))
    half_T = T // 2
    n_frames = rgb_means.shape[0]
    X_list = []
    y_list = []
    times_list = []
    for _, row in gt_df.iterrows():
        t = float(row['time'])
        # 找到 closest frame index to time t
        center_idx = np.argmin(np.abs(frame_timestamps - t))
        start = center_idx - half_T
        end = start + T
        if start < 0 or end > n_frames:
            continue
        window = rgb_means[start:end, :].T  # shape (3, T)
        X_list.append(window.astype(np.float32))
        y_list.append(float(row['spo2']))
        times_list.append(t)
    if len(X_list) == 0:
        return np.empty((0,3,T)), np.empty((0,)), []
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y, times_list

# -----------------------------
# 3) 简单 preprocessing: detrend / bandpass optional & channel-wise standardize
# -----------------------------
def bandpass_signal(x, fs, low=0.5, high=5.0, order=3):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def preprocess_X(X, fps, apply_bandpass=False):
    # X: (N, 3, T)
    X_proc = X.copy()
    N, C, T = X_proc.shape
    for i in range(N):
        for c in range(C):
            sig = X_proc[i,c,:]
            if apply_bandpass:
                try:
                    sigf = bandpass_signal(sig, fs=fps, low=0.5, high=5.0)
                except Exception:
                    sigf = sig
                X_proc[i,c,:] = sigf
            # remove mean (DC) but keep scale
            X_proc[i,c,:] = sig - np.mean(sig)
    # channel-wise standardization across dataset
    ch_means = X_proc.mean(axis=(0,2), keepdims=True)  # shape (1,C,1)
    ch_stds  = X_proc.std(axis=(0,2), keepdims=True) + 1e-8
    X_proc = (X_proc - ch_means) / ch_stds
    return X_proc, ch_means.squeeze(), ch_stds.squeeze()

# -----------------------------
# 4) PyTorch Dataset & model (simple 2D conv-ish via Conv1d over time with 3 channels)
#    Paper used 3 conv layers + 2 linear layers; 这里实现一个同等复杂度的网络
# -----------------------------
class OximetryDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 3, T)
        self.X = torch.from_numpy(X)  # float32
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, seq_len=90):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), # preserve length
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/4
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (batch,128,1)
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # regression
        )
    def forward(self, x):
        # x: (batch, 3, T)
        return self.net(x).squeeze(1)

# -----------------------------
# 5) 训练与评估函数（简单 demo）
# -----------------------------
def train_eval(X, y, fps, epochs=30, batch_size=32, lr=1e-4, val_split=0.2, device='cpu'):
    Xp, ch_means, ch_stds = preprocess_X(X, fps, apply_bandpass=False)
    ds = OximetryDataset(Xp, y)
    n_val = int(len(ds)*val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = SimpleCNN(in_channels=3, seq_len=X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= n_train
        # val
        model.eval()
        val_loss = 0.0
        preds = []
        gts = []
        with torch.no_grad():
            for xb, yb in va:
                xb=xb.to(device); yb=yb.to(device)
                p = model(xb)
                val_loss += loss_fn(p,yb).item() * xb.size(0)
                preds.append(p.cpu().numpy())
                gts.append(yb.cpu().numpy())
        val_loss /= max(1, n_val)
        preds = np.concatenate(preds) if len(preds)>0 else np.array([])
        gts   = np.concatenate(gts) if len(gts)>0 else np.array([])
        if epoch % 5 == 0 or epoch==epochs-1:
            print(f"Epoch {epoch:03d} train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")
    # final predictions on val
    return model, (ch_means, ch_stds), (preds, gts)

# -----------------------------
# 6) 示例主流程（用户需替换 video_path 与 gt_csv_path）
# -----------------------------
if __name__ == "__main__":
    # === USER INPUTS: 修改这两个路径到你的文件 ===
    video_path = "finger.mp4"      # 或多视频循环调用 extract_rgb_means_from_video 后拼接
    gt_csv_path = "ground_truth_spo2.csv"   # csv columns: time (sec), spo2 (percent)
    # ==================================================

    if not os.path.exists(video_path):
        print("请把 mp4 放在脚本同目录，或修改 video_path。示例结束。")
    else:
        print("提取视频每帧 RGB mean ...")
        rgb_means, fps, frame_ts = extract_rgb_means_from_video(video_path, downsize=(176,144))
        print("frames:", rgb_means.shape[0], "fps:", fps)

        # load GT csv
        if not os.path.exists(gt_csv_path):
            # 如果没有 ground truth，示范使用合成（仅用于演示）
            print("未找到 ground truth CSV，生成模拟 gt（仅用于 demo）")
            # generate synthetic gt: constant 97% with slow sinusoidal variation
            times = frame_ts[::30]  # 每秒一个 gt 假设
            spo2 = 97 + 2.0 * np.sin(0.01 * np.arange(len(times)))
            gt_df = pd.DataFrame({'time': times, 'spo2': spo2})
        else:
            gt_df = pd.read_csv(gt_csv_path)
            # 确保列名为 'time' 和 'spo2'
            assert 'time' in gt_df.columns and 'spo2' in gt_df.columns

        # make samples (3s window centered on gt time)
        X, y, sample_times = make_samples_from_rgb(rgb_means, frame_ts, gt_df, sample_duration_s=3.0, fps=fps)
        print("samples:", X.shape, "labels:", y.shape)

        if len(y)==0:
            print("没有可用样本（可能视频太短或 gt 与视频不同步）。请检查对齐。")
        else:
            # train / eval
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model, stats, (preds, gts) = train_eval(X, y, fps=fps, epochs=40, batch_size=32, device=device)
            if len(preds)>0:
                # 画图：部分 val 预测 vs GT
                plt.figure(figsize=(8,4))
                plt.plot(gts, label='GT SpO2')
                plt.plot(preds, label='Pred (val)')
                plt.legend(); plt.title("Validation: GT vs Pred (demo)")
                plt.show()

