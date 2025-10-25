# phone_oximetry_v3_3.py
# Smartphone camera oximetry pipeline — robust, debug-stable, auto time align
# Author: GPT-5 (2025)

import os, glob, math, copy
import numpy as np
import pandas as pd
import cv2
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- CONFIG ----------------
LEFT_VIDEOS_DIR  = r"D:\Mobile\raw\Left"
RIGHT_VIDEOS_DIR = r"D:\Mobile\raw\Right"
GT_DIR           = r"D:\Mobile\Ground truth"
OUT_DIR          = r"D:\Mobile\results_phone_oximetry"
FPS_EXPECTED     = 30.0
WINDOW_S         = 3.0
BANDPASS_LOW     = 0.7
BANDPASS_HIGH    = 3.0
EPOCHS           = 80
BATCH_SIZE       = 32
LR               = 3e-4
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MAX_OFFSET_SEC   = 600  # ±10 minutes
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- UTILITIES ----------
def butter_bandpass(x, fs, low, high, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    try:
        return filtfilt(b, a, x)
    except:
        return x

def hilbert_env(x):
    try:
        return np.abs(hilbert(x))
    except:
        return np.abs(x)

# ---------- Extract RGB from video ----------
def extract_rgb_means_from_video(video_path, downsize=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️  Can't open video: {video_path}")
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_EXPECTED
    rgb, ts = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if downsize:
            frame = cv2.resize(frame, downsize)
        h, w, _ = frame.shape
        # ✅ use central ROI to improve SNR
        roi = frame[h//2-100:h//2+100, w//2-100:w//2+100, :]
        r, g, b = roi[:,:,2].mean(), roi[:,:,1].mean(), roi[:,:,0].mean()
        rgb.append([r,g,b])
        ts.append(idx/fps)
        idx += 1
    cap.release()
    if len(rgb) == 0:
        print(f"⚠️  No frames extracted from {video_path}")
        return None, None, None
    return np.array(rgb), fps, np.array(ts)

# ---------- Parse Ground Truth ----------
def parse_gt_csv(path):
    df = pd.read_csv(path)
    # locate time column
    time_col = None
    for c in df.columns:
        if "time" in str(c).strip().lower():
            time_col = c; break
    if not time_col:
        raise ValueError("No time column found")

    # detect spo2 columns
    spo2_cols = [c for c in df.columns if "spo2" in str(c).lower()]
    if not spo2_cols:
        raise ValueError("No SpO2 columns found")

    # clean and parse
    df['spo2'] = df[spo2_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    df['time'] = df[time_col].astype(str).str.strip()
    df['time'] = pd.to_datetime(df['time'], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=['time','spo2']).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid rows after parsing time")

    # relative seconds
    df['time_rel'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

    print(f"  ✅ Parsed {os.path.basename(path)}: start={df['time'].iloc[0].time()}, "
          f"end={df['time'].iloc[-1].time()}, range={df['time_rel'].iloc[-1]-df['time_rel'].iloc[0]:.1f}s")

    return df[['time_rel','spo2']].rename(columns={'time_rel':'time'})

# ---------- Offset detection ----------
def align_time_offset(env, env_t, gt_y, gt_t, max_off=600):
    env = (env - env.mean()) / (env.std() + 1e-8)
    gt_y = (gt_y - gt_y.mean()) / (gt_y.std() + 1e-8)
    step = 1.0
    offsets = np.arange(-max_off, max_off, step)
    best_corr, best_off = -np.inf, 0
    for off in offsets:
        gt_shift = gt_t + off
        y_interp = np.interp(env_t, gt_shift, gt_y, left=np.nan, right=np.nan)
        valid = ~np.isnan(y_interp)
        if valid.sum() < 20: continue
        corr = np.corrcoef(env[valid], y_interp[valid])[0,1]
        if not np.isnan(corr) and corr > best_corr:
            best_corr, best_off = corr, off
    return best_off

def make_windows(rgb, ts, gt, offset, dur=3.0, fps=30.0):
    T = int(round(dur*fps)); half=T//2
    X,y = [],[]
    for t_gt,s in zip(gt['time']+offset, gt['spo2']):
        idx = np.argmin(np.abs(ts-t_gt))
        st,ed = idx-half, idx+half
        if st<0 or ed>len(ts): continue
        X.append(rgb[st:ed,:].T); y.append(s)
    if not X: return np.zeros((0,3,T)), np.zeros((0,))
    return np.stack(X).astype(np.float32), np.array(y).astype(np.float32)

# ---------- CNN ----------
class CNN_Oximetry(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> T/2

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> T/4

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (batch,128,1)

            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # input (batch, 1, 3, T) → reshape to (batch, 3, T)
        x = x.squeeze(1)
        return self.net(x).squeeze(1)


# ---------- helpers ----------
def standardize(X):
    m,s = X.mean((0,2),keepdims=True), X.std((0,2),keepdims=True)+1e-8
    return (X-m)/s, m, s

def bland_altman(y_true,y_pred,outpath):
    diff=y_pred-y_true; mean=(y_pred+y_true)/2
    md,sd=np.mean(diff),np.std(diff,ddof=1)
    plt.figure(figsize=(5,4))
    plt.scatter(mean,diff,s=6)
    plt.axhline(md,color='r',label=f"mean={md:.2f}")
    plt.axhline(md+1.96*sd,color='b',ls='--')
    plt.axhline(md-1.96*sd,color='b',ls='--')
    plt.xlabel("Mean SpO2"); plt.ylabel("Pred-GT"); plt.legend()
    plt.tight_layout(); plt.savefig(outpath); plt.close()
    return md,md-1.96*sd,md+1.96*sd

def visualize_offset(env, env_t, gt, off, outpath):
    plt.figure(figsize=(8,3))
    gt_norm = (gt['spo2'] - gt['spo2'].min()) / (gt['spo2'].max()-gt['spo2'].min()+1e-6)
    plt.plot(env_t, (env-env.min())/(env.max()-env.min()+1e-6), label='PPG env')
    plt.plot(gt['time'], gt_norm, label='GT (raw)')
    plt.plot(gt['time']+off, gt_norm, '--', label=f'GT shifted {off:.1f}s')
    plt.legend(); plt.title(f"Offset alignment visualization ({off:.1f}s)")
    plt.xlabel("Time (s)"); plt.tight_layout(); plt.savefig(outpath); plt.close()

# ---------- main pipeline ----------
def process_hand(hand,dir_v):
    out_sub=os.path.join(OUT_DIR,hand)
    os.makedirs(out_sub,exist_ok=True)
    print(f"\n--- Processing {hand} hand ---")
    vids=sorted(glob.glob(os.path.join(dir_v,"*.mp4")))
    for v in vids:
        sid=os.path.splitext(os.path.basename(v))[0][1:]
        gt_path=os.path.join(GT_DIR,f"{sid}.csv")
        if not os.path.exists(gt_path):
            print(f"  GT missing for {sid}"); continue
        try:
            gt=parse_gt_csv(gt_path)
        except Exception as e:
            print(f"  Skip {sid}: {e}"); continue
        rgb,fps,ts=extract_rgb_means_from_video(v)
        if rgb is None:
            print(f"  Skip {sid} (video not readable)"); continue
        print(f"  Video {sid}: length={len(ts)/fps:.1f}s, fps={fps:.1f}")
        # ---------- 可视化原始RGB信号 (PPG波形) ----------
        plt.figure(figsize=(10, 4))
        plt.plot(ts, rgb[:, 0], color='r', label='Red channel')
        plt.plot(ts, rgb[:, 1], color='g', label='Green channel')
        plt.plot(ts, rgb[:, 2], color='b', label='Blue channel')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean intensity')
        plt.title(f'Raw RGB PPG signal - {hand} {sid}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_sub, f'PPG_Raw_{hand}_{sid}.png'))
        plt.close()
        g=butter_bandpass(rgb[:,1],fps,BANDPASS_LOW,BANDPASS_HIGH)
        env = hilbert_env(g)
        plt.figure(figsize=(10, 4))
        plt.plot(ts, g, color='lime', alpha=0.6, label='Filtered Green (0.7–3 Hz)')
        plt.plot(ts, env, color='k', linewidth=1.5, label='Hilbert envelope')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized amplitude')
        plt.title(f'Filtered PPG & Envelope - {hand} {sid}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_sub, f'PPG_Filtered_{hand}_{sid}.png'))
        plt.close()
        env=hilbert_env(g)
        off=align_time_offset(env,ts,gt['spo2'].values,gt['time'].values,MAX_OFFSET_SEC)
        visualize_offset(env,ts,gt,off,os.path.join(out_sub,f"OffsetCheck_{hand}_{sid}.png"))
        X,y=make_windows(rgb,ts,gt,off,WINDOW_S,FPS_EXPECTED)
        if len(y)==0:
            print(f"  No valid windows for {sid}"); continue
        for i in range(X.shape[0]):
            for c in range(3): X[i,c]=butter_bandpass(X[i,c],FPS_EXPECTED,BANDPASS_LOW,BANDPASS_HIGH)
        Xs,m,s=standardize(X)
        Xt=torch.tensor(Xs).unsqueeze(1);Yt=torch.tensor(y)
        n=len(Yt); nv=max(1,int(0.1*n))
        train=DataLoader(TensorDataset(Xt[:-nv],Yt[:-nv]),BATCH_SIZE,True)
        val=DataLoader(TensorDataset(Xt[-nv:],Yt[-nv:]),BATCH_SIZE)
        model=CNN_Oximetry(X.shape[2]).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        lossfn=nn.SmoothL1Loss()
        best,bestv=None,1e9
        for ep in range(EPOCHS):
            model.train(); tl=[]
            for xb,yb in train:
                xb,yb=xb.to(DEVICE),yb.to(DEVICE)
                l=lossfn(model(xb),yb)
                opt.zero_grad(); l.backward(); opt.step()
                tl.append(l.item())
            model.eval(); vl=[]
            with torch.no_grad():
                for xb,yb in val:
                    xb,yb=xb.to(DEVICE),yb.to(DEVICE)
                    vl.append(lossfn(model(xb),yb).item())
            vm=np.mean(vl)
            if vm<bestv: bestv=vm; best=copy.deepcopy(model.state_dict())
        if best: model.load_state_dict(best)
        preds,gts=[],[]
        model.eval()
        with torch.no_grad():
            for xb,yb in val:
                xb=xb.to(DEVICE)
                preds.append(model(xb).cpu().numpy())
                gts.append(yb.numpy())
        preds,gts=np.concatenate(preds),np.concatenate(gts)
        mae=mean_absolute_error(gts,preds)
        rmse=math.sqrt(mean_squared_error(gts,preds))
        md,lo,hi=bland_altman(gts,preds,os.path.join(out_sub,f"BA_{hand}_{sid}.png"))
        plt.plot(gts,label='GT'); plt.plot(preds,label='Pred')
        plt.legend(); plt.title(f"{hand} {sid} | MAE={mae:.2f} | off={off:.1f}s")
        plt.tight_layout(); plt.savefig(os.path.join(out_sub,f"GTvsPred_{hand}_{sid}.png")); plt.close()
        pd.DataFrame([{'subject':sid,'mae':mae,'rmse':rmse,'mean_diff':md,'loa_lo':lo,'loa_hi':hi,'offset_s':off}]).to_csv(
            os.path.join(out_sub,"metrics_"+hand+".csv"),mode='a',index=False,header=not os.path.exists(os.path.join(out_sub,"metrics_"+hand+".csv"))
        )
        print(f"  {hand} {sid}: MAE={mae:.2f}, offset={off:.1f}s")
    print(f"Finished {hand}. Results in {out_sub}")

if __name__=="__main__":
    process_hand("Left",LEFT_VIDEOS_DIR)
    process_hand("Right",RIGHT_VIDEOS_DIR)
    print("✅ All done. Results saved in:",OUT_DIR)
