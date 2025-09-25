import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .preprocess import rolling_smooth

def detect_steps(df: pd.DataFrame, q_col: str, time_col: str) -> Dict[str, Optional[int]]:
    x = df[time_col].values.astype(float)
    q = pd.to_numeric(df[q_col], errors="coerce").ffill().bfill().astype(float)
    qs = rolling_smooth(q, 5).values
    dq = np.gradient(qs, x)
    if not np.any(np.isfinite(dq)):
        return {"up":None, "down":None}
    thr = 0.25 * (np.nanmax(qs) - np.nanmin(qs) + 1e-9) / max(np.nanmax(x)-np.nanmin(x),1e-6)
    cand_up = np.where(dq>thr)[0]
    up = int(cand_up[0]) if cand_up.size>0 else None
    if up is not None:
        cand_down = np.where((np.arange(len(dq))>up+5) & (dq<-thr))[0]
    else:
        cand_down = np.where(dq<-thr)[0]
    down = int(cand_down[0]) if cand_down.size>0 else None
    if up is not None:
        w = slice(max(0,up-3), min(len(df), up+3))
        up = int(np.argmax(dq[w]) + max(0,up-3))
    if down is not None:
        w = slice(max(0,down-3), min(len(df), down+3))
        down = int(np.argmin(dq[w]) + max(0,down-3))
    return {"up":up, "down":down}

def classify_levels(df: pd.DataFrame, q_col: str) -> Tuple[float,float]:
    q = pd.to_numeric(df[q_col], errors="coerce").dropna().astype(float)
    if q.empty:
        return 0.0, 0.0
    v = np.sort(q.values)
    if len(v)<10:
        return float(q.min()), float(q.max())
    bins = 64
    hist, edges = np.histogram(v, bins=bins)
    p = hist.astype(float)/max(hist.sum(),1.0)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1]+edges[1:])/2.0)
    mu_t = mu[-1]
    sigma2 = (mu_t*omega - mu)**2 / (omega*(1.0-omega) + 1e-12)
    k = int(np.nanargmax(sigma2)) if np.isfinite(sigma2).any() else int(bins/2)
    t = (edges[k]+edges[k+1])/2.0
    low = float(np.mean(v[v<=t])) if np.any(v<=t) else float(v.min())
    high = float(np.mean(v[v>t])) if np.any(v>t) else float(v.max())
    return low, high

def build_phase_mask(n: int, up: Optional[int], down: Optional[int]) -> pd.Series:
    phase = np.array(["Low_Light"]*n, dtype=object)
    if up is not None:
        phase[up:] = "High_Light"
    if down is not None and down<n:
        phase[down:] = "Low_Light_2"
    return pd.Series(phase)
