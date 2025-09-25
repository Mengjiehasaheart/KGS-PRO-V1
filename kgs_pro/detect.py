import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .preprocess import rolling_smooth

def detect_steps(df: pd.DataFrame, q_col: str, time_col: str) -> Dict[str, Optional[int]]:
    x = df[time_col].values.astype(float)
    q = pd.to_numeric(df[q_col], errors="coerce").ffill().bfill().astype(float).values
    if len(x) < 5:
        return {"up": None, "down": None}
    dt = np.median(np.diff(x)) if len(x) > 1 else 0.1
    w = int(max(5, round(0.5 / max(dt, 1e-6))))
    qs = rolling_smooth(pd.Series(q), w).values
    low, high = classify_levels(pd.DataFrame({q_col: qs}), q_col)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        dq = np.gradient(qs, x)
        up = int(np.argmax(dq)) if np.isfinite(dq).any() else None
        down = int(np.argmin(dq[up+5:]) + up + 5) if up is not None and np.isfinite(dq[up+5:]).any() else None
        return {"up": up, "down": down}
    mid = (low + high) / 2.0
    tol = 0.1 * (high - low)
    pre_n = int(max(5, round(0.5 / max(dt, 1e-6))))
    post_n = pre_n
    up = None
    for i in range(1, len(qs)):
        if qs[i-1] < mid <= qs[i]:
            ip = max(0, i - pre_n)
            im = min(len(qs), i + post_n)
            pre_med = float(np.nanmedian(qs[ip:i])) if i - ip > 0 else np.nan
            post_med = float(np.nanmedian(qs[i:im])) if im - i > 0 else np.nan
            if np.isfinite(pre_med) and np.isfinite(post_med) and pre_med <= low + tol and post_med >= high - tol:
                up = i
                break
    down = None
    if up is not None:
        for j in range(up + 1, len(qs)):
            if qs[j-1] > mid >= qs[j]:
                jp = max(up + 1, j - pre_n)
                jm = min(len(qs), j + post_n)
                pre_med = float(np.nanmedian(qs[jp:j])) if j - jp > 0 else np.nan
                post_med = float(np.nanmedian(qs[j:jm])) if jm - j > 0 else np.nan
                if np.isfinite(pre_med) and np.isfinite(post_med) and pre_med >= high - tol and post_med <= low + tol:
                    down = j
                    break
    if up is None:
        dq = np.gradient(qs, x)
        up = int(np.argmax(dq)) if np.isfinite(dq).any() else None
        down = int(np.argmin(dq[up+5:]) + up + 5) if up is not None and np.isfinite(dq[up+5:]).any() else None
    return {"up": up, "down": down}

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
