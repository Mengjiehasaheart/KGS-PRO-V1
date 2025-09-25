import pandas as pd
import numpy as np
from typing import Optional, Tuple

def parse_hhmmss_to_minutes(series: pd.Series) -> Optional[pd.Series]:
    try:
        s = pd.to_datetime(series.astype(str), errors="coerce")
        if s.isna().all():
            return None
        t0 = s.dropna().iloc[0]
        m = (s - t0).dt.total_seconds() / 60.0
        return m
    except Exception:
        return None

def infer_time(df: pd.DataFrame, fallback_dt_s: Optional[float]=None) -> Tuple[pd.DataFrame, str]:
    d = df.copy()
    if "time_min" in d.columns:
        return d, "time_min"
    if "hhmmss" in d.columns:
        m = parse_hhmmss_to_minutes(d["hhmmss"])
        if m is not None:
            d["time_min"] = m
            return d, "time_min"
    for c in ["elapsed","elapsed_s","time_s","seconds","sec","t"]:
        if c in d.columns:
            try:
                x = pd.to_numeric(d[c], errors="coerce").astype(float)
                d["time_min"] = x/60.0
                return d, "time_min"
            except Exception:
                pass
    if "obs" in d.columns:
        x = pd.to_numeric(d["obs"], errors="coerce")
        if x.notna().sum()>=2:
            idx = x.dropna().index
            if len(idx)>=2:
                di = np.diff(idx.values).astype(float)
                if np.all(di>0):
                    if fallback_dt_s is None:
                        fallback_dt_s = 1.0
                    t = np.arange(len(d)) * float(fallback_dt_s) / 60.0
                    d["time_min"] = t
                    return d, "time_min"
    if fallback_dt_s is not None:
        t = np.arange(len(d)) * float(fallback_dt_s) / 60.0
        d["time_min"] = t
        return d, "time_min"
    d["time_min"] = np.arange(len(d), dtype=float)
    return d, "time_min"

def rolling_smooth(x: pd.Series, window: int=5) -> pd.Series:
    w = max(1,int(window))
    if w<=1:
        return x
    return x.rolling(w, center=True, min_periods=1).median()

