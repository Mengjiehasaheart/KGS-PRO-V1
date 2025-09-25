import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def _interp_time(t: np.ndarray, y: np.ndarray, target: float) -> float:
    for i in range(1, len(t)):
        y0, y1 = y[i-1], y[i]
        if (y0 <= target <= y1) or (y1 <= target <= y0):
            if y1 == y0:
                return float(t[i])
            r = (target - y0) / (y1 - y0)
            return float(t[i-1] + r * (t[i] - t[i-1]))
    return float('nan')

def percent_times(t: np.ndarray, y: np.ndarray, y0: float, y1: float, levels: List[float]) -> Dict[str, float]:
    out = {}
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    dy = (y1 - y0)
    if not np.isfinite(dy) or abs(dy) < 1e-9:
        return {f"t{int(p*100)}": float('nan') for p in levels}
    yn = (y - y0) / dy
    for p in levels:
        out[f"t{int(p*100)}"] = _interp_time(t, yn, p)
    return out

def deficit(t: np.ndarray, y: np.ndarray, ymax: float) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.maximum(ymax - y, 0.0)
    if len(t) < 2:
        return float('nan')
    return float(np.trapz(d, t))

def sl_approx(t: np.ndarray, A: np.ndarray, gs: np.ndarray, Amin: float, Amax: float, gsmin: float, gsmax: float) -> Tuple[float, float]:
    An = (A - Amin) / max(Amax - Amin, 1e-9)
    G = (gs - gsmin) / max(gsmax - gsmin, 1e-9)
    An = np.clip(An, 0, 2)
    G = np.clip(G, 1e-6, 2)
    S = 1.0 - (An / G)
    S = np.clip(S, 0.0, 1.0)
    return float(np.nanmax(S)), float(np.nanmean(S))

