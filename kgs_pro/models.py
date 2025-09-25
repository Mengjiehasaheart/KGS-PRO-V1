import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.optimize import least_squares

def _increase_fun_single(t, tlag, tau):
    z = np.clip(t - tlag, 0, None)
    return 1.0 - np.exp(-z/np.maximum(tau,1e-6))

def _decrease_fun(t, tlag, tau1, tau2, f):
    z = np.clip(t - tlag, 0, None)
    return f*np.exp(-z/np.maximum(tau1,1e-6)) + (1.0-f)*np.exp(-z/np.maximum(tau2,1e-6))

def fit_step_increase(t, y, gsmin, gsmax) -> Dict[str,float]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    amp = max(gsmax - gsmin, 0.0)
    if amp <= 1e-4:
        return {"ok":0}
    def res(p):
        tlag, tau = p
        m = gsmin + amp*_increase_fun_single(t, tlag, abs(tau)+1e-6)
        return m - y
    p0 = np.array([t[0], 8.0])
    lb = np.array([t.min(), 0.05])
    ub = np.array([t.max(), 300.0])
    r = least_squares(res, p0, bounds=(lb,ub), max_nfev=2000)
    if not r.success:
        return {"ok":0}
    tlag, tau = r.x
    yhat = gsmin + amp*_increase_fun_single(t, tlag, abs(tau)+1e-6)
    rmse = float(np.sqrt(np.nanmean((yhat - y)**2)))
    ss = float(np.nansum((y - np.nanmean(y))**2))
    r2 = float(1.0 - np.nansum((y - yhat)**2)/ss) if ss>0 else np.nan
    return {"ok":1, "tlag":float(tlag), "tau":abs(float(tau)), "rmse":rmse, "r2":r2}

def fit_step_decrease(t, y, gsmin, gsmax) -> Dict[str,float]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    amp = max(gsmax - gsmin, 0.0)
    if amp <= 1e-4:
        return {"ok":0}
    def res(p):
        tlag, tau = p
        m = gsmin + amp*np.exp(-(np.clip(t - tlag, 0, None))/np.maximum(tau,1e-6))
        return m - y
    p0 = np.array([t[0], 8.0])
    lb = np.array([t.min(), 0.05])
    ub = np.array([t.max(), 300.0])
    r = least_squares(res, p0, bounds=(lb,ub), max_nfev=2000)
    if not r.success:
        return {"ok":0}
    tlag, tau = r.x
    yhat = gsmin + amp*np.exp(-(np.clip(t - tlag, 0, None))/np.maximum(tau,1e-6))
    rmse = float(np.sqrt(np.nanmean((yhat - y)**2)))
    ss = float(np.nansum((y - np.nanmean(y))**2))
    r2 = float(1.0 - np.nansum((y - yhat)**2)/ss) if ss>0 else np.nan
    return {"ok":1, "tlag":float(tlag), "tau":abs(float(tau)), "rmse":rmse, "r2":r2}
