import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.optimize import least_squares

def _increase_fun(t, tlag, tau1, tau2, f):
    z = np.clip(t - tlag, 0, None)
    return 1.0 - (f*np.exp(-z/np.maximum(tau1,1e-6)) + (1.0-f)*np.exp(-z/np.maximum(tau2,1e-6)))

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
        tlag, tau1, tau2, f, s = p
        f = 1.0/(1.0+np.exp(-f))
        s = 1.0/(1.0+np.exp(-s))
        m = gsmin + (amp*s)*_increase_fun(t, tlag, abs(tau1)+1e-6, abs(tau2)+1e-6, f)
        return m - y
    p0 = np.array([t[0], 1.0, 8.0, 0.0, 2.0])
    lb = np.array([t.min(), 1e-3, 1e-3, -5.0, -5.0])
    ub = np.array([t.max(), 60.0, 120.0, 5.0, 5.0])
    r = least_squares(res, p0, bounds=(lb,ub), max_nfev=2000)
    if not r.success:
        return {"ok":0}
    tlag, tau1, tau2, f, s = r.x
    f = 1.0/(1.0+np.exp(-f))
    s = 1.0/(1.0+np.exp(-s))
    yhat = gsmin + (amp*s)*_increase_fun(t, tlag, abs(tau1)+1e-6, abs(tau2)+1e-6, f)
    rmse = float(np.sqrt(np.nanmean((yhat - y)**2)))
    ss = float(np.nansum((y - np.nanmean(y))**2))
    r2 = float(1.0 - np.nansum((y - yhat)**2)/ss) if ss>0 else np.nan
    return {"ok":1, "tlag":float(tlag), "tau1":abs(float(tau1)), "tau2":abs(float(tau2)), "f":float(f), "scale":float(s), "rmse":rmse, "r2":r2}

def fit_step_decrease(t, y, gsmin, gsmax) -> Dict[str,float]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    amp = max(gsmax - gsmin, 0.0)
    if amp <= 1e-4:
        return {"ok":0}
    def res(p):
        tlag, tau1, tau2, f = p
        f = 1.0/(1.0+np.exp(-f))
        m = gsmax - amp*_decrease_fun(t, tlag, abs(tau1)+1e-6, abs(tau2)+1e-6, f)
        return m - y
    p0 = np.array([t[0], 1.0, 8.0, 0.0])
    lb = np.array([t.min(), 1e-3, 1e-3, -5.0])
    ub = np.array([t.max(), 60.0, 120.0, 5.0])
    r = least_squares(res, p0, bounds=(lb,ub), max_nfev=2000)
    if not r.success:
        return {"ok":0}
    tlag, tau1, tau2, f = r.x
    f = 1.0/(1.0+np.exp(-f))
    yhat = gsmax - amp*_decrease_fun(t, tlag, abs(tau1)+1e-6, abs(tau2)+1e-6, f)
    rmse = float(np.sqrt(np.nanmean((yhat - y)**2)))
    ss = float(np.nansum((y - np.nanmean(y))**2))
    r2 = float(1.0 - np.nansum((y - yhat)**2)/ss) if ss>0 else np.nan
    return {"ok":1, "tlag":float(tlag), "tau1":abs(float(tau1)), "tau2":abs(float(tau2)), "f":float(f), "rmse":rmse, "r2":r2}

