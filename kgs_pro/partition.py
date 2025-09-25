import numpy as np
import pandas as pd
from typing import Dict

def _arrhenius(Tk: float, Ea: float) -> float:
    R = 8.314
    return float(np.exp(Ea*(Tk-298.15)/(R*Tk*298.15)))

def _peaked(Tk: float, k25: float, Ea: float, Hd: float, S: float) -> float:
    R = 8.314
    num = k25 * np.exp(Ea*(Tk-298.15)/(R*Tk*298.15))
    den = 1.0 + np.exp((S*Tk - Hd)/(R*Tk))
    den25 = 1.0 + np.exp((S*298.15 - Hd)/(R*298.15))
    return float(num * den25 / den)

def _kinetics(Tleaf_C: float) -> Dict[str,float]:
    Tk = float(Tleaf_C) + 273.15
    Gamma25 = 42.75
    Kc25 = 404.9
    Ko25 = 278400.0
    EaG = 37830.0
    EaKc = 79430.0
    EaKo = 36380.0
    Gamma = Gamma25 * _arrhenius(Tk, EaG)
    Kc = Kc25 * _arrhenius(Tk, EaKc)
    Ko = Ko25 * _arrhenius(Tk, EaKo)
    return {"Gamma":Gamma, "Kc":Kc, "Ko":Ko, "O":210000.0, "Tk":Tk}

def _J(PPFD: float, Jmax: float, alpha: float, theta: float) -> float:
    I = max(PPFD, 0.0)
    a = alpha*I + Jmax
    b = 4.0*theta*alpha*I*Jmax
    d = max(a*a - b, 0.0)
    return (a - d**0.5)/(2.0*theta)

def A_model(Ci: float, PPFD: float, Vc25: float, J25: float, Rd: float, Tleaf_C: float, Ca: float, alpha: float, theta: float, use_temp: bool=True) -> float:
    if use_temp:
        kin = _kinetics(Tleaf_C)
        Vc = _peaked(kin["Tk"], Vc25, 58550.0, 200000.0, 650.0)
        Jm = _peaked(kin["Tk"], J25, 37000.0, 200000.0, 650.0)
        Gamma = kin["Gamma"]
        Kc = kin["Kc"]
        Ko = kin["Ko"]
        O = kin["O"]
    else:
        Vc = Vc25
        Jm = J25
        Gamma = 42.75
        Kc = 404.9
        Ko = 278400.0
        O = 210000.0
    Ci = max(Ci, 1e-6)
    Ac = Vc * (Ci - Gamma) / (Ci + Kc*(1.0 + O/Ko))
    Jv = _J(PPFD, Jm, alpha, theta)
    Aj = Jv * (Ci - Gamma) / (4.0*Ci + 8.0*Gamma)
    A = min(Ac, Aj) - Rd
    return float(A)

def calibrate_params(df: pd.DataFrame, time_col: str, q_col: str, pts_avg: int=10, ratio_jv: float=1.8, alpha: float=0.24, theta: float=0.9, use_temp: bool=True, rd_override: float=None, vcmax25_override: float=None, jmax25_override: float=None) -> Dict[str,float]:
    t = pd.to_numeric(df[time_col], errors="coerce").astype(float).values
    A = pd.to_numeric(df.get("A", pd.Series([np.nan]*len(df))), errors="coerce").astype(float).values
    Ci = pd.to_numeric(df.get("Ci", pd.Series([np.nan]*len(df))), errors="coerce").astype(float).values
    Ca = pd.to_numeric(df.get("Ca", pd.Series([400.0]*len(df))), errors="coerce").astype(float).values
    Q = pd.to_numeric(df[q_col], errors="coerce").astype(float).values
    up = 0
    if (Q.max()-Q.min())>1.0:
        dq = np.gradient(Q, t)
        cu = np.where(dq>0)[0]
        up = int(cu[0]) if cu.size>0 else 0
    hi = len(df)
    A0 = float(pd.Series(A[max(0,up-pts_avg):up]).dropna().mean()) if up>0 else float(np.nan)
    Rd = float(rd_override) if rd_override is not None else (float(abs(min(A0, 0.5))) if np.isfinite(A0) else 1.0)
    A1 = float(pd.Series(A[max(0,hi-pts_avg):hi]).dropna().mean())
    Ci1 = float(pd.Series(Ci[max(0,hi-pts_avg):hi]).dropna().mean()) if np.isfinite(Ci).any() else 280.0
    Q1 = float(pd.Series(Q[max(0,hi-pts_avg):hi]).dropna().mean())
    Ca1 = float(pd.Series(Ca[max(0,hi-pts_avg):hi]).dropna().mean())
    rJ = float(ratio_jv)
    T1 = float(pd.to_numeric(df.get("Tleaf", pd.Series([25]*len(df))), errors="coerce").fillna(25.0).iloc[-1])
    def f(vc):
        jm = rJ*vc
        return A_model(Ci1, Q1, vc, jm, Rd, T1, Ca1, alpha, theta, use_temp) - A1
    if vcmax25_override is not None or jmax25_override is not None:
        if vcmax25_override is not None and jmax25_override is not None:
            Vc = float(vcmax25_override)
            Jm = float(jmax25_override)
        elif vcmax25_override is not None:
            Vc = float(vcmax25_override)
            Jm = float(rJ*Vc)
        else:
            Jm = float(jmax25_override)
            Vc = float(Jm/rJ)
        return {"Vcmax25":Vc, "Jmax25":Jm, "Rd":Rd, "Ca":Ca1, "Qhi":Q1, "alpha":float(alpha), "theta":float(theta), "use_temp": bool(use_temp)}
    lo, hi_v = 10.0, 200.0
    v = 60.0
    for _ in range(40):
        fv = f(v)
        dv = (f(v*1.05)-fv)/(v*0.05)
        if not np.isfinite(dv) or abs(dv)<1e-9:
            break
        v = max(lo, min(hi_v, v - fv/dv))
    Vc = float(max(lo, min(hi_v, v)))
    Jm = float(rJ*Vc)
    return {"Vcmax25":Vc, "Jmax25":Jm, "Rd":Rd, "Ca":Ca1, "Qhi":Q1, "alpha":float(alpha), "theta":float(theta), "use_temp": bool(use_temp)}

def partition_limits(df: pd.DataFrame, time_col: str, q_col: str, pars: Dict[str,float]) -> pd.DataFrame:
    A = pd.to_numeric(df.get("A", pd.Series([np.nan]*len(df))), errors="coerce").astype(float).values
    Ci = pd.to_numeric(df.get("Ci", pd.Series([np.nan]*len(df))), errors="coerce").astype(float).values
    Ca = pd.to_numeric(df.get("Ca", pd.Series([pars.get("Ca",400.0)]*len(df))), errors="coerce").fillna(pars.get("Ca",400.0)).astype(float).values
    Q = pd.to_numeric(df[q_col], errors="coerce").astype(float).values
    T = pd.to_numeric(df.get("Tleaf", pd.Series([25.0]*len(df))), errors="coerce").astype(float).values
    Vc25 = pars["Vcmax25"]
    J25 = pars["Jmax25"]
    Rd = pars["Rd"]
    alpha = float(pars.get("alpha",0.24))
    theta = float(pars.get("theta",0.9))
    use_temp = bool(pars.get("use_temp", True))
    A_meas = A
    A_pot = np.array([A_model(Ci[i] if np.isfinite(Ci[i]) else Ca[i], Q[i], Vc25, J25, Rd, T[i], Ca[i], alpha, theta, use_temp) for i in range(len(A))])
    A_max = np.array([A_model(Ca[i], float(np.nanmax(Q)), Vc25, J25, Rd, T[i], Ca[i], alpha, theta, use_temp) for i in range(len(A))])
    denom = np.where(np.isfinite(A_max), A_max, np.nan)
    SL = (A_pot - A_meas)/denom
    BL = (A_max - A_pot)/denom
    TL = (A_max - A_meas)/denom
    out = df.copy()
    out["A_pot"] = A_pot
    out["A_max"] = A_max
    out["SL"] = np.clip(SL, 0.0, 1.0)
    out["BL"] = np.clip(BL, 0.0, 1.0)
    out["TL"] = np.clip(TL, 0.0, 1.0)
    return out
