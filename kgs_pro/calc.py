import pandas as pd
import numpy as np
from typing import Optional

def _leaf_area_m2(df: pd.DataFrame, default_cm2: float=6.0) -> float:
    for c in ["Leaf_Area","LeafArea","Area","ChArea","Ch_Area","Area_cm2"]:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v)>0:
                a = float(v.iloc[-1])
                if a>0:
                    return a/10000.0
    return float(default_cm2)/10000.0

def _flow_umol_s(df: pd.DataFrame, default_umol_s: float=500.0) -> float:
    if "Flow" in df.columns:
        v = pd.to_numeric(df["Flow"], errors="coerce").dropna()
        if len(v)>0:
            return float(v.median())
    return float(default_umol_s)

def _estimate_gbw(flow_umol_s: float) -> float:
    return 1.37 * np.sqrt(max(flow_umol_s,1.0)/500.0)

def _is_all_zero_or_nan(x: pd.Series) -> bool:
    v = pd.to_numeric(x, errors="coerce")
    return v.isna().all() or float(v.abs().max())==0.0

def ensure_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    S = _leaf_area_m2(d)
    F = _flow_umol_s(d)
    if ("H2O_s" in d.columns and "H2O_a" in d.columns) and ("E" not in d.columns or _is_all_zero_or_nan(d["E"])):
        w = pd.to_numeric(d["H2O_s"], errors="coerce") - pd.to_numeric(d["H2O_a"], errors="coerce")
        Emmol = (F * w) * 1e-6 / max(S,1e-9)
        d["E"] = Emmol
    if ("CO2_r" in d.columns and "CO2_s" in d.columns) and ("A" not in d.columns or _is_all_zero_or_nan(d["A"])):
        cdiff = pd.to_numeric(d["CO2_r"], errors="coerce") - pd.to_numeric(d["CO2_s"], errors="coerce")
        A = (F * cdiff) / (1e6 * max(S,1e-9))
        d["A"] = A
    if ("H2O_s" in d.columns and "H2O_a" in d.columns) and ("gtw" not in d.columns or _is_all_zero_or_nan(d["gtw"])):
        if "E" in d.columns and not _is_all_zero_or_nan(d["E"]):
            E = pd.to_numeric(d["E"], errors="coerce")
        else:
            w = pd.to_numeric(d["H2O_s"], errors="coerce") - pd.to_numeric(d["H2O_a"], errors="coerce")
            E = (F * w) * 1e-6 / max(S,1e-9)
        wgrad = pd.to_numeric(d["H2O_s"], errors="coerce") - pd.to_numeric(d["H2O_a"], errors="coerce")
        gtw = pd.Series(0.0, index=d.index)
        mask = wgrad.abs()>1e-3
        gtw.loc[mask] = (E.loc[mask]) / (wgrad.loc[mask])
        d["gtw"] = gtw
    if "gsw" not in d.columns or _is_all_zero_or_nan(d["gsw"]):
        if "gtw" in d.columns:
            gtw = pd.to_numeric(d["gtw"], errors="coerce")
            gbw_col = pd.to_numeric(d["gbw"], errors="coerce") if "gbw" in d.columns else None
            if gbw_col is not None and not _is_all_zero_or_nan(gbw_col):
                gbw = gbw_col.replace(0, np.nan).fillna(method="ffill").fillna(method="bfill")
                inv = 1.0/gtw.replace(0,np.nan) - 1.0/gbw.replace(0,np.nan)
                gsw = 1.0/inv
                d["gsw"] = gsw
            else:
                gbw_est = _estimate_gbw(F)
                inv = 1.0/gtw.replace(0,np.nan) - 1.0/gbw_est
                d["gsw"] = 1.0/inv
    return d

