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
    if "Ca" not in d.columns or _is_all_zero_or_nan(d["Ca"]):
        if "CO2_r" in d.columns:
            d["Ca"] = pd.to_numeric(d["CO2_r"], errors="coerce")
        elif "CO2_a" in d.columns:
            d["Ca"] = pd.to_numeric(d["CO2_a"], errors="coerce")
    if ("H2O_s" in d.columns and ("H2O_r" in d.columns or "H2O_a" in d.columns)) and ("E" not in d.columns or _is_all_zero_or_nan(d["E"])):
        if "H2O_r" in d.columns:
            w = pd.to_numeric(d["H2O_s"], errors="coerce") - pd.to_numeric(d["H2O_r"], errors="coerce")
        else:
            w = pd.to_numeric(d["H2O_s"], errors="coerce") - pd.to_numeric(d["H2O_a"], errors="coerce")
        Emmol = (F * w) * 1e-6 / max(S,1e-9)
        d["E"] = Emmol
    if ("CO2_r" in d.columns and "CO2_s" in d.columns) and ("A" not in d.columns or _is_all_zero_or_nan(d["A"])):
        cdiff = pd.to_numeric(d["CO2_r"], errors="coerce") - pd.to_numeric(d["CO2_s"], errors="coerce")
        A = (F * cdiff) / (1e6 * max(S,1e-9))
        d["A"] = A
    if ("H2O_s" in d.columns and ("H2O_r" in d.columns or "H2O_a" in d.columns) and "H2O_a" in d.columns and "Tleaf" in d.columns and "Pa" in d.columns) and ("gtw" not in d.columns or _is_all_zero_or_nan(d["gtw"])):
        if "E" in d.columns and not _is_all_zero_or_nan(d["E"]):
            E = pd.to_numeric(d["E"], errors="coerce")
        else:
            if "H2O_r" in d.columns:
                w = pd.to_numeric(d["H2O_s"], errors="coerce") - pd.to_numeric(d["H2O_r"], errors="coerce")
            else:
                w = pd.to_numeric(d["H2O_s"], errors="coerce") - pd.to_numeric(d["H2O_a"], errors="coerce")
            E = (F * w) * 1e-6 / max(S,1e-9)
        T = pd.to_numeric(d["Tleaf"], errors="coerce")
        P = pd.to_numeric(d["Pa"], errors="coerce")
        es = 0.61121 * np.exp((18.678 - T/234.5) * (T/(257.14+T)))
        wi = 1000.0 * es / P.replace(0,np.nan)
        wa = pd.to_numeric(d["H2O_a"], errors="coerce")
        wgrad = wi - wa
        gtw = pd.Series(np.nan, index=d.index)
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
    if "gsc" not in d.columns and "gsw" in d.columns:
        d["gsc"] = pd.to_numeric(d["gsw"], errors="coerce")/1.6
    if "gbw" not in d.columns:
        d["gbw"] = _estimate_gbw(F)
    if "gbc" not in d.columns and "gbw" in d.columns:
        d["gbc"] = pd.to_numeric(d["gbw"], errors="coerce")/1.37
    if "gtc" not in d.columns and "gsc" in d.columns and "gbc" in d.columns:
        gsc = pd.to_numeric(d["gsc"], errors="coerce")
        gbc = pd.to_numeric(d["gbc"], errors="coerce")
        inv = 1.0/gsc.replace(0,np.nan) + 1.0/gbc.replace(0,np.nan)
        gtc = 1.0/inv
        gtc = gtc.where(gtc != 0)
        d["gtc"] = gtc
    if "Ci" not in d.columns or _is_all_zero_or_nan(d["Ci"]):
        if "Ca" in d.columns and "A" in d.columns:
            Ca = pd.to_numeric(d["Ca"], errors="coerce")
            A = pd.to_numeric(d["A"], errors="coerce")
            if "gtc" in d.columns and not _is_all_zero_or_nan(d["gtc"]):
                gtc = pd.to_numeric(d["gtc"], errors="coerce")
                Ci = pd.Series(np.nan, index=d.index)
                mask = gtc.replace(0,np.nan).notna()
                Ci.loc[mask] = Ca.loc[mask] - A.loc[mask]/gtc.loc[mask]
                d["Ci"] = Ci
            else:
                Ci = Ca * 0.7
                d["Ci"] = Ci
    if "Pca" not in d.columns and "Ca" in d.columns:
        if "Pa" in d.columns:
            d["Pca"] = pd.to_numeric(d["Ca"], errors="coerce") * pd.to_numeric(d["Pa"], errors="coerce")/1000.0
    if "Pci" not in d.columns and "Ci" in d.columns:
        if "Pa" in d.columns:
            d["Pci"] = pd.to_numeric(d["Ci"], errors="coerce") * pd.to_numeric(d["Pa"], errors="coerce")/1000.0
    if "Rabs" not in d.columns and "Qin" in d.columns:
        d["Rabs"] = pd.to_numeric(d["Qin"], errors="coerce")
    return d
