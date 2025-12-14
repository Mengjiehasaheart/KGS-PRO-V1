import pandas as pd
import numpy as np
from typing import Optional

def _leaf_area_m2(df: pd.DataFrame, default_cm2: float=6.0) -> float:
    attr_cm2 = df.attrs.get("leaf_area_cm2")
    if attr_cm2 is not None:
        try:
            val = float(attr_cm2)
            if val > 0:
                return val/10000.0
        except Exception:
            pass
    attr_m2 = df.attrs.get("leaf_area_m2")
    if attr_m2 is not None:
        try:
            val = float(attr_m2)
            if val > 0:
                return val
        except Exception:
            pass
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
    meta = d.attrs.get("licor_meta", {}) or {}
    def mval(key, default=None):
        v = meta.get(key, default)
        try:
            f = float(v)
            if np.isnan(f):
                return default
            return f
        except Exception:
            return v if v is not None else default
    area_m2 = _leaf_area_m2(d)
    area_cm2_eff = mval("leaf_area_cm2", area_m2*10000.0 if area_m2 is not None else 6.0)
    k_ratio = mval("k_ratio", 0.5)
    d7_label = meta.get("d7_label", "")
    e7_value = mval("e7_value", None)
    d5 = mval("d5", 0.0)
    e5 = mval("e5", 0.0)
    f5 = mval("f5", 0.0)
    g5 = mval("g5", 0.0)
    h5 = mval("h5", 0.0)
    i5 = mval("i5", 0.0)
    j5 = mval("j5", 0.0)
    k5 = mval("k5", 1.0)
    b13 = mval("b13", 0.0)
    c13 = mval("c13", 0.0)
    f13 = mval("f13", 1.0)
    b15 = mval("b15", 0.0)
    c15 = mval("c15", 0.0)
    d15 = mval("d15", 0.0)
    e15 = mval("e15", 0.0)
    h15 = mval("h15", 0.0)
    corr_factor = None
    ak = None
    leak = pd.to_numeric(d["Leak"], errors="coerce") if "Leak" in d.columns else None
    fan = pd.to_numeric(d["Fan_speed"], errors="coerce") if "Fan_speed" in d.columns else None
    pa_col = pd.to_numeric(d["Pa"], errors="coerce") if "Pa" in d.columns else None
    tair_col = pd.to_numeric(d["Tair"], errors="coerce") if "Tair" in d.columns else (pd.to_numeric(d["Tleaf"], errors="coerce") if "Tleaf" in d.columns else None)
    if leak is not None and fan is not None and pa_col is not None and tair_col is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ak = ((b15 + c15*fan)/(1.0 + d15*fan)) * pa_col/(tair_col + 273.0) * e15
        ak = pd.to_numeric(pd.Series(ak, index=d.index), errors="coerce")
        ak = ak.clip(lower=0)
        corr_raw = np.where(ak.notna(), ak, np.nan)
        if h15 is not None:
            denom_corr = np.maximum(ak - leak*h15, 1e-9)
            corr_raw = np.where(leak*h15 >= ak, 1.0, ak/denom_corr)
        corr_factor = pd.to_numeric(pd.Series(corr_raw, index=d.index), errors="coerce")
        d["CorrFact"] = corr_factor
    flow_series = pd.to_numeric(d["Flow"], errors="coerce") if "Flow" in d.columns else pd.Series(_flow_umol_s(d), index=d.index)
    hs = pd.to_numeric(d["H2O_s"], errors="coerce") if "H2O_s" in d.columns else None
    hr = pd.to_numeric(d["H2O_r"], errors="coerce") if "H2O_r" in d.columns else (pd.to_numeric(d["H2O_a"], errors="coerce") if "H2O_a" in d.columns else None)
    cf = corr_factor if corr_factor is not None else 1.0
    if hs is not None and hr is not None and ("Emm" not in d.columns or _is_all_zero_or_nan(d["Emm"])):
        denom_em = (1000.0 - cf*hs).replace(0, np.nan)
        Emmol = 1000.0*flow_series*cf*(hs - hr) / (100.0 * max(area_cm2_eff, 1e-9) * denom_em)
        d["Emm"] = Emmol
    if "Emm" in d.columns and ("E" not in d.columns or _is_all_zero_or_nan(d["E"])):
        d["E"] = pd.to_numeric(d["Emm"], errors="coerce")/1000.0
    if ("CO2_r" in d.columns and "CO2_s" in d.columns and hs is not None and hr is not None) and ("A" not in d.columns or _is_all_zero_or_nan(d["A"])):
        adj = pd.to_numeric(d["CO2_r"], errors="coerce") - pd.to_numeric(d["CO2_s"], errors="coerce") * (1000.0 - cf*hr) / (1000.0 - cf*hs)
        A = flow_series * cf * adj / (100.0 * max(area_cm2_eff, 1e-9))
        d["A"] = A
    if "Qin" not in d.columns or _is_all_zero_or_nan(d["Qin"]):
        parts = []
        qamb_in = pd.to_numeric(d["Qamb_in"], errors="coerce") if "Qamb_in" in d.columns else None
        qamb_out = pd.to_numeric(d["Qamb_out"], errors="coerce") if "Qamb_out" in d.columns else None
        qcol = pd.to_numeric(d["Q"], errors="coerce") if "Q" in d.columns else None
        far = pd.to_numeric(d["f_farred"], errors="coerce") if "f_farred" in d.columns else None
        if qamb_in is not None:
            parts.append(b13 * qamb_in)
        if qamb_out is not None:
            parts.append(c13 * qamb_out)
        if qcol is not None:
            if far is None:
                far = 0.0
            parts.append(f13 * qcol * (1.0 - far))
        if parts:
            qin = parts[0]
            for p in parts[1:]:
                qin = qin.add(p, fill_value=0)
            d["Qin"] = qin
    if ("gbw" not in d.columns or _is_all_zero_or_nan(d["gbw"])) and fan is not None and pa_col is not None:
        base = fan * pa_col / (max(k5, 1e-9) * 1000.0)
        label = str(d7_label).strip().lower()
        if label and not label.startswith("0"):
            if label.startswith("1"):
                gbw_series = pd.Series(3.0, index=d.index)
            else:
                gbw_series = pd.Series(e7_value if e7_value is not None else np.nan, index=d.index)
        else:
            span = np.maximum(np.minimum(area_cm2_eff, j5 if j5 is not None else 0.0), i5 if i5 is not None else 0.0)
            gbw_series = (d5 + e5*base) + (f5*base*(span**2)) + (g5*span*base) + (h5*(base**2))
        d["gbw"] = gbw_series
    if ("gtw" not in d.columns or _is_all_zero_or_nan(d["gtw"])) and "E" in d.columns and hs is not None and pa_col is not None:
        Ecol = pd.to_numeric(d["E"], errors="coerce")
        tleaf_cnd = pd.to_numeric(d["TleafCnd"], errors="coerce") if "TleafCnd" in d.columns else None
        if tleaf_cnd is not None and _is_all_zero_or_nan(tleaf_cnd):
            tleaf_cnd = None
        if tleaf_cnd is None and "Tleaf" in d.columns:
            tleaf_cnd = pd.to_numeric(d["Tleaf"], errors="coerce")
        dp = pd.to_numeric(d["ΔPcham"], errors="coerce") if "ΔPcham" in d.columns else 0.0
        if tleaf_cnd is not None:
            sat = 0.61365 * np.exp(17.502 * tleaf_cnd / (240.97 + tleaf_cnd))
            denom_gtw = 1000.0 * sat / (pa_col + dp) - hs
            numer_gtw = 1000.0 - (1000.0 * sat / (pa_col + dp) + hs) / 2.0
            with np.errstate(divide="ignore", invalid="ignore"):
                gtw = Ecol * numer_gtw / denom_gtw
            d["gtw"] = gtw
    if "gsw" not in d.columns or _is_all_zero_or_nan(d["gsw"]):
        if "gtw" in d.columns and "gbw" in d.columns:
            gtw = pd.to_numeric(d["gtw"], errors="coerce")
            gbw_series = pd.to_numeric(d["gbw"], errors="coerce")
            inv_s = 1.0/gtw.replace(0, np.nan)
            inv_b = 1.0/gbw_series.replace(0, np.nan)
            term = inv_s - inv_b
            inner = term*term + 4.0*k_ratio/((k_ratio+1.0)*(k_ratio+1.0))*(2.0*inv_s*inv_b - inv_b*inv_b)
            with np.errstate(divide="ignore", invalid="ignore"):
                root = np.sqrt(inner)
                gsw = 2.0/(term + np.sign(gtw)*root)
            d["gsw"] = gsw
        elif "gtw" in d.columns:
            gtw = pd.to_numeric(d["gtw"], errors="coerce")
            gbw_est = _estimate_gbw(_flow_umol_s(d))
            inv = 1.0/gtw.replace(0,np.nan) - 1.0/gbw_est
            d["gsw"] = 1.0/inv
    if "gsc" not in d.columns and "gsw" in d.columns:
        d["gsc"] = pd.to_numeric(d["gsw"], errors="coerce")/1.6
    if "gbc" not in d.columns and "gbw" in d.columns:
        d["gbc"] = pd.to_numeric(d["gbw"], errors="coerce")/1.37
    if ("gtc" not in d.columns or _is_all_zero_or_nan(d["gtc"])) and "gsc" in d.columns and "gbc" in d.columns:
        gsc = pd.to_numeric(d["gsc"], errors="coerce")
        gbc = pd.to_numeric(d["gbc"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            base_gtc = 1.0/((k_ratio+1.0)/(gsc) + 1.0/(gbc))
            add_gtc = k_ratio/(((k_ratio+1.0)/(gsc)) + k_ratio/(gbc))
        gtc = base_gtc + add_gtc
        gtc = gtc.where(gtc != 0)
        d["gtc"] = gtc
    if "Ca" not in d.columns or _is_all_zero_or_nan(d["Ca"]):
        if "CO2_s" in d.columns:
            ca = pd.to_numeric(d["CO2_s"], errors="coerce")
            if corr_factor is not None and ak is not None and "A" in d.columns:
                adj = pd.to_numeric(d["A"], errors="coerce") * area_cm2_eff * 100.0 / ak.replace(0,np.nan)
                ca = ca - adj.where(corr_factor > 1.0, 0)
            d["Ca"] = ca
        elif "CO2_r" in d.columns:
            d["Ca"] = pd.to_numeric(d["CO2_r"], errors="coerce")
    if "Ci" not in d.columns or _is_all_zero_or_nan(d["Ci"]):
        if all(c in d.columns for c in ["Ca","A","gtc"]) and "E" in d.columns:
            Ca = pd.to_numeric(d["Ca"], errors="coerce")
            A = pd.to_numeric(d["A"], errors="coerce")
            gtc = pd.to_numeric(d["gtc"], errors="coerce")
            Ecol = pd.to_numeric(d["E"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                Ci = ((gtc - Ecol/2.0) * Ca - A) / (gtc + Ecol/2.0)
            d["Ci"] = Ci
    if "Pca" not in d.columns and "Ca" in d.columns and pa_col is not None:
        press = pa_col
        if "ΔPcham" in d.columns:
            press = press + pd.to_numeric(d["ΔPcham"], errors="coerce")
        d["Pca"] = pd.to_numeric(d["Ca"], errors="coerce") * press/1000.0
    if "Pci" not in d.columns and "Ci" in d.columns and pa_col is not None:
        press = pa_col
        if "ΔPcham" in d.columns:
            press = press + pd.to_numeric(d["ΔPcham"], errors="coerce")
        d["Pci"] = pd.to_numeric(d["Ci"], errors="coerce") * press/1000.0
    if "Rabs" not in d.columns and "Qin" in d.columns:
        d["Rabs"] = pd.to_numeric(d["Qin"], errors="coerce")
    return d
