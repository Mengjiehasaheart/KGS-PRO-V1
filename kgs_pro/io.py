import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from typing import Optional, Tuple, Dict

def _try_read_csv(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        try:
            return pd.read_csv(StringIO(text), sep="\t")
        except Exception:
            return pd.read_csv(StringIO(text), sep=";")

def read_pasted_table(text: str) -> pd.DataFrame:
    return _try_read_csv(text)

def _read_csv_bytes(data: BytesIO) -> pd.DataFrame:
    b = data.getvalue()
    s = b.decode("utf-8", errors="ignore")
    return _try_read_csv(s)

def _read_excel_bytes(data: BytesIO) -> pd.DataFrame:
    try:
        return pd.read_excel(data, sheet_name=0, engine="openpyxl")
    except Exception:
        return pd.read_excel(data, sheet_name=0)

def _read_licor_excel_smart_path(path: str) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path, sheet_name="Measurements", header=None, engine="openpyxl")
    except Exception:
        try:
            raw = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
        except Exception:
            raw = pd.read_excel(path, sheet_name=0, header=None)
    hdr = None
    for i in range(min(len(raw), 100)):
        vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        if any(v in ("obs","hhmmss","qin","a","gsw") for v in vals):
            hdr = i
            break
    if hdr is None:
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")
    cols = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    start = hdr + 2 if hdr + 2 < len(raw) else hdr + 1
    data = raw.iloc[start:].copy()
    data.columns = cols
    data = data.dropna(how="all").reset_index(drop=True)
    return data

def _read_licor_excel_smart_bytes(data: BytesIO) -> pd.DataFrame:
    try:
        raw = pd.read_excel(data, sheet_name=0, header=None, engine="openpyxl")
    except Exception:
        raw = pd.read_excel(data, sheet_name=0, header=None)
    hdr = None
    for i in range(min(len(raw), 100)):
        vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        if any(v in ("obs","hhmmss","qin","a","gsw") for v in vals):
            hdr = i
            break
    if hdr is None:
        data.seek(0)
        return _read_excel_bytes(data)
    cols = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    start = hdr + 2 if hdr + 2 < len(raw) else hdr + 1
    data_df = raw.iloc[start:].copy()
    data_df.columns = cols
    data_df = data_df.dropna(how="all").reset_index(drop=True)
    return data_df

def read_any(path_or_buffer, filename: Optional[str]=None) -> pd.DataFrame:
    if isinstance(path_or_buffer, (str,)):
        p = str(path_or_buffer).lower()
        if p.endswith((".xlsx",".xls")):
            return _read_licor_excel_smart_path(path_or_buffer)
        else:
            try:
                return pd.read_csv(path_or_buffer)
            except Exception:
                try:
                    return pd.read_csv(path_or_buffer, sep="\t")
                except Exception:
                    return pd.read_csv(path_or_buffer, sep=";")
    else:
        if filename:
            f = filename.lower()
            if f.endswith((".xlsx",".xls")):
                return _read_licor_excel_smart_bytes(path_or_buffer)
        try:
            return _read_csv_bytes(path_or_buffer)
        except Exception:
            return _read_licor_excel_smart_bytes(path_or_buffer)

def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,str]]:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    cols = {c.lower():c for c in d.columns}
    mapping = {}
    def has(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    A = has("a","anet","photo","assimilation")
    gsw = has("gsw","gs","condw","cndtotal","cond")
    Ci = has("ci","intercellularco2","ci_mmol")
    Qin = has("qin","q","pari","ppfd","parin","light","lamp_q")
    obs = has("obs","observation","index")
    hhmmss = has("hhmmss","time","timestamp","datetime","clock")
    fred = has("f_red","fred","red_frac","red")
    fblue = has("f_blue","fblue","blue_frac","blue")
    ffar = has("f_farred","ffar","farred_frac","farred")
    if A and A!="A":
        d.rename(columns={A:"A"}, inplace=True)
        mapping[A] = "A"
    if gsw and gsw!="gsw":
        d.rename(columns={gsw:"gsw"}, inplace=True)
        mapping[gsw] = "gsw"
    if Ci and Ci!="Ci":
        d.rename(columns={Ci:"Ci"}, inplace=True)
        mapping[Ci] = "Ci"
    if Qin and Qin!="Qin":
        d.rename(columns={Qin:"Qin"}, inplace=True)
        mapping[Qin] = "Qin"
    if obs and obs!="obs":
        d.rename(columns={obs:"obs"}, inplace=True)
        mapping[obs] = "obs"
    if hhmmss and hhmmss!="hhmmss":
        d.rename(columns={hhmmss:"hhmmss"}, inplace=True)
        mapping[hhmmss] = "hhmmss"
    if fred and fred!="f_red":
        d.rename(columns={fred:"f_red"}, inplace=True)
        mapping[fred] = "f_red"
    if fblue and fblue!="f_blue":
        d.rename(columns={fblue:"f_blue"}, inplace=True)
        mapping[fblue] = "f_blue"
    if ffar and ffar!="f_farred":
        d.rename(columns={ffar:"f_farred"}, inplace=True)
        mapping[ffar] = "f_farred"
    return d, mapping

def choose_light_column(df: pd.DataFrame) -> Optional[str]:
    cands = ["Qin","Q","PARi","PPFD","par","Light","Rabs"]
    best = None
    best_rng = -1.0
    for c in cands:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().sum()>3:
                rng = float(v.max() - v.min())
                if rng > best_rng:
                    best_rng = rng
                    best = c
    return best
