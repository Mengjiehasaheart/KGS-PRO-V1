import os
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from typing import Optional, Tuple, Dict
from tempfile import NamedTemporaryFile
from openpyxl.utils import get_column_letter

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

def _read_excel_with_xlcalculator_path(path: str) -> pd.DataFrame:
    fast = _read_licor_excel_smart_path(path)
    cols_fast = [str(c).strip() for c in fast.columns]
    base_ok = "CO2_r" in cols_fast and "CO2_s" in cols_fast
    h2o_ok = "H2O_s" in cols_fast and ("H2O_r" in cols_fast or "H2O_a" in cols_fast)
    if base_ok and h2o_ok:
        return fast
    from kgs_pro.xlcalculator.xlcalculator import ModelCompiler, Evaluator
    compiler = ModelCompiler()
    model = compiler.read_and_parse_archive(path, build_code=True)
    evaluator = Evaluator(model)
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet_name = "Measurements" if "Measurements" in xls.sheet_names else xls.sheet_names[0]
        raw = xls.parse(sheet_name=sheet_name, header=None)
    except Exception:
        raw = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
        sheet_name = None
    if sheet_name is None:
        try:
            sheet_name = pd.ExcelFile(path, engine="openpyxl").sheet_names[0]
        except Exception:
            sheet_name = "Sheet1"
    hdr = None
    for i in range(min(len(raw), 120)):
        vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        if any(v in ("obs","hhmmss","qin","a","gsw") for v in vals):
            hdr = i
            break
    if hdr is None:
        return _read_licor_excel_smart_path(path)
    cols = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    start = hdr + 2 if hdr + 2 < len(raw) else hdr + 1
    data_df = raw.iloc[start:].copy()
    data_df.columns = cols
    data_df = data_df.reset_index(drop=True)
    sheet = sheet_name
    target_eval = set(["A","E","gsw","gtw","Ci","Ca","gbw","gbc","gsc","gtc","VPDleaf","RHcham","Pca","Pci"])
    def has_all(names):
        return all(n in data_df.columns for n in names)
    skip_if_base = {
        "A": ["CO2_r","CO2_s"],
        "E": ["H2O_s","H2O_r"],
        "gsw": ["gtw"],
        "gtw": ["H2O_s","H2O_r","H2O_a","Tleaf","Pa"],
        "Ca": ["CO2_r"],
        "Ci": ["Ca","A"],
        "gbw": ["Flow"],
        "gsc": ["gsw"],
        "gtc": ["gsc","gbc"],
        "gbc": ["gbw"],
    }
    eval_cols = set()
    for c in target_eval:
        if c not in data_df.columns:
            continue
        col_vals = pd.to_numeric(data_df[c], errors="coerce")
        if col_vals.notna().any():
            continue
        base_need = skip_if_base.get(c, [])
        if base_need and has_all(base_need):
            continue
        eval_cols.add(c)
    for r_idx in range(len(data_df)):
        excel_row = start + r_idx + 1
        for c_idx, c in enumerate(cols):
            if c not in eval_cols:
                continue
            cur = data_df.iloc[r_idx, c_idx]
            needs_eval = isinstance(cur, str) and cur.strip().startswith("=")
            if not needs_eval:
                continue
            addr = f"{sheet}!{get_column_letter(c_idx+1)}{excel_row}"
            try:
                val = evaluator.evaluate(addr)
                data_df.iat[r_idx, c_idx] = val
            except Exception:
                pass
    data_df = data_df.dropna(how="all").reset_index(drop=True)
    data_df.columns = [str(c).strip() for c in data_df.columns]
    return data_df

def _read_excel_with_xlcalculator_bytes(data: BytesIO) -> pd.DataFrame:
    tmp = NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.write(data.getvalue())
    tmp.flush()
    tmp.close()
    try:
        return _read_excel_with_xlcalculator_path(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

def read_any(path_or_buffer, filename: Optional[str]=None, calc_engine: str="mf") -> pd.DataFrame:
    if isinstance(path_or_buffer, (str,)):
        p = str(path_or_buffer).lower()
        if p.endswith((".xlsx",".xls")):
            if calc_engine == "xlcalculator":
                try:
                    return _read_excel_with_xlcalculator_path(path_or_buffer)
                except Exception:
                    pass
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
                if calc_engine == "xlcalculator":
                    try:
                        if isinstance(path_or_buffer, BytesIO):
                            return _read_excel_with_xlcalculator_bytes(path_or_buffer)
                        else:
                            return _read_excel_with_xlcalculator_bytes(BytesIO(path_or_buffer.getvalue()))
                    except Exception:
                        path_or_buffer.seek(0)
                        return _read_licor_excel_smart_bytes(path_or_buffer)
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
