import os
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from typing import Optional, Tuple, Dict
from tempfile import NamedTemporaryFile
from openpyxl.utils import get_column_letter, column_index_from_string

def _coerce_number(v):
    if v is None:
        return None
    try:
        f = float(v)
        if pd.isna(f):
            return None
        return f
    except Exception:
        pass
    if isinstance(v, str):
        cleaned = "".join([ch for ch in v if (ch.isdigit() or ch in ".-")])
        if cleaned:
            try:
                f = float(cleaned)
                if pd.isna(f):
                    return None
                return f
            except Exception:
                return None
    return None

def _extract_leaf_area_cm2(raw: pd.DataFrame) -> Optional[float]:
    limit = min(len(raw), 50)
    for i in range(limit - 1):
        row_lower = [str(x).strip().lower() if isinstance(x, str) else None for x in raw.iloc[i]]
        if any(x == "const" for x in row_lower):
            next_row = raw.iloc[i + 1]
            for idx, lab in enumerate(row_lower):
                if lab in ("s", "leaf area", "leaf_area", "leafarea", "area"):
                    val = _coerce_number(next_row.iloc[idx])
                    if val is not None and val > 0:
                        return float(val)
        for idx, lab in enumerate(row_lower):
            if lab in ("leaf area", "leaf_area", "leafarea"):
                target = raw.iloc[i].iloc[idx + 1] if idx + 1 < len(raw.columns) else None
                val = _coerce_number(target)
                if val is not None and val > 0:
                    return float(val)
    return None

def _extract_metadata(raw: pd.DataFrame) -> Dict[str, float]:
    def cell_value(coord: str):
        letters = "".join([ch for ch in coord if ch.isalpha()])
        digits = "".join([ch for ch in coord if ch.isdigit()])
        if not letters or not digits:
            return None
        try:
            r = int(digits) - 1
            c = column_index_from_string(letters) - 1
        except Exception:
            return None
        if r < 0 or c < 0 or r >= len(raw) or c >= raw.shape[1]:
            return None
        return raw.iat[r, c]
    coord_map = {
        "leaf_area_cm2": "B7",
        "k_ratio": "C7",
        "d7_label": "D7",
        "e7_value": "E7",
        "d5": "D5",
        "e5": "E5",
        "f5": "F5",
        "g5": "G5",
        "h5": "H5",
        "i5": "I5",
        "j5": "J5",
        "k5": "K5",
        "b13": "B13",
        "c13": "C13",
        "f13": "F13",
        "b15": "B15",
        "c15": "C15",
        "d15": "D15",
        "e15": "E15",
        "h15": "H15",
    }
    meta: Dict[str, float] = {}
    for key, coord in coord_map.items():
        val = cell_value(coord)
        if key == "d7_label":
            if isinstance(val, str) and val:
                meta[key] = val
            continue
        num = _coerce_number(val)
        if num is not None:
            meta[key] = num
    if "leaf_area_cm2" not in meta:
        area = _extract_leaf_area_cm2(raw)
        if area is not None:
            meta["leaf_area_cm2"] = area
    return meta

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
    meta = _extract_metadata(raw)
    area_cm2 = meta.get("leaf_area_cm2")
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
    if area_cm2 is not None:
        data.attrs["leaf_area_cm2"] = area_cm2
    if meta:
        data.attrs["licor_meta"] = meta
    return data

def _read_licor_excel_smart_bytes(data: BytesIO) -> pd.DataFrame:
    try:
        raw = pd.read_excel(data, sheet_name=0, header=None, engine="openpyxl")
    except Exception:
        raw = pd.read_excel(data, sheet_name=0, header=None)
    hdr = None
    meta = _extract_metadata(raw)
    area_cm2 = meta.get("leaf_area_cm2")
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
    if area_cm2 is not None:
        data_df.attrs["leaf_area_cm2"] = area_cm2
    if meta:
        data_df.attrs["licor_meta"] = meta
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
    meta = _extract_metadata(raw)
    area_cm2 = meta.get("leaf_area_cm2")
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
    if area_cm2 is not None:
        data_df.attrs["leaf_area_cm2"] = area_cm2
    if meta:
        data_df.attrs["licor_meta"] = meta
    target_eval = set(["A","E","gsw","gtw","Ci","Ca","gbw","gbc","gsc","gtc","VPDleaf","RHcham","Pca","Pci"])
    eval_cols = set()
    for c in target_eval:
        if c not in data_df.columns:
            continue
        col_vals = pd.to_numeric(data_df[c], errors="coerce")
        max_abs = col_vals.abs().max()
        all_zero_or_nan = col_vals.isna().all() or (pd.notna(max_abs) and float(max_abs)==0.0)
        if not all_zero_or_nan:
            continue
        eval_cols.add(c)
    for r_idx in range(len(data_df)):
        excel_row = start + r_idx + 1
        for c_idx, c in enumerate(cols):
            if c not in eval_cols:
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
