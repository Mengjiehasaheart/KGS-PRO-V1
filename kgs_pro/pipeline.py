from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Dict
import pandas as pd
from .io import read_any, standardize_columns, choose_light_column
from .calc import ensure_derived
from .preprocess import infer_time, rolling_smooth
from .detect import detect_steps, build_phase_mask


@dataclass
class ProcessResult:
    filename: str
    df: Optional[pd.DataFrame]
    time_col: Optional[str]
    light_col: Optional[str]
    steps: Dict[str, Optional[int]]
    error: Optional[str]
    colmap: Dict[str, str]


def prepare_dataframe(df: pd.DataFrame, dt_val: Optional[float], smooth: bool, win: int, light_choice: Optional[str], light_auto: bool, detect: bool) -> ProcessResult:
    d, colmap = standardize_columns(df)
    d, time_col = infer_time(d, dt_val)
    if smooth:
        for c in ["A", "gsw", "Ci", "Qin"]:
            if c in d.columns:
                d[c] = rolling_smooth(pd.to_numeric(d[c], errors="coerce"), win)
    light_col = choose_light_column(d) if light_auto else light_choice
    steps = {"up": None, "down": None}
    if detect and light_col:
        steps = detect_steps(d, light_col, time_col)
        d["Phase_auto"] = build_phase_mask(len(d), steps.get("up"), steps.get("down"))
    return ProcessResult(filename="", df=d, time_col=time_col, light_col=light_col, steps=steps, error=None, colmap=colmap)


def process_source(source, filename: Optional[str], calc_engine: str, dt_val: Optional[float], smooth: bool, win: int, light_choice: Optional[str], light_auto: bool, detect: bool, apply_calc: bool=True) -> ProcessResult:
    try:
        buf = source
        if isinstance(source, (bytes, bytearray)):
            buf = BytesIO(source)
        df_raw = read_any(buf, filename, calc_engine=calc_engine)
        if apply_calc:
            df_raw = ensure_derived(df_raw)
        prepared = prepare_dataframe(df_raw, dt_val, smooth, win, light_choice, light_auto, detect)
        prepared.filename = filename if filename else ""
        return prepared
    except Exception as e:
        return ProcessResult(filename=filename if filename else "", df=None, time_col=None, light_col=None, steps={"up": None, "down": None}, error=str(e), colmap={})


def process_bytes_job(payload):
    data_bytes, filename, calc_engine, dt_val, smooth, win, apply_calc = payload
    return process_source(data_bytes, filename, calc_engine, dt_val, smooth, win, None, True, True, apply_calc=apply_calc)
