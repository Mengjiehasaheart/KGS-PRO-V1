import streamlit as st
import pandas as pd
import numpy as np
from kgs_pro.io import read_any, read_pasted_table, standardize_columns, choose_light_column
from kgs_pro.preprocess import infer_time, rolling_smooth
from kgs_pro.calc import ensure_derived
from kgs_pro.detect import detect_steps, classify_levels, build_phase_mask
from kgs_pro.models import fit_step_increase, fit_step_decrease, _increase_fun, _decrease_fun
from kgs_pro.plotting import make_timeseries, overlay_fit
from pathlib import Path

st.set_page_config(page_title="KGS-PRO V1", layout="wide")

st.title("KGS-PRO V1: Dynamic Photosynthesis Analysis")

with st.sidebar:
    st.header("Data")
    src = st.radio("Input", ["Upload","Paste","Sample"], index=0)
    uploaded = None
    pasted = None
    sample_path = Path("Sample_Dataset/Demo_dataset_1.xlsx")
    if src=="Upload":
        uploaded = st.file_uploader("File", type=["csv","txt","xlsx","xls"]) 
    elif src=="Paste":
        pasted = st.text_area("Paste table (CSV/TSV)")
    else:
        st.write(str(sample_path))
    st.header("Plot")
    smooth = st.checkbox("Smooth", value=True)
    win = st.slider("Window", 1, 21, 5, 2)
    st.header("Time")
    manual_dt = st.checkbox("Set sampling interval (s)")
    dt_val = st.number_input("Interval (s)", min_value=0.1, value=1.0, step=0.1) if manual_dt else None
    st.header("Light")
    light_auto = st.checkbox("Auto-select light column", value=True)
    light_choice = None
    if not light_auto:
        light_opts = [c for c in ["Qin","Q","PARi","PPFD","par","Light","Rabs"] if c in (df.columns if 'df' in globals() and df is not None else [])]
        light_choice = st.selectbox("Light column", options=light_opts) if light_opts else None
    st.header("Fit")
    fit_var = st.selectbox("Variable", options=["gsw","A","Ci"])
    pts_avg = st.number_input("Points to average", min_value=3, value=5, step=1)
    do_fit_inc = st.checkbox("Fit increase", value=True)
    do_fit_dec = st.checkbox("Fit decrease", value=True)
    auto_steps = st.checkbox("Auto-detect steps", value=True)

def load_df():
    if src=="Upload" and uploaded is not None:
        return read_any(uploaded, uploaded.name)
    if src=="Paste" and pasted:
        return read_pasted_table(pasted)
    if src=="Sample" and sample_path.exists():
        return read_any(str(sample_path))
    return None

df = load_df()

if df is None:
    st.info("Load data to begin")
    st.stop()

df, colmap = standardize_columns(df)
df = ensure_derived(df)
df, time_col = infer_time(df, dt_val)

if smooth:
    for c in ["A","gsw","Ci","Qin"]:
        if c in df.columns:
            df[c] = rolling_smooth(pd.to_numeric(df[c], errors="coerce"), win)

q_col = choose_light_column(df) if light_auto else light_choice

steps = {"up":None, "down":None}
low_level = None
high_level = None
if q_col:
    steps = detect_steps(df, q_col, time_col) if auto_steps else steps
    low_level, high_level = classify_levels(df, q_col)
    df["Phase_auto"] = build_phase_mask(len(df), steps.get("up"), steps.get("down"))

left = st.selectbox("Left axis", options=[c for c in ["A","gsw","Ci", q_col] if c in df.columns], index=0)
right = st.selectbox("Right axis", options=["None"] + [c for c in ["A","gsw","Ci", q_col] if c in df.columns and c!=left])
hover_cols = [c for c in ["f_red","f_blue","f_farred","Phase","Phase_auto", q_col] if c in df.columns]

fig = make_timeseries(df, time_col, left, None if right=="None" else right, steps=steps, hover_extra=hover_cols)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Experiment")
cols = st.columns(3)
cols[0].metric("Low PPFD", f"{low_level:.1f}" if low_level is not None else "-")
cols[1].metric("High PPFD", f"{high_level:.1f}" if high_level is not None else "-")
exp_type = "Unknown"
if steps.get("up") is not None and steps.get("down") is None:
    exp_type = "Step Increase"
elif steps.get("up") is not None and steps.get("down") is not None:
    exp_type = "Increase + Decrease"
cols[2].metric("Type", exp_type)

st.subheader("Model Fitting")

if fit_var not in df.columns:
    st.warning("Variable not found")
    st.stop()

t = df[time_col].values.astype(float)
y = pd.to_numeric(df[fit_var], errors="coerce").astype(float).values

up_idx = steps.get("up")
down_idx = steps.get("down")

res_inc = None
res_dec = None
if do_fit_inc and up_idx is not None:
    li = max(0, up_idx - int(pts_avg))
    hi = down_idx if down_idx is not None else len(df)
    gsmin = float(pd.Series(y[max(0,up_idx-pts_avg):up_idx]).dropna().tail(int(pts_avg)).mean())
    gsmax = float(pd.Series(y[max(0,hi-pts_avg):hi]).dropna().tail(int(pts_avg)).mean())
    amp = gsmax - gsmin
    snr_ok = np.isfinite(gsmin) and np.isfinite(gsmax) and amp > max(1e-3, 2.0*np.nanstd(pd.Series(y[li:hi]).diff().dropna()))
    if snr_ok:
        tt = t[up_idx:hi]
        yy = y[up_idx:hi]
        res_inc = fit_step_increase(tt - tt[0], yy, gsmin, gsmax)
        if res_inc.get("ok",0)==1:
            yhat = gsmin + (gsmax-gsmin)*res_inc["scale"]*_increase_fun(tt-tt[0], res_inc["tlag"], res_inc["tau1"], res_inc["tau2"], res_inc["f"])
            overlay_fit(fig, tt, yhat, name="Fit Increase")
    else:
        res_inc = {"ok":0}

if do_fit_dec and down_idx is not None:
    lo = down_idx
    hi2 = len(df)
    gsmax2 = float(pd.Series(y[max(0,up_idx-pts_avg):up_idx]).dropna().tail(int(pts_avg)).mean()) if up_idx is not None else float(np.nan)
    gsmin2 = float(pd.Series(y[max(0,hi2-pts_avg):hi2]).dropna().tail(int(pts_avg)).mean())
    amp2 = gsmax2 - gsmin2 if np.isfinite(gsmax2) else 0.0
    snr_ok2 = np.isfinite(gsmin2) and np.isfinite(gsmax2) and amp2 > max(1e-3, 2.0*np.nanstd(pd.Series(y[lo:hi2]).diff().dropna()))
    if snr_ok2:
        tt2 = t[lo:hi2]
        yy2 = y[lo:hi2]
        res_dec = fit_step_decrease(tt2 - tt2[0], yy2, gsmin2, gsmax2)
        if res_dec.get("ok",0)==1:
            yhat2 = gsmax2 - (gsmax2-gsmin2)*_decrease_fun(tt2-tt2[0], res_dec["tlag"], res_dec["tau1"], res_dec["tau2"], res_dec["f"])
            overlay_fit(fig, tt2, yhat2, name="Fit Decrease")
    else:
        res_dec = {"ok":0}

st.plotly_chart(fig, use_container_width=True)

def fmt_res(r):
    if not r or r.get("ok",0)==0:
        return None
    keys = ["tlag","tau1","tau2","f","scale","rmse","r2"] if "scale" in r else ["tlag","tau1","tau2","f","rmse","r2"]
    return {k:r.get(k,np.nan) for k in keys}

tbl_inc = fmt_res(res_inc)
tbl_dec = fmt_res(res_dec)
cols2 = st.columns(2)
cols2[0].write(pd.DataFrame([tbl_inc]).T.rename(columns={0:"Increase"})) if tbl_inc else cols2[0].info("Increase: not fitted")
cols2[1].write(pd.DataFrame([tbl_dec]).T.rename(columns={0:"Decrease"})) if tbl_dec else cols2[1].info("Decrease: not fitted")
