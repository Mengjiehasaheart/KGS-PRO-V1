import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from kgs_pro.io import read_any, read_pasted_table, standardize_columns, choose_light_column
from kgs_pro.preprocess import infer_time, rolling_smooth
from kgs_pro.calc import ensure_derived
from kgs_pro.detect import detect_steps, classify_levels, build_phase_mask
from kgs_pro.models import fit_step_increase, fit_step_decrease
from kgs_pro.plotting import make_timeseries, overlay_fit
from kgs_pro.metrics import percent_times, deficit, sl_approx
from kgs_pro.partition import calibrate_params, partition_limits
from pathlib import Path
from io import StringIO

st.set_page_config(page_title="KGS-PRO V1", layout="wide")

st.title("KGS-PRO V1: Dynamic Photosynthesis Analysis")
st.markdown("<div style='text-align: right; font-size: 12px; font-weight: bold; color: #333;'>© Mengjie Fan</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data")
    src = st.radio("Input", ["Upload","Paste","Sample"], index=0)
    uploaded = None
    pasted = None
    app_sample = Path(__file__).parent / "sample_data" / "Demo_dataset_1.xlsx"
    root_sample = Path("Sample_Dataset/Demo_dataset_1.xlsx")
    sample_path = app_sample if app_sample.exists() else root_sample
    if src=="Upload":
        uploaded = st.file_uploader("File", type=["csv","txt","xlsx","xls"]) 
    elif src=="Paste":
        pasted = st.text_area("Paste table (CSV/TSV)")
    else:
        st.caption("Sample dataset: Demo_dataset_1.xlsx")
    st.header("Preprocess")
    smooth = st.checkbox("Smooth", value=True)
    win = st.slider("Window", 1, 21, 5, 2)
    st.header("Time base")
    manual_dt = st.checkbox("Set sampling interval (s)")
    dt_val = st.number_input("Interval (s)", min_value=0.1, value=1.0, step=0.1) if manual_dt else None

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

with st.sidebar:
    st.header("Light")
    light_auto = st.checkbox("Auto-select light column", value=True, key="light_auto")
    light_choice = None
    if not light_auto:
        light_opts = [c for c in ["Qin","Q","PARi","PPFD","par","Light","Rabs"] if c in df.columns]
        light_choice = st.selectbox("Light column", options=light_opts) if light_opts else None
q_col = choose_light_column(df) if light_auto else light_choice

## placeholder; experiment metrics will render after step detection below

st.subheader("Model Fitting")

with st.sidebar:
    st.header("Fitting")
    fit_var = st.selectbox("Variable", options=[c for c in ["gsw","A","Ci"] if c in df.columns])
    pts_avg = st.number_input("Points to average", min_value=3, value=5, step=1)
    auto_steps = st.checkbox("Auto-detect steps", value=True, key="auto_steps")
    phase_mode = st.radio("Phase window", ["Auto Increase","Auto Decrease","Custom"], index=0)

steps = {"up": None, "down": None}
low_level = None
high_level = None
if q_col:
    steps = detect_steps(df, q_col, time_col) if auto_steps else steps
    low_level, high_level = classify_levels(df, q_col)
    df["Phase_auto"] = build_phase_mask(len(df), steps.get("up"), steps.get("down"))

left_options = [c for c in ["A", q_col, "Ci", "gsw"] if c in df.columns]
default_left_idx = 0
if "A" not in df.columns and q_col in left_options:
    default_left_idx = left_options.index(q_col)
left = st.selectbox("Left axis", options=left_options, index=default_left_idx)
right_options = [c for c in ["gsw","A","Ci", q_col] if c in df.columns and c!=left]
right_default_idx = 0 if "gsw" in right_options else 0
right = st.selectbox("Right axis", options=["None"] + right_options, index=(right_default_idx+1 if right_options else 0))
hover_cols = [c for c in ["f_red","f_blue","f_farred","Phase","Phase_auto", q_col] if c in df.columns]

up_idx = steps.get("up")
down_idx = steps.get("down")
t_full = df[time_col].values.astype(float)
y_full = pd.to_numeric(df[fit_var], errors="coerce").astype(float).values
if phase_mode == "Auto Increase" and up_idx is not None:
    start_t = float(t_full[up_idx])
    end_t = float(t_full[down_idx]) if down_idx is not None else float(t_full[-1])
elif phase_mode == "Auto Decrease" and down_idx is not None:
    start_t = float(t_full[down_idx])
    end_t = float(t_full[-1])
else:
    start_t = float(t_full[0])
    end_t = float(t_full[-1])

with st.sidebar:
    st.header("Window")
    win_sel = st.slider("Time range (min)", min_value=float(t_full[0]), max_value=float(t_full[-1]), value=(start_t, end_t))

mask_win = (t_full>=win_sel[0]) & (t_full<=win_sel[1])
t = t_full[mask_win]
y = y_full[mask_win]

res_inc = None
res_dec = None
fig = make_timeseries(df, time_col, left, None if right=="None" else right, steps=steps, hover_extra=hover_cols)
if up_idx is not None:
    t0 = float(df[time_col].iloc[0])
    tu = float(df[time_col].iloc[up_idx])
    te = float(df[time_col].iloc[-1])
    if down_idx is not None:
        td = float(df[time_col].iloc[down_idx])
    else:
        td = None
    fig.add_vrect(x0=t0, x1=tu, fillcolor="#cce5ff", opacity=0.3, layer="below", line_width=0, annotation_text="Low Light", annotation_position="top left")
    fig.add_vrect(x0=tu, x1=(td if td is not None else te), fillcolor="#ffe0b2", opacity=0.3, layer="below", line_width=0, annotation_text="High Light", annotation_position="top")
    if td is not None:
        fig.add_vrect(x0=td, x1=te, fillcolor="#cce5ff", opacity=0.3, layer="below", line_width=0, annotation_text="Low Light 2", annotation_position="top right")
fig.add_vrect(x0=win_sel[0], x1=win_sel[1], fillcolor="#9ad0ff", opacity=0.25, line_width=0)
if up_idx is not None:
    li = max(0, up_idx - int(pts_avg))
    hi = down_idx if down_idx is not None else len(df)
    gsmin = float(pd.Series(y_full[max(0,up_idx-pts_avg):up_idx]).dropna().tail(int(pts_avg)).mean())
    gsmax = float(pd.Series(y_full[max(0,hi-pts_avg):hi]).dropna().tail(int(pts_avg)).mean())
    tt = t[(t>=t_full[up_idx]) & (t<= (t_full[down_idx] if down_idx is not None else t_full[-1]))]
    yy = y[(t>=t_full[up_idx]) & (t<= (t_full[down_idx] if down_idx is not None else t_full[-1]))]
    if len(tt) >= 5 and np.isfinite(gsmin) and np.isfinite(gsmax) and (gsmax-gsmin)>1e-4:
        res_inc = fit_step_increase(tt - tt[0], yy, gsmin, gsmax)
        if res_inc.get("ok",0)==1:
            yhat = gsmin + (gsmax-gsmin)*(1.0 - np.exp(-(np.clip((tt-tt[0]) - res_inc["tlag"], 0, None))/np.maximum(res_inc["tau"],1e-6)))
            yaxis_fit = "y2" if right != "None" and fit_var == right else "y"
            overlay_fit(fig, tt, yhat, name="Fit Increase", yaxis=yaxis_fit, color="#228B22")
if down_idx is not None:
    lo = down_idx
    hi2 = len(df)
    gsmax2 = float(pd.Series(y_full[max(0,down_idx-pts_avg):down_idx]).dropna().tail(int(pts_avg)).mean())
    gsmin2 = float(pd.Series(y_full[max(0,hi2-pts_avg):hi2]).dropna().tail(int(pts_avg)).mean())
    tt2 = t[(t>=t_full[down_idx]) & (t<= t_full[-1])]
    yy2 = y[(t>=t_full[down_idx]) & (t<= t_full[-1])]
    if len(tt2) >= 5 and np.isfinite(gsmin2) and np.isfinite(gsmax2) and (gsmax2-gsmin2)>1e-4:
        res_dec = fit_step_decrease(tt2 - tt2[0], yy2, gsmin2, gsmax2)
        if res_dec.get("ok",0)==1:
            yhat2 = gsmin2 + (gsmax2-gsmin2)*np.exp(-(np.clip((tt2-tt2[0]) - res_dec["tlag"], 0, None))/np.maximum(res_dec["tau"],1e-6))
            yaxis_fit2 = "y2" if right != "None" and fit_var == right else "y"
            overlay_fit(fig, tt2, yhat2, name="Fit Decrease", yaxis=yaxis_fit2, color="#ff8c00")

st.plotly_chart(fig, use_container_width=True)

def fmt_res(r):
    if not r or r.get("ok",0)==0:
        return None
    out = {}
    for k in ["tlag","tau","rmse","r2"]:
        if k in r:
            out[k] = r.get(k)
    return out

tbl_inc = fmt_res(res_inc)
tbl_dec = fmt_res(res_dec)
def fmt_table(d, label):
    if not d:
        return pd.DataFrame()
    disp = {}
    if "tlag" in d:
        disp["(tlag)"] = d["tlag"]
    if "tau" in d:
        disp["(τ)"] = d["tau"]
    if "rmse" in d:
        disp["RMSE"] = d["rmse"]
    if "r2" in d:
        disp["R²"] = d["r2"]
    return pd.DataFrame([disp]).T.rename(columns={0:label})

cols2 = st.columns(2)
if tbl_inc:
    cols2[0].write(fmt_table(tbl_inc, "Increase"))
else:
    cols2[0].info("Increase: not fitted")
if tbl_dec:
    cols2[1].write(fmt_table(tbl_dec, "Decrease"))
else:
    cols2[1].info("Decrease: not fitted")

warns = []
if tbl_inc:
    if tbl_inc.get("tau1",0) > 60 or tbl_inc.get("tau2",0) > 180:
        warns.append("Increase tau unusually large")
    if tbl_inc.get("tlag",0) > 10:
        warns.append("Increase lag unusually large")
    if tbl_inc.get("r2",1) < 0.2:
        warns.append("Increase fit R2 low")
if tbl_dec:
    if tbl_dec.get("tau",0) > 180:
        warns.append("Decrease tau unusually large")
    if tbl_dec.get("tlag",0) > 10:
        warns.append("Decrease lag unusually large")
    if tbl_dec.get("r2",1) < 0.2:
        warns.append("Decrease fit R2 low")
if len(warns)>0:
    for w in warns:
        st.warning(w)

st.subheader("Metrics")
metrics_out = None
if all(c in df.columns for c in ["A","gsw"]):
    if up_idx is not None:
        hi = down_idx if down_idx is not None else len(df)
        A_ser = pd.to_numeric(df["A"], errors="coerce").astype(float).values
        gs_ser = pd.to_numeric(df["gsw"], errors="coerce").astype(float).values
        A0 = float(pd.Series(A_ser[max(0,up_idx-pts_avg):up_idx]).dropna().tail(int(pts_avg)).mean())
        A1 = float(pd.Series(A_ser[max(0,hi-pts_avg):hi]).dropna().tail(int(pts_avg)).mean())
        gs0 = float(pd.Series(gs_ser[max(0,up_idx-pts_avg):up_idx]).dropna().tail(int(pts_avg)).mean())
        gs1 = float(pd.Series(gs_ser[max(0,hi-pts_avg):hi]).dropna().tail(int(pts_avg)).mean())
        t_ind = t_full[up_idx:hi]
        A_ind = A_ser[up_idx:hi]
        gs_ind = gs_ser[up_idx:hi]
        pA = percent_times(t_ind - t_ind[0], A_ind, A0, A1, [0.1,0.5,0.9])
        pG = percent_times(t_ind - t_ind[0], gs_ind, gs0, gs1, [0.1,0.5,0.9])
        dA = deficit(t_ind - t_ind[0], A_ind, A1)
        dG = deficit(t_ind - t_ind[0], gs_ind, gs1)
        sLmax, sLmean = sl_approx(t_ind - t_ind[0], A_ind, gs_ind, A0, A1, gs0, gs1)
        mtab = {
            "A_t10 (increase)": pA.get("t10"),
            "A_t50 (increase)": pA.get("t50"),
            "A_t90 (increase)": pA.get("t90"),
            "gs_t10 (increase)": pG.get("t10"),
            "gs_t50 (increase)": pG.get("t50"),
            "gs_t90 (increase)": pG.get("t90"),
            "A_deficit (increase)": dA,
            "gs_deficit (increase)": dG,
            "SLmax_approx": sLmax,
            "SLmean_approx": sLmean,
        }
        st.write(pd.DataFrame([mtab]).T.rename(columns={0:"value"}))
        metrics_out = mtab
    st.header("Partition")
    do_part = st.checkbox("Compute SL/BL partition", value=True)
    ratio_jv = st.number_input("J:Vc ratio", min_value=0.5, max_value=3.0, value=1.8, step=0.05)
    alpha_ph = st.number_input("Alpha", min_value=0.05, max_value=0.4, value=0.24, step=0.01)
    theta_ph = st.number_input("Theta", min_value=0.5, max_value=0.99, value=0.9, step=0.01)
    use_temp = st.checkbox("Temperature corrections", value=True)
    set_rd = st.checkbox("Override Rd", value=False)
    rd_val = st.number_input("Rd (µmol m⁻² s⁻¹)", min_value=0.0, value=1.0, step=0.1) if set_rd else None
    set_vcjm = st.checkbox("Override Vcmax25/Jmax25", value=False)
    vc_in = st.number_input("Vcmax25 (µmol m⁻² s⁻¹)", min_value=5.0, max_value=400.0, value=60.0, step=5.0) if set_vcjm else None
    jm_in = st.number_input("Jmax25 (µmol m⁻² s⁻¹)", min_value=10.0, max_value=600.0, value=100.0, step=5.0) if set_vcjm else None

partition_out = None
if do_part and q_col is not None:
    st.subheader("Stomatal/Biochemical Limitations")
    pars = calibrate_params(df, time_col, q_col, pts_avg=int(pts_avg), ratio_jv=ratio_jv, alpha=alpha_ph, theta=theta_ph, use_temp=use_temp, rd_override=rd_val, vcmax25_override=vc_in, jmax25_override=jm_in)
    df_part = partition_limits(df, time_col, q_col, pars)
    st.write({k: round(float(v),3) for k,v in pars.items()})
    fig2 = make_timeseries(df_part, time_col, "SL", "BL", steps=steps, hover_extra=[q_col], y1_name="SL", y2_name="BL")
    st.plotly_chart(fig2, use_container_width=True)
    slmax = float(pd.to_numeric(df_part["SL"], errors="coerce").max())
    blmax = float(pd.to_numeric(df_part["BL"], errors="coerce").max())
    st.write({"SLmax": slmax, "BLmax": blmax})
    partition_out = {"params": pars, "SLmax": slmax, "BLmax": blmax}

    from io import StringIO
    csv_buf = StringIO()
    df_part.to_csv(csv_buf, index=False)
    st.download_button("Download partition CSV", data=csv_buf.getvalue(), file_name="partition_output.csv", mime="text/csv")

csv_all = StringIO()
df.to_csv(csv_all, index=False)
st.download_button("Download processed CSV", data=csv_all.getvalue(), file_name="processed_output.csv", mime="text/csv")

report = {
    "timestamp": datetime.now().isoformat(),
    "source": {"mode": src, "file": uploaded.name if src=="Upload" and uploaded is not None else (str(sample_path) if src=="Sample" else None)},
    "columns": list(df.columns),
    "time_col": time_col,
    "light_col": q_col,
    "steps": steps,
    "fit": {"increase": tbl_inc, "decrease": tbl_dec},
    "metrics": metrics_out,
    "partition": partition_out,
    "warnings": warns,
}
rep_buf = StringIO()
rep_buf.write(json.dumps(report, default=float, indent=2))
st.download_button("Download JSON report", data=rep_buf.getvalue(), file_name="kgs_pro_report.json", mime="application/json")

st.markdown("<div style='text-align: right; font-weight: bold;'>© Mengjie Fan</div>", unsafe_allow_html=True)

tabs_refs = st.tabs(["References"])
with tabs_refs[0]:
    lit_dir = Path(__file__).parent / "literatures"
    if lit_dir.exists():
        for p in sorted(lit_dir.glob("*.pdf")):
            with open(p, "rb") as f:
                st.download_button(label=str(p.name), data=f.read(), file_name=p.name, mime="application/pdf")
