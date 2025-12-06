import streamlit as st
import pandas as pd
import numpy as np
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO, BytesIO
from pathlib import Path
from kgs_pro.io import read_any, read_pasted_table, standardize_columns, choose_light_column
from kgs_pro.preprocess import infer_time, rolling_smooth
from kgs_pro.calc import ensure_derived
from kgs_pro.detect import detect_steps, classify_levels, build_phase_mask
from kgs_pro.models import fit_step_increase, fit_step_decrease
from kgs_pro.plotting import make_timeseries, overlay_fit
from kgs_pro.metrics import percent_times, deficit, sl_approx
from kgs_pro.partition import calibrate_params, partition_limits
from kgs_pro.pipeline import process_source, process_bytes_job

st.set_page_config(page_title="KGS-PRO V2", layout="wide")

st.title("KGS-PRO V2: Dynamic Photosynthesis Analysis")
st.markdown("<div style='text-align: right; font-size: 12px; font-weight: bold; color: #333;'>© Mengjie Fan</div>", unsafe_allow_html=True)

def split_groups(name: str, delim: str, count: int):
    if count <= 0:
        return []
    parts = name.split(delim) if delim else [name]
    if len(parts) < count:
        parts = parts + [""]*(count - len(parts))
    else:
        parts = parts[:count]
    return parts

def build_group_names(raw_names: str, count: int):
    names = [n.strip() for n in raw_names.split(",") if n.strip()]
    out = []
    for i in range(count):
        out.append(names[i] if i < len(names) else f"group_{i+1}")
    return out

def merge_results(successes, delim, group_count, group_names):
    if not successes:
        return None
    merged_frames = []
    seen = {}
    for res in successes:
        if res.df is None:
            continue
        df = res.df.copy()
        df.columns = pd.Index([str(c) for c in df.columns], dtype=object)
        df = df.loc[:, ~df.columns.duplicated()]
        base = Path(res.filename).stem
        seen[base] = seen.get(base, 0) + 1
        file_id = base if seen[base] == 1 else f"{base}_{seen[base]}"
        df.insert(0, "file_id", file_id)
        groups = split_groups(base, delim, group_count)
        for idx, val in enumerate(groups):
            col_name = group_names[idx] if idx < len(group_names) else f"group_{idx+1}"
            df.insert(1 + idx, col_name, val)
        merged_frames.append(df)
    if not merged_frames:
        return None
    merged = pd.concat(merged_frames, ignore_index=True, sort=False)
    merged.columns = pd.Index([str(c) for c in merged.columns], dtype=object)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged

def build_excel_buffer(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf

def ensure_time_min(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "time_min" in d.columns:
        return d
    if "elapsed" in d.columns:
        d["time_min"] = pd.to_numeric(d["elapsed"], errors="coerce")/60.0
    elif "elapsed_s" in d.columns:
        d["time_min"] = pd.to_numeric(d["elapsed_s"], errors="coerce")/60.0
    elif "time_s" in d.columns:
        d["time_min"] = pd.to_numeric(d["time_s"], errors="coerce")/60.0
    return d

def ensure_wue(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "WUE" not in d.columns and all(c in d.columns for c in ["A","gsw"]):
        A = pd.to_numeric(d["A"], errors="coerce")
        g = pd.to_numeric(d["gsw"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            w = A/g
            w = w.replace([np.inf, -np.inf], np.nan)
            d["WUE"] = w
    return d

def format_fig(fig, xlab, ylab):
    fig.update_layout(
        xaxis_title=xlab,
        yaxis_title=ylab,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13, color="#111", family="Arial"),
        legend_title=None,
    )
    fig.update_xaxes(showgrid=False, showline=True, linewidth=2, linecolor="#111", mirror=True, title_font=dict(size=14, color="#111"))
    fig.update_yaxes(showgrid=False, showline=True, linewidth=2, linecolor="#111", mirror=True, title_font=dict(size=14, color="#111"))
    return fig

def aggregate_timeseries(df: pd.DataFrame, value: str, group_cols, time_col: str="time_min"):
    cols_need = [c for c in group_cols if c in df.columns]
    cols = [time_col, value] + cols_need
    data = df[cols].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna(subset=[value, time_col])
    grp_keys = cols_need + [time_col] if cols_need else [time_col]
    agg = data.groupby(grp_keys)[value].agg(["mean","count","std"]).reset_index()
    agg["sem"] = agg["std"]/np.sqrt(agg["count"].replace(0,np.nan))
    return agg

def plot_aggregated(agg: pd.DataFrame, value: str, group_cols, time_col: str="time_min"):
    import plotly.graph_objects as go
    fig = go.Figure()
    if group_cols:
        uniq = agg[group_cols].drop_duplicates()
        for _, row in uniq.iterrows():
            mask = pd.Series(True, index=agg.index)
            for g in group_cols:
                mask &= agg[g] == row[g]
            sub = agg.loc[mask].sort_values(time_col)
            name = " / ".join([str(row[g]) for g in group_cols])
            fig.add_trace(go.Scatter(x=sub[time_col], y=sub["mean"], mode="lines", name=name))
            if "sem" in sub:
                fig.add_trace(go.Scatter(x=pd.concat([sub[time_col], sub[time_col][::-1]]), y=pd.concat([sub["mean"]-sub["sem"], (sub["mean"]+sub["sem"])[::-1]]), fill="toself", mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False, opacity=0.12, name=None))
    else:
        sub = agg.sort_values(time_col)
        fig.add_trace(go.Scatter(x=sub[time_col], y=sub["mean"], mode="lines", name=value))
        if "sem" in sub:
            fig.add_trace(go.Scatter(x=pd.concat([sub[time_col], sub[time_col][::-1]]), y=pd.concat([sub["mean"]-sub["sem"], (sub["mean"]+sub["sem"])[::-1]]), fill="toself", mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False, opacity=0.12, name=None))
    fig = format_fig(fig, "Time (min)", value)
    return fig

mode = st.sidebar.radio("Mode", ["Single Analysis", "Batch Processing"], index=0)

if mode == "Batch Processing":
    merged_cached = st.session_state.get("batch_merged")
    summary_cached = st.session_state.get("batch_summary")
    errors_cached = st.session_state.get("batch_errors")
    with st.sidebar:
        st.header("Batch")
        calc_batch = st.radio("Calculator", ["MF Calculator", "XL Calculator"], index=0, key="calc_batch")
        calc_engine_batch = "mf" if calc_batch == "MF Calculator" else "xlcalculator"
        uploads = st.file_uploader("Files", type=["csv", "txt", "xlsx", "xls"], accept_multiple_files=True, key="batch_uploads")
        use_sample = st.checkbox("Include sample raw files", value=False, key="batch_use_sample")
        smooth_batch = st.checkbox("Smooth", value=True, key="batch_smooth")
        win_batch = st.slider("Window", 1, 21, 5, 2, key="batch_win")
        st.header("Time base")
        manual_dt_batch = st.checkbox("Set sampling interval (s)", key="batch_manual_dt")
        dt_batch = st.number_input("Interval (s)", min_value=0.1, value=1.0, step=0.1, key="batch_dt") if manual_dt_batch else None
        cores_max = max(1, multiprocessing.cpu_count())
        cores_default = min(4, cores_max)
        cores_batch = st.slider("CPU cores", 1, cores_max, cores_default, 1, key="batch_cores")
        st.header("Grouping")
        delim = st.text_input("Filename delimiter", value="-", key="batch_delim")
        group_count = st.number_input("Grouping parts", min_value=0, max_value=20, value=0, step=1, key="batch_group_count")
        group_names_raw = st.text_input("Group names (comma-separated)", value="", key="batch_group_names")
    st.subheader("Batch Processing")
    st.caption("Upload or load sample files, select calculator, and process up to 1000 files with parallel workers.")
    process_btn = st.button("Process batch", key="batch_process")
    if process_btn:
        files = []
        if uploads:
            for f in uploads:
                files.append((f.getvalue(), f.name))
        if use_sample:
            sample_dir = Path("Sample_Dataset/Raw_licor_files")
            if sample_dir.exists():
                for p in sorted(sample_dir.glob("*")):
                    if p.is_file():
                        files.append((p.read_bytes(), p.name))
        total = len(files)
        if total == 0:
            st.warning("No files to process")
            st.stop()
        if total > 1000:
            files = files[:1000]
            total = len(files)
            st.warning("Processing capped at 1000 files per batch")
        group_names = build_group_names(group_names_raw, int(group_count))
        st.write({"queued_files": total, "calculator": calc_batch, "cores": int(cores_batch)})
        successes = []
        errors = []
        summary_rows = []
        start_time = time.time()
        progress = st.progress(0.0)
        with st.status("Processing batch...", expanded=True) as status:
            status_text = st.empty()
            if int(cores_batch) == 1:
                for idx, (data, name) in enumerate(files, start=1):
                    res = process_source(data, name, calc_engine_batch, dt_batch, smooth_batch, win_batch, None, True, True)
                    if res.error or res.df is None:
                        errors.append({"file": name, "error": res.error if res.error else "Unknown failure"})
                    else:
                        successes.append(res)
                        summary_rows.append({
                            "file": name,
                            "rows": len(res.df),
                            "columns": len(res.df.columns),
                            "time_col": res.time_col,
                            "light_col": res.light_col,
                            "step_up": res.steps.get("up"),
                            "step_down": res.steps.get("down")
                        })
                    elapsed = time.time() - start_time
                    rate = idx/elapsed if elapsed > 0 else 0.0
                    remaining = (total - idx)/rate if rate > 0 else None
                    progress.progress(idx/total)
                    status_text.write(f"{idx}/{total} files processed ({rate:.2f}/s" + (f", eta {remaining:.1f}s" if remaining else "") + f") latest: {name}")
            else:
                with ThreadPoolExecutor(max_workers=int(cores_batch)) as ex:
                    futures = {ex.submit(process_bytes_job, (data, name, calc_engine_batch, dt_batch, smooth_batch, win_batch)): name for data, name in files}
                    for idx, fut in enumerate(as_completed(futures), start=1):
                        try:
                            res = fut.result()
                        except Exception as e:
                            name = futures[fut]
                            errors.append({"file": name, "error": str(e)})
                            res = None
                        if res is not None:
                            name = res.filename
                            if res.error or res.df is None:
                                errors.append({"file": name, "error": res.error if res.error else "Unknown failure"})
                            else:
                                successes.append(res)
                                summary_rows.append({
                                    "file": name,
                                    "rows": len(res.df),
                                    "columns": len(res.df.columns),
                                    "time_col": res.time_col,
                                    "light_col": res.light_col,
                                    "step_up": res.steps.get("up"),
                                    "step_down": res.steps.get("down")
                                })
                        elapsed = time.time() - start_time
                        rate = idx/elapsed if elapsed > 0 else 0.0
                        remaining = (total - idx)/rate if rate > 0 else None
                        progress.progress(idx/total)
                        latest_name = name if res is not None else futures[fut]
                        status_text.write(f"{idx}/{total} files processed ({rate:.2f}/s" + (f", eta {remaining:.1f}s" if remaining else "") + f") latest: {latest_name}")
            status.update(label="Processing complete", state="complete")
        st.success(f"Finished {len(successes)} files, {len(errors)} errors")
        if summary_rows:
            st.write(pd.DataFrame(summary_rows))
        if errors:
            st.error("Files with issues")
            st.write(pd.DataFrame(errors))
        merged = merge_results(successes, delim, int(group_count), group_names)
        if merged is not None:
            merged = ensure_time_min(merged)
            merged = ensure_wue(merged)
            st.session_state["batch_merged"] = merged
            st.session_state["batch_summary"] = summary_rows
            st.session_state["batch_errors"] = errors
            merged_cached = merged
            summary_cached = summary_rows
            errors_cached = errors
        st.success("Batch processing stored; see analysis below.")

    if merged_cached is not None:
        st.subheader("Merged dataset")
        st.write({"rows": len(merged_cached), "columns": len(merged_cached.columns)})
        st.dataframe(merged_cached.head())
        csv_buf = StringIO()
        merged_cached.to_csv(csv_buf, index=False)
        st.download_button("Download merged CSV", data=csv_buf.getvalue(), file_name="merged_processed.csv", mime="text/csv")
        xlsx_buf = build_excel_buffer(merged_cached)
        st.download_button("Download merged XLSX", data=xlsx_buf.getvalue(), file_name="merged_processed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.subheader("Batch analysis")
        time_col_batch = "time_min" if "time_min" in merged_cached.columns else None
        if time_col_batch is None:
            st.warning("time_min not found; cannot plot batch averages")
        else:
            numeric_cols = [c for c in merged_cached.columns if merged_cached[c].dtype!=object and c not in ["file_id"]]
            if "WUE" in merged_cached.columns and "WUE" not in numeric_cols:
                numeric_cols.append("WUE")
            if not numeric_cols:
                st.warning("No numeric columns available for plotting")
                value_col = None
            else:
                default_val = "A" if "A" in numeric_cols else numeric_cols[0]
                value_col = st.selectbox("Value", options=numeric_cols, index=numeric_cols.index(default_val), key="batch_value")
            group_opts = [c for c in merged_cached.columns if c.startswith("group_")] + ["file_id"]
            group_sel = st.multiselect("Group by", options=group_opts, default=[c for c in group_opts if c.startswith("group_")], key="batch_group_sel")
            if value_col:
                agg = aggregate_timeseries(merged_cached, value_col, group_sel, time_col=time_col_batch)
                if len(agg)==0:
                    st.warning("No data to plot for selected value")
                else:
                    fig_batch = plot_aggregated(agg, value_col, group_sel, time_col=time_col_batch)
                    fig_batch = format_fig(fig_batch, "Time (min)", value_col)
                    st.plotly_chart(fig_batch, use_container_width=True)
                    csv_agg = StringIO()
                    agg.to_csv(csv_agg, index=False)
                    st.download_button("Download aggregated CSV", data=csv_agg.getvalue(), file_name="batch_aggregated.csv", mime="text/csv")
            st.markdown("---")
            st.subheader("Per-file plots")
            file_opts = sorted(merged_cached["file_id"].unique().tolist()) if "file_id" in merged_cached.columns else []
            default_vars = []
            if "A" in merged_cached.columns:
                default_vars.append("A")
            if "gsw" in merged_cached.columns:
                default_vars.append("gsw")
            plot_vars = st.multiselect("Variables", options=[c for c in numeric_cols if c not in ["file_id"]], default=default_vars[:2], key="per_file_vars")
            files_pick = st.multiselect("Files", options=file_opts, default=file_opts[:1], key="per_file_files")
            def make_per_file_fig(df, fid, vars_sel):
                import plotly.graph_objects as go
                sub = df[df["file_id"]==fid].copy()
                sub = ensure_time_min(sub)
                sub = ensure_wue(sub)
                fig = go.Figure()
                for v in vars_sel:
                    if v not in sub.columns:
                        continue
                    fig.add_trace(go.Scatter(x=sub["time_min"], y=pd.to_numeric(sub[v], errors="coerce"), mode="lines", name=f"{fid} - {v}"))
                lab_map = {
                    "A":"A (µmol m⁻² s⁻¹)",
                    "gsw":"gsw (mol m⁻² s⁻¹)",
                    "Ci":"Ci (µmol mol⁻¹)",
                    "Rabs":"Rabs (µmol m⁻² s⁻¹)",
                    "WUE":"WUE (µmol CO₂ mol⁻¹ H₂O)"
                }
                ylab = ", ".join([lab_map.get(v,v) for v in vars_sel]) if vars_sel else ""
                fig = format_fig(fig, "Time (min)", ylab)
                return fig
            perfile_figs = []
            if files_pick and plot_vars:
                for fid in files_pick:
                    fig_pf = make_per_file_fig(merged_cached, fid, plot_vars)
                    st.plotly_chart(fig_pf, use_container_width=True)
                    perfile_figs.append((fid, fig_pf))
            if perfile_figs:
                import zipfile
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fid, fig_pf in perfile_figs:
                        try:
                            png_bytes = fig_pf.to_image(format="png", scale=3)
                            zf.writestr(f"{fid}.png", png_bytes)
                        except Exception:
                            continue
                zip_buf.seek(0)
                st.download_button("Download plotted files (PNG, 300 dpi approx)", data=zip_buf.getvalue(), file_name="batch_plots.zip", mime="application/zip")
    st.stop()
    else:
        st.info("Add files and start batch processing")
        st.stop()

with st.sidebar:
    st.header("Data")
    calc_single = st.radio("Calculator", ["MF Calculator", "XL Calculator"], index=0, key="calc_single")
    calc_engine_single = "mf" if calc_single == "MF Calculator" else "xlcalculator"
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
        return read_any(uploaded, uploaded.name, calc_engine=calc_engine_single)
    if src=="Paste" and pasted:
        return read_pasted_table(pasted)
    if src=="Sample" and sample_path.exists():
        return read_any(str(sample_path), calc_engine=calc_engine_single)
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
