import pandas as pd
import numpy as np
import plotly.graph_objs as go
from typing import Optional, List, Dict

DEFAULT_UNITS = {
    "time_min": "min",
    "A": "µmol m⁻² s⁻¹",
    "gsw": "mol m⁻² s⁻¹",
    "Ci": "µmol mol⁻¹",
    "Ca": "µmol mol⁻¹",
    "Qin": "µmol m⁻² s⁻¹",
    "Q": "µmol m⁻² s⁻¹",
    "PARi": "µmol m⁻² s⁻¹",
    "PPFD": "µmol m⁻² s⁻¹",
    "Rabs": "W m⁻²",
}

def _unit_for(col: str, units: Optional[Dict[str,str]] = None) -> str:
    if units and col in units:
        return units[col]
    return DEFAULT_UNITS.get(col, "")

def make_timeseries(df: pd.DataFrame, x: str, y1: str, y2: Optional[str]=None, steps: Optional[dict]=None, hover_extra: Optional[List[str]]=None, y1_name: Optional[str]=None, y2_name: Optional[str]=None, units: Optional[Dict[str,str]]=None):
    hover = [c for c in (hover_extra or []) if c in df.columns]
    fig = go.Figure()
    y1_label = y1_name or (f"{y1} ({_unit_for(y1, units)})" if _unit_for(y1, units) else y1)
    fig.add_trace(go.Scatter(x=df[x], y=df[y1], mode="lines+markers", name=y1_label, hovertemplate=None, hoverinfo="skip"))
    if y2 and y2 in df.columns:
        y2_label = y2_name or (f"{y2} ({_unit_for(y2, units)})" if _unit_for(y2, units) else y2)
        fig.add_trace(go.Scatter(x=df[x], y=df[y2], mode="lines+markers", name=y2_label, yaxis="y2", hovertemplate=None, hoverinfo="skip"))
    ht = []
    for i in range(len(df)):
        s = []
        s.append(f"{x}: {df.iloc[i][x]} {_unit_for(x, units)}")
        s.append(f"{y1}: {df.iloc[i][y1]} {_unit_for(y1, units)}")
        if y2 and y2 in df.columns:
            s.append(f"{y2}: {df.iloc[i][y2]} {_unit_for(y2, units)}")
        for c in hover:
            s.append(f"{c}: {df.iloc[i][c]}")
        ht.append("<br>".join(s))
    fig.update_traces(hovertext=ht, hoverinfo="text")
    if steps:
        if steps.get("up") is not None:
            xv = df[x].iloc[int(steps["up"])]
            fig.add_vline(x=xv, line=dict(color="green", dash="dash"))
        if steps.get("down") is not None:
            xv = df[x].iloc[int(steps["down"])]
            fig.add_vline(x=xv, line=dict(color="red", dash="dash"))
    x_label = f"{x} ({_unit_for(x, units)})" if _unit_for(x, units) else x
    fig.update_layout(
        xaxis=dict(title=dict(text=x_label, font=dict(size=20, color="black", family="Arial Black")), showline=True, linewidth=2, linecolor="black", mirror=True, tickfont=dict(size=16, color="black", family="Arial")),
        yaxis=dict(title=dict(text=y1_label, font=dict(size=20, color="black", family="Arial Black")), showline=True, linewidth=2, linecolor="black", mirror=True, tickfont=dict(size=16, color="black", family="Arial")),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=16, color="black", family="Arial")
    )
    if y2 and y2 in df.columns:
        fig.update_layout(yaxis2=dict(title=dict(text=y2_label, font=dict(size=20, color="black", family="Arial Black")), overlaying="y", side="right", showline=True, linewidth=2, linecolor="black", tickfont=dict(size=16, color="black", family="Arial")))
    fig.update_layout(legend=dict(orientation="h"), font=dict(size=14))
    return fig

def overlay_fit(fig, t, yhat, name, yaxis="y", color=None, width=3):
    fig.add_trace(go.Scatter(x=t, y=yhat, mode="lines", name=name, yaxis=yaxis, line=dict(width=width, color=color)))
    return fig
