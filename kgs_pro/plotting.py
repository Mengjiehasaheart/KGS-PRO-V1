import pandas as pd
import numpy as np
import plotly.graph_objs as go
from typing import Optional, List

def make_timeseries(df: pd.DataFrame, x: str, y1: str, y2: Optional[str]=None, steps: Optional[dict]=None, hover_extra: Optional[List[str]]=None, y1_name: Optional[str]=None, y2_name: Optional[str]=None):
    hover = [c for c in (hover_extra or []) if c in df.columns]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y1], mode="lines+markers", name=y1_name or y1, hovertemplate=None, hoverinfo="skip"))
    if y2 and y2 in df.columns:
        fig.add_trace(go.Scatter(x=df[x], y=df[y2], mode="lines+markers", name=y2_name or y2, yaxis="y2", hovertemplate=None, hoverinfo="skip"))
    ht = []
    for i in range(len(df)):
        s = []
        s.append(f"{x}: {df.iloc[i][x]}")
        s.append(f"{y1}: {df.iloc[i][y1]}")
        if y2 and y2 in df.columns:
            s.append(f"{y2}: {df.iloc[i][y2]}")
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
    fig.update_layout(xaxis=dict(title=x), yaxis=dict(title=y1_name or y1))
    if y2 and y2 in df.columns:
        fig.update_layout(yaxis2=dict(title=y2_name or y2, overlaying="y", side="right"))
    fig.update_layout(legend=dict(orientation="h"))
    return fig

def overlay_fit(fig, t, yhat, name, yaxis="y"):
    fig.add_trace(go.Scatter(x=t, y=yhat, mode="lines", name=name, yaxis=yaxis))
    return fig

