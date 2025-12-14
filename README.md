# KGS-PRO V2
For analysis of dynamic photosynthesis (From LI‑6800 raw files),hosted on Streamlit (for dev preview).
#### © Mengjie Fan 2024

## Batch demo (new 2025 updates)
### Batch processing is now available in V2 using any of the calculator.  
- Process limitless files (RAM limiting) with MF/XL/None calculators using parallel cores, 
- I also included string split features for grouping columns, and then cached merged data
- Exports: you can download merged CSV/XLSX with file_id, per-file summaries/errors, processed/partition outputs
- I also added a progress UI with throughput and ETA (the estimation logic should be sound if all files are equal size); sample raw files supported in batch runs
![Batch processing](images/batch_processing%20DEMO.gif)

## Run
- From repo root: `pip install -r KGS-PRO-V1/requirements.txt`
- Launch: `streamlit run KGS-PRO-V1/app.py`

## Core flow
- Import LI-6800 CSV/XLSX or pasted tables; sample data included; column synonym mapping and time inference with manual interval fallback
- (Single analysis only for the moment)Detect PPFD step up/down, tag phases, derive low/high levels; optional smoothing and guardrails to skip tiny signals
- Fit single-tau stomatal kinetics for increase/decrease; gsmin/gsmax from phase endpoints; show RMSE/R2 overlays
- Partition A into stomatal vs biochemical limitation via minimal Farquhar calibration; SL/BL time series and peaks
- Derived metrics: t10/t50/t90 for A and gsw, deficits vs steady state, SLmax_approx


## Phase Detection should work most of the time (not tested for diurnal program yet)
 from smoothed PPFD (Qin or best alternative) using plateau aware crossings with fallback to gradient extrema in some cases

## Kinetics & Fitting (gsw)
- Step increase (induction):
  - g(t) = gs_min + Δgs · [1 − exp(−(t − t_lag)/τ)]
- Step decrease (relaxation):
  - g(t) = gs_min + Δgs · exp(−(t − t_lag)/τ)
- gs_min and gs_max estimated from endpoints of each phase; τ (time constant) and t_lag fitted with least squares and sensible bounds.

## Some other useful Metrics for dyn photosynthesis (Induction)
- Percent times: t10, t50, t90 to reach 10/50/90% of Δ under increase.
- Deficits: ∫(A_max − A) dt and ∫(gs_max − gsw) dt during induction.
- SLmax (approx): max[1 − (A_norm/gs_norm)] during induction.

## Stomatal vs Biochemical Limitations (Partition)
- Farquhar‑type model (temperature corrected) , can be from ur ACi measurement:
  - Ac = Vcmax · (Ci − Γ*) / (Ci + Kc·(1 + O/Ko))
  - Aj = J · (Ci − Γ*) / (4·Ci + 8·Γ*),  with J from a non‑rectangular hyperbola (α, θ)
  - A_model = min(Ac, Aj) − Rd
- High‑light calibration at steady state: optional overrides for Vcmax25/Jmax25 and Rd.
- Limitations (time‑resolved), see paper in the reference tab to download (scroll down to the bottom of the app):
  - SL = (A_pot − A) / A_max
  - BL = (A_max − A_pot) / A_max
  - TL = (A_max − A) / A_max

- Reference PDFs are bundled under “References” in the app for quick download.