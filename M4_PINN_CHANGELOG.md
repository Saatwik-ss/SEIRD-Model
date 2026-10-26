# Module 4 (PINN) rebuild: change log, 26 October 2026

Module 4's contribution (tracker tasks T27-T35: PINN design, E2-E4, the loss-weight ablation, PINN limitations) was rebuilt from scratch so the PINN actually works. Modules 1-3 were not changed. Module 5's baselines were copied unchanged; only the PINN rows of E5 and the E5 interpretation changed (tracker T39: "Run M4's PINN on real data").
The earlier audit and re-run of the old code is recorded in `PINN_RERUN_CHANGELOG.md`.

## 1. Results

| Experiment | Old PINN | New PINN |
|---|---|---|
| **E2** forward solve, rel. L2 error vs DOP853 (3 seeds) | 4%-82% | **0.035%-0.094%** (worst seed 0.15%) |
| **E3** β(t) from noisy cases + deaths only, rel. L2 (5 seeds) | 42% (flat line, failed; with all 5 compartments observed) | **10.8% (1% noise), 12.0% (5%)** |
| **E3** death rate μ | not estimated | **4.4% / 5.9%** error |
| **E4** hidden s(t), e(t), i(t) | s 2.8%, e 30% (with true β given) | **s 0.3-0.5%, e and i 5.5-6.8%** (β, μ unknown) |
| **E5** held-out days 90-119, case rel. error | 13.4% | **9.4%** (piecewise-β 9.8%, constant-β 55.3%) |
| **E5** deaths rel. error | 1.5% | 8.3% (piecewise-β **1.2%**) |
| **E7** (new) 8 rolling 4-week forecasts, WHO 2014-15, cases MAE | - | **100.7** (piecewise-β 108.1, constant-β 120.9) |
| **E7** deaths MAE | - | 70.5 (piecewise-β **42.0**, constant-β 50.7) |

E6 (ablation, replaces the old figure that no code produced): without L-BFGS the hidden-state error is 96%; physics weight 10 instead of 1 roughly triples the errors; Fourier features cut the β error by about 25%; absolute residuals make the forward solve fail (75-100% error).

Summary: the PINN is accurate on synthetic problems and forecasts **cases** at least as well as the best classical baseline on real data, but forecasts **deaths** worse.

## 2. Files

**New**
- `seird_pinn.py`: the PINN (model, loss, Adam then L-BFGS training, forecasting).
- `pinn_experiments.py`: E2, E3/E4, E5 (PINN + Module 5's baselines), E6, E7; saves `results/m4_*.json`.
- `pinn_figures.py`: figures (.eps without transparency, and .png).
- `data_pipeline_who.py`, `data/who_hdx_ebola_data_db_format.csv` (unchanged WHO file), `data/sierra_leone_who_2014_2015.csv` (processed).
- `results/m4_e2_forward.json`, `m4_e3_e4_inverse.json`, `m4_e5_heldout.json`, `m4_e6_ablation.json`, `m4_e6_forward_absolute.json`, `m4_e7_rolling.json`.
- `figs/fig_e7_rolling_forecasts.eps/.png`.
- `report/ai_use_member4.md` (Member 4 must complete the last section).

**Replaced figures:** `fig_e2_forward_pinn`, `fig_e3_inverse_beta`, `fig_e4_hidden_states`, `fig_e5_real_forecast`, `fig_e6_ablations` (.eps and .png).

**Removed (old Module 4 code and its outputs; still in git history):** `pinn_model.py`, `run_pinn_pipeline.py`, `pinn_complete.py`, `generate_metric_evaluations.py`, `scratch/inspect_pinn_module.py`, `scratch/test_proper_scaling.py`, `scratch/update_cell_43.py`, `figs/evaluation_metrics.csv`, `figs/fig_metrics_*` (8 files).

**Edited**
- `notebook.ipynb`: Module 4 cells replaced (design, algorithm, E2, E3/E4); the E5 cell now runs the experiment instead of showing a typed-in table; new E6 and E7 cells; the Module 4 row of the overview table, the E4/E5/E6 bullets of the interpretation cell and the AI disclosure updated. With `RUN_PINN_EXPERIMENTS = False` the cells load `results/` (seconds); `True` retrains (about 30 minutes on 8 cores). All new cells were run against the saved results without errors; the full notebook was not re-run top to bottom.
- `report/main.tex`: abstract (PINN sentences), Sections "PINN Methodology", "Synthetic Experiments", the PINN parts of "Real-World Outbreak Application" (table, interpretation, new E7 subsection), "Ablation Studies (E6)", Limitations items 3-6, and a Module 4 line in the AI acknowledgement. Labels, citations, environments and figure paths were checked; **the PDF was not recompiled** (no LaTeX installation here).
- `report/refs.bib`: added the WHO HDX dataset, Tancik et al. 2020, Rahaman et al. 2019, Kingma and Ba 2015, Liu and Nocedal 1989.
- `data/README.md`: added the WHO dataset section and the verification of the Module 5 file.

## 3. Data

- **Module 5's file** (`data/country_timeseries_raw.csv`) is the content of `cmrivers/ebola` `country_timeseries.csv` (commit `2c696c79c72c49ab7a178bf4f4e391613aa42f18`), compiled from WHO situation reports. Its Sierra Leone values match WHO Disease Outbreak News of 31 Jul 2014 (533 cases, 233 deaths) and 22 Aug 2014 (910, 392). Only 37 of its 120 days are real report dates; the new PINN fits only those.
- **New WHO file:** HDX dataset `ebola-cases-2014`, file `ebola_data_db_format.csv`, SHA-256 `053c325038b419619162482b8ab1e98bed66d80cc1929ed7862367fc9bf4b2fa`. Laboratory-confirmed Sierra Leone cases and deaths, 29 Aug 2014 - 6 Nov 2015, 238 report dates; 6 death reports that WHO later revised downwards are excluded.

## 4. How the PINN works (short)

- Model written with C = I + R + D in place of R, so conservation is exact and both observed series are network outputs.
- Exponential outputs with hard initial conditions; residuals divided by the state (relative errors).
- Fourier features (4 frequencies) against spectral bias; CFR bounded to 20-80% to block a spurious "μ → 0, I → ∞" solution.
- Adam (2000 iterations) then L-BFGS (2500-6000), double precision, several seeds.
- Forecast: PINN end state, β held at its last-14-day mean, learned μ, integrated with DOP853.

## 5. How settings were chosen (no test data used)

- Synthetic: smoothing 1e-4, 4 Fourier frequencies, 6000 L-BFGS iterations, tuned on seed 1; results reported on seeds 42-46.
- Real data: smoothing 1e-3, 4 Fourier frequencies, 14-day β window, chosen by mean case relative error on three validation origins (24 Oct, 21 Nov, 19 Dec 2014) and an E5 fit on days 0-59 scored on days 60-89.

## 6. Dropped to save time

A sinusoidal β(t) scenario (β error stayed about 26% even without noise: the oscillation changes cumulative cases by only about 1%, and the smoothness penalty favours a flat β), a naive trend baseline in E7, and validation-origin rows in E7.

## 7. To do before presenting

- **Member 4** completes `report/ai_use_member4.md` honestly and makes sure they can explain every equation and code component (project instruction 4).
- Compile `report/main.tex` and check the layout.
- Update the Module 4 and Module 5 slides with the new results.
- If the report was already submitted on 25 October, ask the instructor whether a revised version can be submitted.
- Reproduce: `pip install numpy pandas scipy matplotlib torch` (and Jupyter), then `python pinn_experiments.py all --force`.
