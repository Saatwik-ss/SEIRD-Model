# Module 4 (PINN) rebuild: change log, 26 October 2026

Module 4's contribution (tracker tasks T27-T35: PINN design, E2-E4, the loss-weight ablation, PINN limitations) was rebuilt from scratch so the PINN actually works. Modules 1-3 were not changed. Module 5's baselines were copied unchanged; only the PINN rows of E5 and the E5 interpretation changed (tracker T39: "Run M4's PINN on real data").
The earlier audit and re-run of the old code was recorded in `PINN_RERUN_CHANGELOG.md` (removed in the clean-up; still in git history).

## 1. Results

| Experiment | Old PINN | New PINN |
|---|---|---|
| **E2** forward solve, rel. L2 error vs DOP853 (3 seeds) | 4%-82% | **0.035%-0.094%** (worst seed 0.15%) |
| **E3** β(t) from noisy cases + deaths only, rel. L2 (5 seeds) | 42% (flat line, failed; with all 5 compartments observed) | **10.8% (1% noise), 12.0% (5%)** |
| **E3** death rate μ | not estimated | **4.4% / 5.9%** error |
| **E4** hidden s(t), e(t), i(t) | s 2.8%, e 30% (with true β given) | **s 0.3-0.5%, e and i 5.5-6.8%** (β, μ unknown) |
| **E5** held-out days 90-119, case rel. error (all methods fitted to and scored on real report dates) | 13.4% | 11.0% (piecewise-β **10.1%**, constant-β 60.8%) |
| **E5** deaths rel. error | 1.5% | 9.5% (piecewise-β **1.9%**) |
| **E7** (new) 8 rolling 4-week forecasts, WHO 2014-15, cases MAE | - | **100.7** (piecewise-β 108.1, constant-β 120.9) |
| **E7** deaths MAE | - | 70.5 (piecewise-β **42.0**, constant-β 50.7) |

E6 (ablation, replaces the old figure that no code produced): without L-BFGS the hidden-state error is 96%; physics weight 10 instead of 1 roughly triples the errors; Fourier features cut the β error by about 25%; absolute residuals make the forward solve fail (75-100% error).

Summary: the PINN is accurate on synthetic problems. On real data it forecasts **cases** about as well as the best classical baseline (slightly behind on the single E5 window, ahead over the 8 E7 windows) and far better than constant-β, but forecasts **deaths** worse.

**E5 protocol change (27 Oct 2026).** Module 5's daily series is linear interpolation between WHO reports (36 real report dates in 120 days). Originally the baselines were fitted to the interpolated series and all methods were scored on it; under that protocol the PINN had 9.4% case error against 9.8%. E5 now fits every method to the 27 real reports in days 0-89 and scores on the 9 real reports in days 90-119. Module 5's baseline models (equations, starting values, loss, solver) are unchanged; only their input changed. The interpolated-protocol numbers are kept in the notebook and report as a reference.

## 1b. Final report from the Colab run (27 Oct 2026)

`report/main.tex` now quotes the outputs of `Final MM notebook.ipynb` as run in Google Colab, not the local run (the table above is the local run). Every figure in the report is the Colab output image, saved in `report/figures/`. The E6 figure was dropped.

**Clean-up (27 Oct 2026).** Removed `scratch/` (one-off helper scripts), `figs/` (generated figures; the notebooks and `pinn_figures.py` recreate it when run), `update_project_tracker.py` (one-off), `PINN_RERUN_CHANGELOG.md` and the old compiled report PDF. All remain in git history. `notebook.ipynb` is kept as the earlier version for reference; `Final MM notebook.ipynb` is the final one. No code was changed, and nothing was run locally for this.

| Colab output | Value |
|---|---|
| E1 slopes (Euler, RK4) | 1.001, 3.736 |
| E2 worst compartment, any seed | 2.0e-3 |
| E3 β rel. L2 (1%, 5% noise) | 11.0%, 11.6% |
| E3 μ rel. error | 4.6%, 5.4% |
| E5 PINN cases / deaths rel. error | 11.3% / 9.6% (seeds 5.6-21.8%) |
| E7 PINN cases MAE / deaths MAE | 96.8 / 69.7 (constant-β 120.9 / 50.7) |

**Open values (left blank in the report).** The piecewise-β baseline gives different results in Colab and locally: in E5, cases 17.4% vs 10.1% and deaths 6.9% vs 1.9%; in E7, cases MAE 112.9 vs 108.1. In Colab it does not reproduce Module 5's published MAE of 136.94 either (270.08). The probable cause is the single-start `least_squares` fit on a β that jumps between segments. Every piecewise-β number, and every comparison with it, is left blank (`\blank`) until the fit is made reliable. The E2 rerun with absolute residuals (E6) is also blank: its output table did not display in the Colab run.

**Removed from the report:** the E5 "interpolated protocol" comparison, the E6 "absolute residuals" row of the inverse ablation (no effect, 0.047 vs 0.042), and the E6 figure, which still plots that row.

**Who presents what:** E6 and E7 moved from Module 4 to Module 5 (text only: report author block, objectives and AI acknowledgement, and the notebook headings and overview table). Module 4 keeps the PINN design and E2-E4.

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
- `notebook.ipynb`: Module 4 cells replaced (design, algorithm, E2, E3/E4); the E5 cell now runs the experiment instead of showing a typed-in table; new E6 and E7 cells; the Module 4 row of the overview table, the E4/E5/E6 bullets of the interpretation cell and the AI disclosure updated.
  - **Self-contained for Google Colab (27 Oct 2026):** the notebook imports no project `.py` file. The PINN, data, experiment and figure code is copied verbatim from `seird_pinn.py`, `data_pipeline_who.py`, `pinn_experiments.py` and `pinn_figures.py` into four code cells. The data are downloaded from the original public URLs (cmrivers/ebola on GitHub; WHO on HDX, checksum-verified), and Module 5's daily series is rebuilt with Module 5's own processing. Experiments run in the notebook one after another and save to `results/`; `QUICK_RUN = True` gives a shorter version (fewer seeds and iterations, slightly different numbers), `FORCE_RERUN = True` retrains even when results exist. Estimated time in Colab: 1.5-2 hours for the full runs (not measured).
  - **Diagrams:** 11 explanatory diagrams embedded as images in markdown cells (overview, Module 1, Module 3, Module 4, Module 5; none in Module 2).
  - All code cells parse; the notebook has not been executed end to end in Colab.
- `report/main.tex`: abstract (PINN sentences), Sections "PINN Methodology", "Synthetic Experiments", the PINN parts of "Real-World Outbreak Application" (table, interpretation, new E7 subsection), "Ablation Studies (E6)", Limitations items 3-6, and a Module 4 line in the AI acknowledgement. Labels, citations, environments and figure paths were checked; **the PDF was not recompiled** (no LaTeX installation here).
- `report/refs.bib`: added the WHO HDX dataset, Tancik et al. 2020, Rahaman et al. 2019, Kingma and Ba 2015, Liu and Nocedal 1989.
- `data/README.md`: added the WHO dataset section and the verification of the Module 5 file.

## 3. Data

- **Module 5's file** (`data/country_timeseries_raw.csv`) is the content of `cmrivers/ebola` `country_timeseries.csv` (commit `2c696c79c72c49ab7a178bf4f4e391613aa42f18`), compiled from WHO situation reports. Its Sierra Leone values match WHO Disease Outbreak News of 31 Jul 2014 (533 cases, 233 deaths) and 22 Aug 2014 (910, 392). Only 36 of its 120 days are usable real report dates (one June report whose deaths WHO later revised down is dropped); in E5 every method is now fitted to and scored on only those.
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
