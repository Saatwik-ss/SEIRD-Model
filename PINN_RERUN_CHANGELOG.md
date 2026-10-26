# PINN Re-run Change Log (26 October 2026)

> **Note (26 Oct 2026, later):** the code described below (`run_pinn_pipeline.py`, `pinn_complete.py`, `generate_metric_evaluations.py`, `pinn_model.py`) has since been replaced by a rebuilt Module 4 PINN. See `M4_PINN_CHANGELOG.md`.

This file records every change made to the repository on 26 October 2026, why it was made, how the dataset was verified, and what the re-run produced. The PINN model code (`pinn_model.py`) and the training loops (losses, weights, epochs, learning rates, network sizes) were **not** changed.

---

## 1. Dataset: source and verification

The dataset was **not** replaced. We checked whether a better real dataset exists for the same window, and verified that the existing one is real.

### Exact source

| Item | Value |
|---|---|
| Upstream file | `country_timeseries.csv` in Caitlin Rivers' repository **https://github.com/cmrivers/ebola** |
| Raw URL | https://raw.githubusercontent.com/cmrivers/ebola/master/country_timeseries.csv |
| Upstream commit (last change to the file) | `2c696c79c72c49ab7a178bf4f4e391613aa42f18` (9 Jan 2015, message "through jan 5") |
| Original data provider | World Health Organization: Ebola situation reports and Disease Outbreak News (DON) bulletins |
| Columns used | `Date`, `Cases_SierraLeone`, `Deaths_SierraLeone` (cumulative confirmed + probable + suspected) |
| Window used | 25 May 2014 to 21 Sep 2014 (120 days), Sierra Leone, N = 7,092,113 |

### Checks performed

1. **Local raw file = upstream file.** `data/country_timeseries_raw.csv` was compared with a fresh download of the raw URL. The content is identical. The only difference is Windows line endings added by git (`core.autocrlf`). SHA-256 of both, with line endings normalised: `fda0a9972ec3bc827c7ccdae688acb85dc7c36581efee261c4bb439605d96be4`.
2. **Processed file is reproducible.** Re-running `data_pipeline.process_sierra_leone_data()` on the raw file gives a table identical to `data/outbreak_data.csv` (`DataFrame.equals` → `True`).
3. **Spot checks against WHO's own bulletins:**

   | WHO bulletin | Figures as of | WHO: Sierra Leone | Dataset row | Match |
   |---|---|---|---|---|
   | [DON, 31 Jul 2014](https://www.who.int/emergencies/disease-outbreak-news/item/2014_07_31_ebola-en) | 27 Jul 2014 | 533 cases, 233 deaths | 533, 233 | Yes |
   | [DON, 22 Aug 2014](https://www.who.int/emergencies/disease-outbreak-news/item/2014_08_22_ebola-en) | 20 Aug 2014 | 910 cases, 392 deaths | 910, 392 | Yes |

### Alternatives considered and rejected

| Source | Why not used |
|---|---|
| WHO data on HDX, [`ebola-cases-2014`](https://data.humdata.org/dataset/ebola-cases-2014) (`ebola_data_db_format.csv`) | Sierra Leone series starts on 29 Aug 2014, so it does not cover the 25 May – 21 Sep window. Switching would mean changing the window and the hard-coded initial conditions in the code. |
| WHO patient database, `cmrivers/ebola/who_data/sierraleone.csv` | Weekly new cases only; no deaths, which the model needs. |
| Sierra Leone MoHS daily sitreps, `cmrivers/ebola/sl_data/` | Start on 12 Aug 2014; do not cover the window. |

### Caveat to state in the report

WHO did not report every day. In the 120-day window there are **37 real report dates** (about 28 in the training window and 9 in the test window). `data_pipeline.py` fills the other days by straight-line interpolation between reports (`data_pipeline.py:62-64`). The "120 daily observations" are therefore 37 real values plus interpolated ones. The raw data also contains one decrease in reported deaths (49 on 17 Jun, then 28 on 18 Jun), which the pipeline flattens by never letting the cumulative total go down.

---

## 2. Code changes

### 2.1 E5 forecast: case definition (bug fix)

**Files:** `run_pinn_pipeline.py`, `generate_metric_evaluations.py`, `pinn_complete.py` (the same E5 code exists in all three).

**Problem:** the PINN is trained on cases = I + R + D, and the two baselines use 1 − s − e (also I + R + D). But the PINN's test-window forecast counted cases as **E** + I + R + D, so the held-out score compared a different quantity from the one the model was trained on. E is about 190 people near day 90, which is larger than the old reported MAE of 98.

**Fix:** test-window cases are now `(i + r + d) * N`.

### 2.2 E5 forecast: one-day offset (bug fix)

**Problem:** the forecast started from the PINN's day-89 state but treated it as day 90 (`t_test - t_test[0]`), so every forecast point was one day behind.

**Fix:** the ODE is integrated from day 89 over `[89, 90, …, 119]`, and the day-89 point is dropped.

### 2.3 Metrics script: E1–E4 now computed, not typed in

**File:** `generate_metric_evaluations.py`

**Problem:** `get_synthetic_benchmark_stats()` returned fixed numbers for E1–E4 (including a "standard deviation over seeds" for E3 that was never computed). These fed `fig_metrics_synthetic_pinn` and `fig_metrics_summary_dashboard`. The dashboard also had fixed bar values, fixed table rows, fixed axis limits, and labels declaring which model was "Best".

**Fix:**
- New `run_e1_convergence()` runs forward Euler and RK4 against DOP853 with the same setup as the notebook's E1 cells: β = 0.4858, y₀ = (0.99, 0.005, 0.005, 0, 0), T = 60, h ∈ {1, …, 1/32}, rtol 1e-12, atol 1e-14.
- `get_synthetic_benchmark_stats()` now calls `run_experiment_e2/e3/e4` from `run_pinn_pipeline.py`. E3 runs over seeds 42–46 and reports mean ± SD.
- All plot values, axis limits and dashboard table cells are taken from the computed results. The "Best Cases / Best Deaths" labels were removed.
- `main()` now also writes `figs/evaluation_metrics.csv` (previously only written by `pinn_complete.py`) and `figs/synthetic_benchmark_metrics.txt` (new: E1–E4 tables).

### 2.4 `run_pinn_pipeline.py`: E2–E4 return numbers (no change to training)

- `run_experiment_e2`: also returns the Pearson r for each compartment.
- `run_experiment_e3`: new argument `seeds=(42,)` (the default reproduces the old single-seed behaviour). Metrics are returned as numbers instead of formatted strings, with R² added. The figure is drawn from the first seed only.
- `run_experiment_e4`: also returns relative errors, Pearson r and maximum absolute error.

### 2.5 `report/main.tex`: repaired paragraph

The "Critical Scientific Interpretation" paragraph had lost its `$` signs and some digits, and contained literal `\n` characters, so it would not compile. These were restored. The values 0.308 and 0.262 for β were reconstructed, not taken from a run; the re-run gives different values (see §4).

### 2.6 Not changed

`pinn_model.py`, `data_pipeline.py`, both data files, all loss weights, epochs, learning rates, network sizes and seeds of the existing experiments, and `notebook.ipynb`.

---

## 3. How the re-run was done

**Environment** (temporary virtual environment outside the project, Python 3.11.9): torch 2.14.1+cpu, numpy 2.4.6, pandas 3.1.0, scipy 1.17.1, matplotlib 3.11.2, scikit-learn 1.9.1.

**Commands, from the project folder:**

```bash
python generate_metric_evaluations.py
```

```bash
python run_pinn_pipeline.py --task e5 --epochs 2000 --lr 2e-3 --colloc 250
```

The second command redraws `fig_e5_real_forecast` with the same hyper-parameters the metrics script uses for E5. Its default CLI values (`--lr 1.5e-3 --colloc 300`) would silently train a slightly different E5 model. Both runs gave **identical** E5 numbers, so E5 is reproducible on CPU with seed 42.

**Run times:** metrics script 6 min 35 s; E5 figure 39 s.

**Files regenerated in `figs/`:** `evaluation_metrics.csv`, `synthetic_benchmark_metrics.txt` (new), and the `.eps`/`.png` pairs for `fig_e2_forward_pinn`, `fig_e3_inverse_beta`, `fig_e4_hidden_states`, `fig_e5_real_forecast`, `fig_metrics_forecast_benchmark`, `fig_metrics_synthetic_pinn`, `fig_metrics_error_residuals` and `fig_metrics_summary_dashboard`.

---

## 4. Results: before and after

### E5: held-out forecast, days 90–119 (real data)

| Model | Target | MAE before | MAE after | Rel. error before | Rel. error after | R² after |
|---|---|---|---|---|---|---|
| Constant-β LS | Cases | 772.5 | 772.5 | 55.3% | 55.3% | −8.20 |
| Constant-β LS | Deaths | 276.1 | 276.1 | 54.2% | 54.2% | −29.64 |
| Piecewise-β LS | Cases | 136.9 | 136.9 | 9.8% | 9.8% | 0.592 |
| Piecewise-β LS | Deaths | 6.3 | 6.3 | 1.2% | 1.2% | 0.982 |
| **PINN** | **Cases** | 98.3 | **187.2** | 7.0% | **13.4%** | **0.364** |
| **PINN** | **Deaths** | 10.6 | **7.5** | 2.1% | **1.5%** | **0.975** |

The baselines are unchanged; only the PINN forecast code was fixed. **With the fixes, the piecewise-β baseline beats the PINN on both cases (9.8% vs 13.4%) and deaths (1.2% vs 1.5%).** The report's claim that the PINN gives the lowest case error no longer holds. Learned β(t) on the training window: 0.285 at day 0 → 0.274 at day 89 (R = β/(γ+μ) ≈ 1.17 → 1.13).

### E1: solver convergence

| | Report | Re-run |
|---|---|---|
| Euler order | 0.999 | 1.001 |
| RK4 order | 3.694 | 3.736 |

The RK4 error at h = 1/32 (1.5e-12) has reached the accuracy of the reference solver, which pulls the fitted slope below 4. Both orders are as expected.

### E2: forward PINN (constant β)

| Compartment | Rel. L2 (old typed-in value) | Rel. L2 (re-run) |
|---|---|---|
| s | 9.6e-4 | 0.044 |
| e | 3.3e-3 | **0.735** |
| i | 2.8e-3 | **0.817** |
| r | 1.2e-3 | **0.742** |
| d | 1.2e-3 | **0.712** |

With the existing settings (1,200 epochs), the forward PINN does **not** converge: final loss 1.5e-2, and the e and i curves are negatively correlated with the truth (Pearson r = −0.19 and −0.29).

### E3: inverse β(t) recovery (5 seeds)

| Profile | Noise | β RMSE, typed-in | β RMSE, re-run (mean ± SD) | Rel. L2, re-run | R², re-run |
|---|---|---|---|---|---|
| Step | 1% | 0.0142 | 0.136 ± 0.014 | 42.5% | −0.06 |
| Step | 5% | 0.0384 | 0.136 ± 0.014 | 42.5% | −0.06 |
| Sinusoid | 1% | 0.0168 | 0.118 ± 0.016 | 35.6% | −0.99 |
| Sinusoid | 5% | 0.0412 | 0.118 ± 0.016 | 35.6% | −0.99 |

**β(t) is not recovered.** The learned β is an almost flat line (about 0.245–0.271) for both the step and the sinusoid; the two curves in `fig_e3_inverse_beta` are the same. A separate check showed the learned β differs by only 5×10⁻⁷ between 1% and 5% noise, so the data has almost no influence on β. The likely cause: in this synthetic setup the epidemic is tiny (infectious fraction peaks at 0.24%), so the β·s·i term in the ODE loss carries almost no gradient, and β is set mainly by the initialisation and the smoothness penalty. R² below 0 means the result is worse than predicting the mean.

### E4: hidden-state recovery (β known)

| Latent state | RMSE, typed-in / report | RMSE, re-run | Rel. error, re-run |
|---|---|---|---|
| s(t) | 0.0088 | 0.0269 | 2.8% |
| e(t) | 0.0038 | 0.0157 | 30.3% |

The shape is tracked (Pearson r ≈ 0.99 for both), but the e(t) error is about 30% of its range. Note that β is given as the true constant in E4, so this test does not show recovery when β is also unknown.

---

## 5. What is now out of date (not changed in this pass)

- **`report/main.tex`:** the E5 table (§ Real-World Outbreak Application), the E3 and E4 numbers, the E1 RK4 order (3.694), the E3 β profiles (the report says 0.60→0.22 and 0.45 + 0.15 sin; the code uses 0.45→0.18 and 0.32 + 0.12 sin), the claim that E2 "closely tracks" DOP853, and the "Critical Scientific Interpretation" paragraph, which says the PINN beats piecewise-β.
- **`notebook.ipynb`:** the E3, E4 and E5 cells display typed-in tables instead of running the experiments. The E3 cell says "Across 10 Random Seeds", which was never run.
- **E6 (ablations):** there is still no code for `fig_e6_ablations`; the figure dates from the initial commit.
