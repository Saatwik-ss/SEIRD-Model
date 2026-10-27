# AI use record: Member 1 (Biological Motivation, Formulation, Published Models and Phase-Space Analysis)

**Tool:** Google Antigravity (Google's agent-based coding environment, running Gemini models), accessed through Jio.

Module 1 was produced entirely with Antigravity: I described what the module needed and Antigravity wrote the text, the derivations and the code. I ran the cells and put the output into the notebook and the report.

## How I used it

| Part | My work | Where Antigravity helped | How it was checked |
|---|---|---|---|
| Biological motivation | Chose the case study (2014-16 Ebola epidemic, Sierra Leone) and the question: can a PINN recover β(t) and the hidden compartments? | Wrote the account of Ebola transmission (clinical isolation, safe burials, behaviour change, outside medical help), why β changes over time, and why surveillance sees only cases and deaths. | |
| SEIRD formulation | Chose an SEIRD model with deaths as a separate compartment. | Wrote the five equations in fractions s, e, i, r, d, the conservation law s + e + i + r + d = 1 and the modelling assumptions. | Module 3's solvers keep the sum equal to 1 to 6.66 × 10⁻¹⁶. |
| Parameters | Asked for literature values for Sierra Leone. | Set N = 7,092,113, σ = 1/9.4, γ = 1/7, μ = 0.1 and β = 0.4858, giving CFR = 41.2% and R0 = 2.000, with sources. | The notebook prints these values; Module 2 recovers R0 = 2.0004 from the early growth rate. |
| Published models | Chose to compare three published Ebola models. | Summarised and critiqued Legrand et al. (2007), Rivers et al. (2014) and Althaus (2014), and argued that a constant β cannot follow a real outbreak. | |
| Published-model simulations | Asked for the models to be compared on the same parameters. | Wrote the code that simulates a static model, Althaus's decaying β, Rivers's step change and an adaptive β, and tabulates peak, peak day, attack rate, deaths and the day R_t falls below 1. | Output table and figure from the Colab run. |
| Phase-plane analysis | Asked for the peak of the epidemic to be derived. | Derived di/ds, the trajectory i(s) and the peak i_max at s* = 1/R0, and wrote the code that checks the peak formula against the ODE for R0 from 1.2 to 3.5. | Formula and ODE agree to between 2 × 10⁻⁷ and 1.3 × 10⁻⁵ relative error. |
| Thresholds | Asked for the intervention thresholds. | Derived p_c = 1 − 1/R0 and δ_c = 1 − 1/R0 and wrote the code for the R0(β, γ + μ) contour map. | Contour map from the Colab run. |
| Constant-β demonstration and diagram | Asked for a demonstration of why a constant β fails, and a diagram of the model. | Wrote the demonstration and the SEIRD flow diagram code. | Figures from the Colab run. |
| Writing | Asked for the Module 1 sections. | Wrote the Module 1 text in the notebook and the report, and its LaTeX. | |

## What I checked

- Every Module 1 number in the report comes from the final notebook (`Final_MM_notebook.ipynb`) run in full in Google Colab. None was typed in by hand or by the AI.
- The derivations are checked numerically in the notebook (the phase-space table), and the parameters are the shared values used and checked by Modules 2-5.
- Other members' sections were not changed by this work, apart from adding the Module 1 line to the AI acknowledgement.
