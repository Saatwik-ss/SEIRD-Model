# AI use record: Member 4 (PINN design and validation)

**Tool:** Claude (Anthropic), used through Claude Code on 26 October 2026.

The Module 4 work (the PINN and Experiments E2, E3, E4, E6 and E7, plus the PINN rows of E5) was rebuilt on 26 October 2026 with Claude doing most of the design, coding, running and first-draft writing. The first version of the PINN, made earlier, did not work: the forward solve had errors of 4-82%, β(t) could not be recovered, and several reported numbers had been typed into the code rather than computed. This record says what Claude did, so the acknowledgement in the report is accurate.

**Who presents what (changed 27 October 2026, text only).** Member 4 presents the PINN design and the synthetic experiments E2-E4. The ablation (E6) and the rolling forecasts (E7) moved to Member 5, alongside E5. The code was not changed and is the same code described below, which Claude wrote. Member 5 should state in their own AI record that they present results produced with this code.

## What Claude did

| Part | What Claude did | How it was checked |
|---|---|---|
| Diagnosis of the old PINN | Re-ran the old code, found typed-in metrics, an E5 forecast that scored a different quantity from the one it was trained on, a one-day offset, and the failure of E2 and E3. | Re-runs recorded in `PINN_RERUN_CHANGELOG.md` (now removed; in git history). |
| Data | Found and verified the WHO 2014-16 Sierra Leone data on HDX and wrote `data_pipeline_who.py`; at the group's request this is now the only dataset (E5 and E7). | SHA-256 of the raw file recorded; downward-revised reports flagged and excluded. |
| PINN design (`seird_pinn.py`) | Chose the observable form (C = I + R + D), exact conservation, hard initial conditions, relative residuals, the CFR bound, Fourier features, and Adam then L-BFGS. Found and fixed several of its own mistakes along the way (see `M4_PINN_CHANGELOG.md`). | Forward error against DOP853 (E2); ablation of each design choice (E6). |
| Experiments (`pinn_experiments.py`, `pinn_figures.py`) | Wrote and ran E2-E7; chose settings on a tuning seed (synthetic) and on validation windows (real data) before running the test cases. | All numbers come from `results/m4_*.json`, written by the code; none were typed in. |
| E5 on WHO data and the baselines | At the group's request, moved E5 to the WHO data (fit on 29 Aug 2014 - 24 Feb 2015, test on 25 Feb - 25 Apr 2015) with constant-β and 3-segment piecewise-β least squares, and made the piecewise fit start from several points so that it gives the same answer on every machine. | Only in `Final_MM_notebook.ipynb`; the numbers come from its Colab run. |
| Writing | Drafted the Module 4 sections of the report and notebook, the E5 interpretation, and this record. | |

## To be completed by Member 4 before submission or presentation

The project instructions require every member to explain every equation, method and line of code they present, and to acknowledge AI use honestly. Please replace this section with an accurate account of your own part, for example:

- which parts you reviewed, re-derived or re-ran yourself;
- which design decisions you understood and agree with (and any you changed);
- what you will present, and that you can explain each equation in Section "Physics-Informed Neural Network (PINN) Methodology" of the report.

Do not claim work you did not do: dishonest acknowledgement can cost the whole group 10 marks.
