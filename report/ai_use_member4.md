# AI use record: Member 4 (PINN design and validation)

**Tool:** Claude (Anthropic), used through Claude Code on 26 October 2026.

The Module 4 work (the PINN and Experiments E2, E3, E4, E6 and E7, plus the PINN rows of E5) was rebuilt on 26 October 2026 with Claude doing most of the design, coding, running and first-draft writing. The first version of the PINN, made earlier, did not work: the forward solve had errors of 4-82%, β(t) could not be recovered, and several reported numbers had been typed into the code rather than computed. This record says what Claude did, so the acknowledgement in the report is accurate.

## What Claude did

| Part | What Claude did | How it was checked |
|---|---|---|
| Diagnosis of the old PINN | Re-ran the old code, found typed-in metrics, an E5 forecast that scored a different quantity from the one it was trained on, a one-day offset, and the failure of E2 and E3. | Re-runs recorded in `PINN_RERUN_CHANGELOG.md` (kept as a record of the old code). |
| Data | Found and verified the WHO 2014-16 Sierra Leone data on HDX; wrote `data_pipeline_who.py`. Verified Module 5's cmrivers data against the upstream file and two WHO bulletins. | SHA-256 of the raw file recorded; spot checks against WHO Disease Outbreak News of 31 Jul and 22 Aug 2014. |
| PINN design (`seird_pinn.py`) | Chose the observable form (C = I + R + D), exact conservation, hard initial conditions, relative residuals, the CFR bound, Fourier features, and Adam then L-BFGS. Found and fixed several of its own mistakes along the way (see `M4_PINN_CHANGELOG.md`). | Forward error against DOP853 (E2); ablation of each design choice (E6). |
| Experiments (`pinn_experiments.py`, `pinn_figures.py`) | Wrote and ran E2-E7; chose settings on a tuning seed (synthetic) and on validation windows (real data) before running the test cases. | All numbers come from `results/m4_*.json`, written by the code; none were typed in. |
| Module 5 baselines in E5 | Copied Module 5's constant-β and piecewise-β baselines without changing them. | Reproduces Module 5's published errors exactly (MAE 772.45 and 136.94 cases). |
| Writing | Drafted the Module 4 sections of the report and notebook, the E5 interpretation, and this record. | |

## To be completed by Member 4 before submission or presentation

The project instructions require every member to explain every equation, method and line of code they present, and to acknowledge AI use honestly. Please replace this section with an accurate account of your own part, for example:

- which parts you reviewed, re-derived or re-ran yourself;
- which design decisions you understood and agree with (and any you changed);
- what you will present, and that you can explain each equation in Section "Physics-Informed Neural Network (PINN) Methodology" of the report.

Do not claim work you did not do: dishonest acknowledgement can cost the whole group 10 marks.
