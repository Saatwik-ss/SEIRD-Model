# AI use record: Member 4 (PINN design and synthetic validation, E2-E4)

**Tool:** Claude (Anthropic), used through Claude Code on 26-27 October 2026.

The Module 4 work was done in Claude Code sessions: we described what we needed and made the decisions, and Claude designed, wrote and ran the code and drafted the text. The first version of the PINN, made earlier, did not work: the forward solve had errors of 4-82%, β(t) could not be recovered, and several reported numbers had been typed into the code rather than computed. The PINN was rebuilt from scratch on 26 October 2026.

**Who presents what (changed 27 October 2026, text only).** Member 4 presents the PINN design and the synthetic experiments E2-E4. The ablation (E6) and the real-data experiments (E5, E7) are presented by Member 5 (see `report/ai_use_member5.md`). The code was not changed when the presentation was split.

## How I used it

| Part | What we decided | Where Claude helped | How it was checked |
|---|---|---|---|
| Diagnosis of the old PINN | Asked for an audit of the old code before rebuilding it. | Re-ran the old code and found typed-in metrics, a forecast scored on a different quantity from the one it was trained on, a one-day offset, and the failure of E2 and E3. | Re-runs recorded in `PINN_RERUN_CHANGELOG.md` (removed; in git history). |
| PINN design (`seird_pinn.py`) | Kept the SEIRD model, σ and γ fixed from the clinical literature, and β(t) and μ learned. | Chose the observable form (C = I + R + D), exact conservation, hard initial conditions, relative ODE residuals, the CFR bound (0.2-0.8), Fourier features, and Adam then L-BFGS; found and fixed several of its own mistakes (see `M4_PINN_CHANGELOG.md`). | Each design choice is tested in the ablation (E6). |
| E2: forward solve | Asked for the PINN to be checked against an accurate ODE solver. | Wrote and ran the forward problem against DOP853 (3 seeds). | Relative L2 error 3.7 × 10⁻⁴ to 1.1 × 10⁻³ per compartment; S + E + I + R + D = N to 2 × 10⁻¹⁶. |
| E3: β(t) and μ from noisy data | Asked for a test where β changes over time and only cases and deaths are seen. | Wrote the step-β problem with 1% and 5% noise (5 seeds), tuning settings on a separate seed (seed 1), never on the reported seeds. | β relative L2 error 0.110 and 0.116; μ estimated as 0.1050 ± 0.0008 (true 0.1). |
| E4: hidden states | Asked whether the hidden compartments can be recovered. | Compared the PINN's s(t), e(t) and i(t) with the truth in the E3 runs. | s(t) within 0.5%, e(t) and i(t) within 6.5%. |
| Notebook and figures (`pinn_experiments.py`, `pinn_figures.py`) | Asked for everything to run in one notebook in Google Colab. | Copied the code into the final notebook and wrote the figure code. | The final notebook was run in full in Colab; all numbers come from its outputs, none were typed in. |
| Writing | Asked for the Module 4 sections of the report and notebook. | Drafted the PINN methodology and the E2-E4 sections and this record. | Every number was checked against the Colab outputs. |

## What I checked

- Every Module 4 number in the report and notebook comes from the final notebook (`Final_MM_notebook.ipynb`) run in full in Google Colab. None was typed in by hand or by the AI.
- The synthetic settings were chosen on seed 1 and reported on seeds 42-46, so the reported results were not tuned on.
- Other members' sections were not changed by this work, apart from the Module 4 and Module 5 lines in the AI acknowledgement.
