# AI use record: Member 5 (Real-world application, baselines, rolling forecasts, ablation and limitations: E5-E7)

**Tool:** Claude (Anthropic), used through Claude Code on 26-27 October 2026.

The Module 5 work was done in Claude Code sessions: we described what we needed and made the decisions, and Claude wrote and ran the code and drafted the text. The real-data experiments (E5, E7) and the ablation (E6) use the PINN code from Module 4 (see `report/ai_use_member4.md`); their presentation moved from Member 4 to Member 5 on 27 October 2026 without any change to the code.

## How I used it

| Part | What we decided | Where Claude helped | How it was checked |
|---|---|---|---|
| Data | Use only real WHO surveillance data, with no interpolated or synthetic series; the earlier interpolated 120-day series was removed from the project. | Found the WHO 2014-16 Sierra Leone situation reports on the Humanitarian Data Exchange and wrote the code that downloads them, checks the file's SHA-256 and drops the 6 death reports later revised downwards. | The notebook prints 238 report dates (29 Aug 2014 - 6 Nov 2015) and 6 dropped death reports. |
| E5: held-out test | Move E5 to the WHO data with a longer window and a strict 75/25 chronological split. | Set the window (29 Aug 2014 - 25 Apr 2015), the split (74 training and 32 test reports) and the metrics (MAE, RMSE, relative error). | The PINN settings were fixed earlier, on validation forecasts ending on 16 Jan 2015, before the test window. |
| Baselines | Compare the PINN with classical least squares that has the same freedom (also estimating μ and the initial conditions). | Wrote the constant-β and piecewise-β fits. When the piecewise fit gave different answers in Colab and locally, made it integrate one segment at a time and start from several points, keeping the lowest cost. | The same fits are used in E5 and E7; the results come from the Colab run. |
| E7: rolling forecasts | Test the PINN across the whole epidemic, not on one window. | Wrote the rolling 4-week forecasts at 8 origins (16 Jan - 31 Jul 2015), each fitted to the previous 8 weeks. | Three earlier origins were used only to choose settings. |
| E6: ablation | Show which design choices of the PINN matter. | Wrote the ablation (one choice changed at a time on the E3 problem, and E2 repeated with absolute residuals). | Results from the Colab run. |
| Removed tests | Remove tests that add nothing (the R0 forest plot and section 4.2.3, the interpolated-protocol comparison). | Removed them from the notebook and the report. | |
| Writing and results | Report the results exactly as the notebook printed them, with no changes. | Drafted the Module 5 text (data provenance, split protocol, interpretation, limitations) and, on 27 October 2026, filled in the E5, E6 and E7 numbers from the outputs of the final Colab run as printed. | Every number can be found in the notebook's outputs. |

## What I checked

- Every Module 5 number in the report and notebook comes from the final notebook (`Final_MM_notebook.ipynb`) run in full in Google Colab, quoted as printed. None was typed in by hand or by the AI, and no result was changed.
- The results are reported as they came out, including where the PINN is not the best method: piecewise-β least squares has the lowest case error in E5 (0.818% against the PINN's 1.350%), and both baselines forecast deaths better than the PINN in E7.
- Other members' sections were not changed by this work, apart from the Module 5 line in the AI acknowledgement.
