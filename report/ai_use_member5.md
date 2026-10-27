# AI use record: Member 5 (Real-world application, baselines, rolling forecasts, ablation and limitations: E5-E7)

**Tool:** Claude (Anthropic), used through Claude Code on 26-27 October 2026.

Member 5 presents the real-data experiments E5 and E7, the ablation E6 and the limitations. The code for these experiments was written by Claude as part of the Module 4 rebuild (see `report/ai_use_member4.md`), and the presentation of E6 and E7 moved from Member 4 to Member 5 on 27 October 2026 without any change to the code. This record says what Claude did for these parts.

## What Claude did

| Part | What Claude did | How it was checked |
|---|---|---|
| Data | Found the WHO 2014-16 Sierra Leone situation reports on the Humanitarian Data Exchange and wrote the code that downloads them, checks the file's SHA-256 and drops the 6 death reports later revised downwards. | The notebook prints 238 report dates, 29 Aug 2014 to 6 Nov 2015, and 6 dropped death reports. |
| E5 design | Set the window (29 Aug 2014 - 25 Apr 2015), the 75/25 chronological split (74 training and 32 test reports) and the metrics (MAE, RMSE, relative error). | The PINN settings were fixed earlier, on validation forecasts ending on 16 Jan 2015. |
| Baselines | Wrote the constant-β and piecewise-β least-squares fits, which also estimate μ and the initial conditions, and made the piecewise fit integrate one segment at a time and start from several points so that it gives the same answer on every machine. | Same code for E5 and E7; results from the Colab run. |
| E6 and E7 | Wrote the ablation (one design choice changed at a time on the E3 problem) and the rolling 4-week forecasts at 8 origins. | Numbers come from the notebook's outputs. |
| Figures | Wrote the E5, E6 and E7 plotting code. | Figures are the images saved by the Colab run. |
| Writing | Drafted the Module 5 text in the notebook and the report (data provenance, split protocol, interpretation, limitations). On 27 October 2026 it filled in the E5, E6 and E7 numbers from the outputs of the final Colab run, quoting them as printed, without changing any result. | Every number can be found in the notebook's outputs. |

## To be completed by Member 5 before submission or presentation

The project instructions require every member to explain every equation, method and line of code they present, and to acknowledge AI use honestly. Please add an accurate account of your own part, for example:

- which parts you reviewed, re-ran or checked yourself (the data cleaning, the baselines, the E5 split, the E7 origins);
- whether you agree with the interpretation of the results, and anything you changed;
- what you will present, and that you can explain the baselines, the metrics and the forecasting protocol.

Do not claim work you did not do: dishonest acknowledgement can cost the whole group 10 marks.
