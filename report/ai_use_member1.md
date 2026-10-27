# AI use record: Member 1 (Biological Motivation, Formulation, Published Models and Phase-Space Analysis)

**Tool:** Google Antigravity (Google's agent-based coding environment, running Gemini models), accessed through Jio.

Module 1 was produced entirely with Antigravity. Antigravity wrote the text, the derivations and the code of this module; Member 1 directed it, ran the cells and put the output into the notebook and the report. This record lists what Antigravity produced, so that the acknowledgement in the report is accurate.

## What Antigravity did

| Part | What Antigravity produced | Where it appears |
|---|---|---|
| Biological motivation | The account of Ebola transmission (clinical isolation, safe burials, behaviour change, outside medical help) and why β changes over time; why surveillance sees only cases and deaths, so E and S are hidden. | Report, Introduction and Module 1; notebook, Module 1 text |
| SEIRD formulation | The five equations in fractions s, e, i, r, d; the conservation law s + e + i + r + d = 1; the modelling assumptions (closed population, homogeneous mixing, lasting immunity, σ and γ fixed from clinical data). | Report, model formulation; notebook, Module 1 text |
| Parameters | The Sierra Leone parameter set: N = 7,092,113, σ = 1/9.4, γ = 1/7, μ = 0.1, β = 0.4858, giving CFR = 41.2% and R0 = 2.000, with the literature sources. | Notebook, "Module 1 setup" cell |
| Published models | The summary and critique of Legrand et al. (2007), Rivers et al. (2014) and Althaus (2014), and the argument that a constant β cannot follow a real outbreak. | Report, published-models section and table |
| Published-model simulations | The code that simulates a static model, Althaus's exponential decay of β, Rivers's step change and an adaptive β, and prints the peak, peak day, attack rate, total deaths and the day R_t falls below 1. | Notebook, "Simulation and quantitative comparison of published transmission models" cell; report figure and table |
| Phase-plane analysis | The derivation of di/ds, the trajectory i(s), the peak i_max at s* = 1/R0, and the code that checks the peak formula against the ODE solution for R0 from 1.2 to 3.5. | Report, phase-space section; notebook, "Phase-space trajectory and peak infection" cell |
| Thresholds | The herd-immunity threshold p_c = 1 − 1/R0, the contact reduction δ_c = 1 − 1/R0, and the code for the R0(β, γ + μ) contour map. | Report, thresholds section; notebook, "Critical intervention thresholds" cell |
| Constant-β demonstration | The code showing how a constant-β model fails on an outbreak whose transmission changes. | Notebook, "Simulation demo 1" cell |
| Compartment diagram | The code for the SEIRD flow diagram and conservation check figure. | Notebook, "Module 1 visualization" cell |
| Writing | The Module 1 text in the notebook and the report, and its LaTeX. | Report and notebook |

## How the output was checked

- The Module 1 numbers in the report are the outputs of the final notebook run in Google Colab (for example, the phase-space check agrees with the ODE to between 2 × 10⁻⁷ and 1.3 × 10⁻⁵ relative error).
- The parameter values are shared with Modules 2–5, whose checks reproduce them (for example, Module 2 recovers R0 = 2.0004 from the early growth rate).

## To be completed by Member 1 before submission or presentation

The project instructions require every member to explain every equation, method and line of code they present, and to acknowledge AI use honestly. Please add an accurate account of your own part, for example:

- which derivations you worked through yourself (i(s), i_max, p_c, δ_c), and which you checked against the code;
- which published papers you read yourself, and whether the summaries match them;
- what you changed in Antigravity's output, if anything;
- what you will present, and that you can explain each equation and code cell of Module 1.

Do not claim work you did not do: dishonest acknowledgement can cost the whole group 10 marks.
