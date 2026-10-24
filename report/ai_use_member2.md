# AI use record: Member 2 (Mathematical Analysis)

The course rules ask us to say honestly where AI tools were used. This file covers Module 2 only.

**Tool:** Claude (Anthropic), used as a coding and writing assistant on 24 October 2026.

## What the AI did

| Part | What was done | How it was checked |
|---|---|---|
| R0 by the next-generation matrix | Re-derived F, V, FV⁻¹ and R0 = β/(γ+μ). Changed the new-infection matrix to carry s* so the same working gives the threshold R0·s* for any disease-free state. | The value R0 = 2.0004 is printed by the notebook from the shared parameters. |
| Equilibria and stability | Derived the Jacobian at (s*, 0, 0, r*, d*) and its characteristic equation. Corrected the old wording: these states are stable but not *asymptotically* stable, because three eigenvalues are zero. | The notebook computes the eigenvalues numerically for four values of s* (report Table 2). |
| Early growth rate (new) | Added λ₊, the positive eigenvalue, as the early growth rate, and the formula linking it back to R0. | A simulation from a tiny seed grows at 0.062732 per day against the predicted 0.062734 (Figure 6). |
| Final-size relation | Derived the relation with r0 and d0 kept in, and added the final-deaths formula. Found that the old check stopped the ODE at day 250, before slow epidemics end, which caused its errors. The check now runs until e + i < 10⁻¹³. | Formula and ODE agree to about 10⁻¹² for R0 from 0.8 to 3.5 (Table 3, Figure 7). |
| Sensitivity | Replaced the old sensitivity code with real Latin hypercube sampling (n = 1000, seed 42) and partial rank correlation, with p-values. Wrote the local sensitivity indices of R0. | σ has no effect on final size, which the formula predicts. The PRCC finds the same (≈ 0, not significant), a built-in check. |
| Identifiability (new) | Wrote a short paragraph on why σ, γ and μ cannot all be learned from cases and deaths. | Follows from the analysis above. |
| Text | Drafted Section 5 of the report, the notebook explanations and slides 4–5 in plain language. | |

## What it did not do

- It did not invent any number. Every value in the report and slides comes from running `notebook.ipynb` (ml_env kernel).
- It did not change other members' sections, apart from adding the Module 2 line to the AI acknowledgement.
