# AI use record: Member 2 (Mathematical Analysis)

**Tool:** Claude (Anthropic), 24 October 2026.

I worked through the analysis myself: R0 by the next-generation matrix, the equilibria and their stability, the final-size relation and the sensitivity analysis. I used Claude in three ways: to check my derivations, to help write and fix the analysis code, and to help me put my explanations into clear words for the report, notebook and slides.

## How I used it

| Part | My work | Where Claude helped | How it was checked |
|---|---|---|---|
| R0 by the next-generation matrix | Set up F and V and worked out R0 = β/(γ+μ). | Checked the derivation and suggested keeping s* in F, so the same working gives the threshold R0·s* for any disease-free state. | The notebook prints R0 = 2.0004 from the shared parameters. |
| Equilibria and stability | Found the disease-free equilibria and the Jacobian's eigenvalues. | Pointed out that the old wording was wrong: these states are stable but not *asymptotically* stable, because three eigenvalues are zero. | The notebook computes the eigenvalues for four values of s* (report Table 2). |
| Early growth rate | Linked the positive eigenvalue λ₊ to the early growth of the outbreak. | Helped write the simulation that checks it. | Simulated growth 0.062732 per day against the predicted 0.062734 (Figure 6). |
| Final-size relation | Derived the relation with r0 and d0 kept in, and the final-deaths formula. | Found why the old check gave errors: it stopped the ODE at day 250, before slow epidemics end. Helped rewrite the check to run until e + i < 10⁻¹³. | Formula and ODE agree to about 10⁻¹² for R0 from 0.8 to 3.5 (Table 3, Figure 7). |
| Sensitivity | Chose the outputs and the parameter ranges around our shared values; wrote the local indices of R0. | Helped write the Latin hypercube + PRCC code (n = 1000, seed 42) with p-values. | σ has no effect on final size, as the formula predicts; the PRCC also gives ≈ 0 (not significant). |
| Identifiability | Explained why σ, γ and μ cannot all be learned from cases and deaths. | Helped with the wording. | Follows from the analysis above. |
| Writing | Section 5 of the report, the notebook text, slides 4–5. | Helped make the wording plain and clear. | |

## What I checked

- Every number in the report and slides comes from running `notebook.ipynb` (ml_env kernel). None was typed in by hand or by the AI.
- I read and ran all the code, and I can explain each step of every derivation.
- Other members' sections were not changed, apart from adding the Module 2 line to the AI acknowledgement.
