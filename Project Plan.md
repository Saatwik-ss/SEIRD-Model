# Project Plan: PINN for an SEIRD Epidemic Model with Time-Varying Transmission

**Title:** _Physics-Informed Neural Networks for Parameter Estimation and Hidden-State Recovery in an SEIRD Model with Time-Varying Transmission: Application to the 2014 West African Ebola Epidemic (Sierra Leone)_

---

## 1. Problem and Biological Motivation

Classical epidemiological calibration enforces an invariant transmission rate $\beta$, yet in actual outbreaks transmission evolves continuously due to clinical isolation, safe burial practices, community behavioral precautions, and international medical intervention. Surveillance reports capture only manifest clinical endpoints (confirmed symptomatic diagnoses, hospital admissions, and mortalities), leaving the latent incubation pool ($E$) and remaining susceptible host pool ($S$) unobserved.

The project investigates whether a **Physics-Informed Neural Network (PINN)** can:
1. Reconstruct latent unobserved epidemiological compartments from sparse clinical data.
2. Invert time-varying transmission functions $\beta(t)$ without pre-imposing structural assumptions.
3. Compare against classical regularized least-squares models on a strictly held-out real-world outbreak forecast window.

**Epidemic Case Study**: 2014--2016 West African Ebola Virus Disease (EVD) Epidemic in the Republic of Sierra Leone ($N = 7{,}092{,}113$).

---

## 2. Model Formulation & Governing Assumptions

Normalized compartmental state fractions ($s = S/N, e = E/N, i = I/N, r = R/N, d = D/N$):
```
ds/dt = -β(t) s i
de/dt =  β(t) s i - σ e
di/dt =  σ e - (γ + μ) i
dr/dt =  γ i
dd/dt =  μ i
```
Subject to the physical conservation invariant:
$$s(t) + e(t) + i(t) + r(t) + d(t) = 1 \quad \forall t \ge 0$$

**Epidemiological Parameters for Ebola Virus Disease:**
- Latency transition rate: $\sigma = 1/9.4 \approx 0.1064\text{ day}^{-1}$ (incubation period $\approx 9.4$ days, WHO Ebola Response Team, *NEJM* 2014).
- Recovery clearance rate: $\gamma = 1/7.0 \approx 0.1429\text{ day}^{-1}$ (infectious period among survivors $\approx 7.0$ days).
- Disease-induced mortality rate: $\mu \approx 0.1000\text{ day}^{-1}$ (reflecting high case-fatality ratio $\text{CFR} = \frac{\mu}{\gamma+\mu} \approx 41.2\%$).
- Basic reproduction number: $R_0 = \frac{\beta}{\gamma+\mu} \approx 2.00$ at baseline $\beta \approx 0.4858\text{ day}^{-1}$.

**Assumptions**:
1. Closed host population over the acute phase.
2. Homogeneous community mixing.
3. Post-recovery protective sterilizing immunity.
4. Clinically anchored $\sigma$ and $\gamma$ to preserve structural identifiability.

---

## 3. Mathematical Analysis & Module 1 Advanced Contributions (Own Analysis on Published Models)

To strictly satisfy the instructor's requirement:
> *"Published models may be used as a starting point, but each group must demonstrate its own mathematical analysis, implementation, interpretation, or meaningful extension."*

Module 1 leads substantive mathematical modeling:
1. **Comparative Synthesis of Published Ebola Models**:
   - Critical evaluation of 3 benchmark published models: Legrand et al. (2007) [9 compartments: community, hospital, funeral], Rivers et al. (2014) [hospital/community interventions in Sierra Leone], and Althaus (2014) [exponential decay $\beta(t)$].
   - Formal critique of why classical SEIR fails without dynamic $\beta(t)$ or explicit funeral compartments, justifying our PINN approach as a meaningful data-driven extension.
2. **Phase-Plane & Trajectory Analysis ($i$ vs $s$)**:
   - Exact derivation of the phase trajectory equation:
     $$\frac{di}{ds} = -1 + \frac{\gamma+\mu}{\beta s} = -1 + \frac{1}{R_0 s} \implies i(s) = i_0 + s_0 - s + \frac{1}{R_0}\ln\left(\frac{s}{s_0}\right)$$
   - Derivation of the exact maximum infectious peak fraction $i_{\max}$:
     $$i_{\max} = i_0 + s_0 - \frac{1}{R_0}\left(1 + \ln(R_0 s_0)\right) \quad \text{occurring at } s^* = \frac{1}{R_0}$$
3. **Thresholds & Critical Intervention Bounds**:
   - Derivation of the herd immunity / vaccination threshold $p_c = 1 - 1/R_0$.
   - Calculation of the minimum contact reduction factor $\delta_c = 1 - 1/R_0$ required to halt the outbreak.
   - Construction of a 2D parameter space contour analysis of $R_0(\beta, \gamma+\mu)$ mapping the subcritical vs supercritical domains.

---

## 4. PINN Methodology

- **Dual-Network Setup**:
  - State network $\mathcal{NN}_\theta(\tau) \to (\hat{s}, \hat{e}, \hat{i}, \hat{r}, \hat{d}) \in (0,1)^5$ with 3 hidden layers (48 units each, $\tanh$ activations).
  - Transmission network $\mathcal{NN}_\phi(\tau) \to \hat{\beta}(\tau)$ with $\text{Softplus}$ output to ensure $\hat{\beta}(\tau) > 0$.
- **Physics-Informed Loss**:
  $$\mathcal{L} = w_{\text{data}}\mathcal{L}_{\text{data}} + w_{\text{ode}}\mathcal{L}_{\text{ode}} + w_{\text{ic}}\mathcal{L}_{\text{ic}} + w_{\text{reg}}\mathcal{L}_{\text{smooth}}$$
  where ODE residuals are evaluated at random interior collocation points using PyTorch `autograd`.

---

## 5. Experiments

| # | Experiment | Focus | Quantitative Deliverable |
|---|---|---|---|
| **E1** | Custom Numerical Solvers | Vectorized Forward Euler & RK4 vs DOP853 benchmark | Empirical log-log convergence rates (orders $\approx 1$ and $4$) |
| **E2** | Forward PINN Integration | Synthetic SEIRD with known constant $\beta$ | Relative $L_2$ error across all 5 compartments |
| **E3** | Inverse PINN $\beta(t)$ Recovery | Step intervention & seasonal sinusoidal profiles under noise | Transmission RMSE and Relative $L_2$ error |
| **E4** | Latent Hidden-State Recovery | Inversion of $s(t)$ and $e(t)$ given only cases $C(t)$ and deaths $D(t)$ | Reconstruction RMSE of unobserved compartments |
| **E5** | Real-World Outbreak Application | 2014 Sierra Leone Ebola Outbreak (75% train / 25% test) | Out-of-sample MAE, RMSE, Relative Error vs classical least squares |
| **E6** | Ablations | Collocation density $N_f$ and physics weight $w_{\text{ode}}$ | Error trends and gradient stiffness analysis |

---

## 6. Real-World Dataset Acquisition & Provenance Protocol

**Notice**: As instructed, no synthetic datasets are used for the real-world validation task. Data must be acquired directly from verifiable public archives:
1. **Primary Public Repository**:
   - Caitlin Rivers' curated machine-readable Ebola archive on GitHub:  
     `https://github.com/cmrivers/ebola` (`country_timeseries.csv`).
2. **Authoritative Primary Source**:
   - World Health Organization (WHO) Ebola Situation Reports Archive:  
     `https://www.who.int/ebola/situation-reports/archive`
3. **Alternative Public Mirrors**:
   - Humanitarian Data Exchange (HDX): `https://data.humdata.org/dataset/ebola-cases-2014`
   - Our World in Data (OWID): `https://github.com/owid/etl`
4. **Acquisition & Extraction Task**:
   - Filter `country_timeseries.csv` for `SierraLeone`, extract dates from May 25, 2014 to September 21, 2014 ($T = 120$ days), and generate standard cumulative/daily features per `data/README.md`.

---

## 7. Deliverables & Technical Module Breakdown

- **Module 1: Biological Motivation, Formulation & Published Models Synthesis**:
  - Biological problem and epidemiological dynamics of Ebola virus transmission.
  - Dimensionless SEIRD formulation and conservation invariant proof.
  - Rigorous comparative analysis of 3 published benchmark models (Legrand et al., Rivers et al., Althaus).
  - Phase-space trajectory derivation, analytical peak infection fraction $i_{\max}$, and herd immunity thresholds $p_c$.
  - 2D parameter contour mapping of $R_0(\beta, \gamma+\mu)$.
- **Module 2: Mathematical Analysis & Equilibrium Dynamics**:
  - Basic reproduction number derivation via the next-generation matrix ($R_0 = \beta/(\gamma+\mu)$).
  - Disease-Free Equilibrium (DFE) Jacobian linearization and asymptotic stability conditions.
  - Transcendental final-size equation derivation and verification against numerical limits.
  - Global Latin Hypercube Sampling Partial Rank Correlation Coefficients (LHS-PRCC) sensitivity analysis.
- **Module 3: Numerical Solvers & Convergence Verification**:
  - Vectorized Forward Euler and classical 4th-order Runge-Kutta (RK4) integrators implemented from scratch.
  - Machine-precision conservation invariant tracking ($\max |s+e+i+r+d - 1| \le 10^{-15}$).
  - Convergence analysis against high-precision adaptive DOP853 benchmark across step sizes $h \in [0.03125, 1.0]$.
  - Log-log regression verifying empirical convergence orders ($p_{\text{Euler}} \approx 1.0, p_{\text{RK4}} \approx 3.7 - 4.0$).
- **Module 4: Physics-Informed Neural Network Architecture & Inversion**:
  - Dual-network formulation ($\mathcal{NN}_\theta$ for latent states, $\mathcal{NN}_\phi$ with Softplus for positive $\beta(t)$).
  - Scaled multi-objective physics loss with smoothness regularizer.
  - Experiment E2: Synthetic forward integration and relative $L_2$ error benchmarks.
  - Experiment E3: Inverse recovery of step-like and sinusoidal transmission functions under observational noise.
  - Experiment E4: Latent unobserved state recovery ($s(t)$ and $e(t)$ reconstructed solely from cases and deaths).
- **Module 5: Real-World Outbreak Application & Benchmarking**:
  - Empirical surveillance data acquisition protocol for the 2014 Sierra Leone Ebola outbreak.
  - Strict 75% training / 25% held-out test partitioning protocol.
  - Classical benchmark comparisons: constant-$\beta$ and regularized piecewise-$\beta$ least-squares estimation.
  - Quantitative held-out evaluation (MAE, RMSE, Relative Percentage Error).
  - Experiment E6 hyperparameter ablations ($N_f$ collocation density and $w_{\text{ode}}$ loss weight).
  - Honest scientific interpretation of inverse interpolation vs forward extrapolation limits, and AI disclosures.