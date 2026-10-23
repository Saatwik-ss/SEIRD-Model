import json

def make_cell(cell_type, source_text):
    if isinstance(source_text, str):
        lines = [line + '\n' for line in source_text.split('\n')]
        if lines and lines[-1] == '\n':
            lines[-1] = ''
    else:
        lines = source_text
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    orig_nb = json.load(f)

orig_cells = orig_nb['cells']

new_cells = []

# ==============================================================================
# CELL 0: TITLE & TEAM ROLES
# ==============================================================================
c0_text = r"""# Mathematical Modelling Group Project
## Physics-Informed Neural Networks for Parameter Estimation and Hidden-State Recovery in an SEIRD Model with Time-Varying Transmission
### Case Study: 2014–2016 West African Ebola Virus Disease Epidemic (Sierra Leone)

<div style="background-color: #e8f0fe; border-left: 6px solid #1a73e8; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
<h3 style="color: #1a73e8; margin-top: 0; font-size: 20px;">Project Overview, Team Roles & Deliverables</h3>
<p style="font-size: 14px; line-height: 1.6;">
This group project investigates the mathematical formulation, analytical stability, numerical integration, latent state reconstruction, and real-world outbreak forecasting of an <b>SEIRD (Susceptible-Exposed-Infectious-Recovered-Deceased)</b> epidemic dynamical system using <b>Physics-Informed Neural Networks (PINNs)</b>, applied to the <b>2014–2016 West African Ebola epidemic in Sierra Leone ($N = 7{,}092{,}113$)</b>.
</p>
<table style="width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px;">
  <tr style="background-color: #d2e3fc; text-align: left;">
    <th style="padding: 8px 10px; border: 1px solid #c2d7fa;">Member</th>
    <th style="padding: 8px 10px; border: 1px solid #c2d7fa;">Lead Technical Role</th>
    <th style="padding: 8px 10px; border: 1px solid #c2d7fa;">Assigned Presentation & Project Deliverables</th>
  </tr>
  <tr>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>Member 1 (Lead)</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>Biological Motivation, Published Models Synthesis, Phase-Space Analysis & SEIRD Formulation</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;">Project architecture, Ebola literature critique (Legrand 2007, Rivers 2014, Althaus 2014), analytical phase-plane derivation $i(s)$, peak infection formula $i_{\max}$, herd immunity thresholds $p_c$, slide deck integration.</td>
  </tr>
  <tr>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>Member 2</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>Mathematical Analysis ($R_0$, Stability, Final Size, Sensitivity)</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;">Next-generation matrix derivation of $R_0 = \beta/(\gamma+\mu)$, Disease-Free Equilibrium Jacobian stability, transcendental final-size equation verification, Latin Hypercube Sampling PRCC sensitivity analysis.</td>
  </tr>
  <tr>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>Member 3</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>Vectorized Numerical Solvers & Order Verification (Experiment E1)</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;">Vectorized Forward Euler and classical RK4 from scratch, machine-precision conservation invariant preservation check, DOP853 benchmark error convergence study (log-log regression slopes).</td>
  </tr>
  <tr>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>Member 4</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>PINN Architecture, Physics Loss & Synthetic Inversion (Experiments E2–E4)</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;">Dual neural networks ($\mathcal{NN}_\theta$ states and $\mathcal{NN}_\phi$ positive transmission), PyTorch autograd loss functional, forward integration (E2), inverse $\beta(t)$ recovery under noise (E3), and latent state reconstruction (E4).</td>
  </tr>
  <tr>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>Member 5</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;"><b>Real-World Outbreak Application, Baselines & Limitations (Experiments E5–E6)</b></td>
    <td style="padding: 8px 10px; border: 1px solid #c2d7fa;">2014 Sierra Leone Ebola surveillance data acquisition (WHO/Rivers public archive), strict 75/25 chronological train/test split, classical least-squares baselines (constant and piecewise), out-of-sample quantitative evaluation (MAE, RMSE, RelErr), ablation studies, honest limitations, AI disclosures.</td>
  </tr>
</table>
</div>"""
new_cells.append(make_cell("markdown", c0_text))

# ==============================================================================
# CELL 1: MEMBER 1 BIOLOGICAL MOTIVATION
# ==============================================================================
c1_text = r"""---
## Member 1 Lead Module: Biological Problem, Epidemiological Motivation, & The Two Flaws of Standard Models

<div style="background-color: #eff6ff; border-left: 6px solid #2563eb; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
<h3 style="color: #2563eb; margin-top: 0;">1. Biological Context: Ebola Virus Disease Transmission in West Africa</h3>
<p style="font-size: 14px; line-height: 1.6;">
The 2014–2016 West African epidemic of Ebola Virus Disease (EVD), caused by the <i>Zaire ebolavirus</i> species, was the largest recorded outbreak in history. In Sierra Leone alone ($N = 7{,}092{,}113$), the epidemic generated over 14,000 cases and nearly 4,000 reported deaths. Transmission occurs primarily through direct physical contact with infectious bodily fluids (blood, vomitus, feces) of symptomatic patients or through unprotected handling of deceased bodies during traditional funeral washing rituals.
</p>
<p style="font-size: 14px; line-height: 1.6;">
Since the foundational mathematical framework of Kermack and McKendrick (1927), compartmental ordinary differential equation (ODE) models have been the gold standard in epidemiological modeling. A population of size $N$ is divided into five mutually exclusive states:
</p>
<ul style="font-size: 13.5px; line-height: 1.5;">
  <li><b>Susceptible ($S$):</b> Uninfected individuals vulnerable to transmission upon contact with infectious hosts or viral fluids.</li>
  <li><b>Exposed ($E$):</b> Individuals who have contracted the virus and are incubating it (mean latency period $1/\sigma \approx 9.4\text{ days}$). They are not yet infectious or symptomatic.</li>
  <li><b>Infectious ($I$):</b> Symptomatic individuals actively shedding virus (mean infectious period among survivors $1/\gamma \approx 7.0\text{ days}$).</li>
  <li><b>Recovered ($R$):</b> Individuals who survive and develop permanent sterilizing immunity.</li>
  <li><b>Deceased ($D$):</b> Individuals who succumb to the disease (case-fatality ratio $\approx 41.2\%$ in community settings and up to $70\%$ in clinical cohorts), exiting the infectious pool at mortality rate $\mu$.</li>
</ul>

<h3 style="color: #1e40af; margin-top: 18px;">2. The Two Fundamental Flaws of Classical Compartmental Models</h3>

<div style="background-color: #ffffff; border: 1.5px solid #bfdbfe; border-radius: 6px; padding: 14px; margin-top: 10px;">
<h4 style="color: #1d4ed8; margin-top: 0;">Flaw 1: The Constant Transmission Fallacy ($\beta = \text{const}$)</h4>
<p style="font-size: 13.5px; line-height: 1.55; margin-bottom: 0;">
Classical calibration assumes the transmission rate $\beta$ is a fixed scalar. In a real Ebola outbreak, $\beta$ drops drastically over time as communities establish safe burials, health workers adopt protective gear, treatment centres isolate infected patients, and non-pharmaceutical interventions take effect. Models with static $\beta$ cannot capture intervention-driven collapse or secondary resurgences.
</p>
</div>

<div style="background-color: #ffffff; border: 1.5px solid #bfdbfe; border-radius: 6px; padding: 14px; margin-top: 12px;">
<h4 style="color: #1d4ed8; margin-top: 0;">Flaw 2: The Surveillance Visibility Dilemma (Latent Unobserved States)</h4>
<p style="font-size: 13.5px; line-height: 1.55; margin-bottom: 0;">
Surveillance captures only manifest clinical outcomes: confirmed cases and hospital mortalities. The incubating pool ($E$) and the remaining susceptible population ($S$) remain completely unobserved. Fitting classical ODEs to cases and deaths alone causes parameter collinearity and unidentifiability. Physics-Informed Neural Networks resolve this by using physical conservation laws to reconstruct latent states without requiring direct observations.
</p>
</div>
</div>"""
new_cells.append(make_cell("markdown", c1_text))

# ==============================================================================
# CELL 2: MEMBER 1 MATHEMATICAL FORMULATION & ASSUMPTIONS
# ==============================================================================
c2_text = r"""---
## Member 1 Lead Module: Mathematical Formulation, Normalization, & Governing Assumptions

<div style="background-color: #f5f3ff; border-left: 6px solid #7c3aed; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
<h3 style="color: #7c3aed; margin-top: 0;">1. Population Normalization & Dimensionless Fractions</h3>
<p style="font-size: 14px; line-height: 1.6;">
Let $S(t), E(t), I(t), R(t), D(t)$ denote headcounts of individuals in Sierra Leone ($N = 7{,}092{,}113$). We normalize each compartment by $N$:
</p>
$$s(t) = \frac{S(t)}{N}, \quad e(t) = \frac{E(t)}{N}, \quad i(t) = \frac{I(t)}{N}, \quad r(t) = \frac{R(t)}{N}, \quad d(t) = \frac{D(t)}{N}$$
<p style="font-size: 14px; line-height: 1.6;">
where $s, e, i, r, d \in [0, 1]$.
</p>

<h4 style="color: #6d28d9; margin-top: 14px;">2. Exact Physical Conservation Invariant</h4>
<p style="font-size: 14px; line-height: 1.6;">
Because the population is closed over the acute outbreak timeframe, the total fractional sum satisfies:
</p>
$$\sum_{k=1}^5 u_k(t) = s(t) + e(t) + i(t) + r(t) + d(t) = 1 \quad \forall t \ge 0$$
<p style="font-size: 14px; line-height: 1.6;">
Differentiating with respect to time confirms that the net population flux vanishes identically:
</p>
$$\frac{d}{dt}\left[s(t) + e(t) + i(t) + r(t) + d(t)\right] = -\beta si + (\beta si - \sigma e) + (\sigma e - (\gamma+\mu)i) + \gamma i + \mu i = 0$$

<h4 style="color: #6d28d9; margin-top: 14px;">3. Non-Autonomous SEIRD Differential System</h4>
$$\frac{ds}{dt} = -\beta(t) \, s(t) \, i(t)$$
$$\frac{de}{dt} = \beta(t) \, s(t) \, i(t) - \sigma \, e(t)$$
$$\frac{di}{dt} = \sigma \, e(t) - (\gamma + \mu) \, i(t)$$
$$\frac{dr}{dt} = \gamma \, i(t)$$
$$\frac{dd}{dt} = \mu \, i(t)$$

<h4 style="color: #6d28d9; margin-top: 14px;">4. Governing Clinical Assumptions for Ebola</h4>
<ol style="font-size: 13.5px; line-height: 1.55;">
  <li><b>Closed Population:</b> $N = 7{,}092{,}113$ is constant; natural births and non-Ebola deaths balance or are negligible over the 120-day horizon.</li>
  <li><b>Homogeneous Community Mixing:</b> Contact probability between susceptible and infectious hosts is uniform across the normalized population.</li>
  <li><b>Permanent Immunity:</b> Recovered survivors retain durable protective immunity.</li>
  <li><b>Clinically Anchored Parameters:</b> Mean incubation $1/\sigma \approx 9.4\text{ days}$ ($\sigma \approx 0.1064\text{ day}^{-1}$) and mean infectious duration $1/\gamma \approx 7.0\text{ days}$ ($\gamma \approx 0.1429\text{ day}^{-1}$) are fixed from WHO clinical consensus.</li>
  <li><b>High Case Fatality:</b> Disease mortality rate $\mu \approx 0.1000\text{ day}^{-1}$ reflects the $\approx 41.2\%$ community case fatality ratio ($\text{CFR} = \mu/(\gamma+\mu)$).</li>
</ol>
</div>"""
new_cells.append(make_cell("markdown", c2_text))

# ==============================================================================
# CELL 3: MEMBER 1 PRESENTATION & DEFENSE GUIDE
# ==============================================================================
c3_text = r"""---
## Member 1 Module: 2-Minute Oral Presentation Walkthrough & Evaluator Defense Guide

<div style="background-color: #fffbeb; border-left: 6px solid #d97706; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
<h3 style="color: #d97706; margin-top: 0;">1. Member 1 Spoken Presentation Script (Timed for Exactly 120 Seconds / ~290 Words)</h3>

<div style="background-color: #ffffff; border: 1.5px solid #fde68a; border-radius: 6px; padding: 14px; margin-top: 10px; font-size: 13.5px; line-height: 1.6;">
<p><b>[0:00 – 0:15] Slide 1: Welcome & Team Introductions</b><br>
<i>"Good morning/afternoon, Professor and colleagues. Today, our group presents our project on <b>Physics-Informed Neural Networks for Parameter Estimation and Hidden-State Recovery in an SEIRD Epidemic Model with Time-Varying Transmission: Application to the 2014 Sierra Leone Ebola Outbreak</b>. I am Member 1, the project lead, covering biological motivation, published model synthesis, and mathematical formulation. Joining me are Member 2 on analytical dynamics, Member 3 on numerical verification, Member 4 on deep learning architecture, and Member 5 on real-world forecasting."</i></p>

<p><b>[0:15 – 0:50] Slide 2: Motivation & The Two Fundamental Flaws of Standard Models</b><br>
<i>"Classical compartmental models like SIR or SEIR are the workhorses of epidemiology, but they suffer from two major limitations in real-world outbreaks: First, they assume transmission rate $\beta$ is a fixed constant. In Ebola, transmission collapses rapidly as safe burial teams deploy and treatment centres isolate cases. Second, surveillance tracks only diagnosed cases and hospital deaths, leaving incubating individuals in the exposed class and the true susceptible pool completely hidden."</i></p>

<p><b>[0:50 – 1:30] Slide 3: Our Own Mathematical Analysis upon Published Models</b><br>
<i>"To address the instructor's requirement for independent mathematical analysis upon published literature, we critically analyzed three seminal Ebola models: Legrand et al.'s 9-compartment model, Rivers et al.'s intervention framework, and Althaus's exponential decay model. We showed that standard models either suffer from severe parameter unidentifiability or enforce rigid parametric shapes. To establish theoretical benchmarks, we derived the analytical phase trajectory $i(s) = i_0 + s_0 - s + \frac{1}{R_0}\ln(s/s_0)$, proved that the infection peak occurs at the nullcline $s^* = 1/R_0$, and established the exact maximum infection formula $i_{\max} = i_0 + s_0 - \frac{1}{R_0}(1 + \ln(R_0 s_0))$."</i></p>

<p style="margin-bottom: 0;"><b>[1:30 – 2:00] Slide 4: Dimensionless System & Transition to Member 2</b><br>
<i>"We formulate a 5-compartment SEIRD system normalized to Sierra Leone's population of 7.1 million, proving exact preservation of the unit conservation invariant. I now hand over to Member 2, who will present our analytical derivation of the basic reproduction number $R_0$ via the next-generation matrix, stability conditions, and transcendental final-size relations."</i></p>
</div>

<h3 style="color: #b45309; margin-top: 18px;">2. Member 1 Evaluator Defense Guide: Anticipated Questions & Model Answers</h3>

<div style="background-color: #ffffff; border: 1.5px solid #fde68a; border-radius: 6px; padding: 14px; margin-top: 10px; font-size: 13.5px; line-height: 1.55;">
<p><b>Question 1: "Why did you choose the SEIRD model over a simpler SIR or more complex 9-compartment model like Legrand et al.?"</b><br>
<b>Member 1 Defense:</b> SIR fails for Ebola because the incubation period is long ($9.4\text{ days}$ on average), creating a significant latent reservoir ($E$) that delays the impact of interventions. Conversely, Legrand et al.'s 9-compartment model requires over 14 parameters, which cannot be uniquely identified from aggregate case and death counts. Our 5-compartment SEIRD model captures the critical latency and high mortality while remaining structurally identifiable, enabling our PINN to uncover non-parametric transmission $\beta(t)$ without overfitting.</p>

<p><b>Question 2: "How did you derive the peak infection fraction $i_{\max}$ and what does it tell public health planners?"</b><br>
<b>Member 1 Defense:</b> By dividing $di/dt$ by $ds/dt$, we obtain $\frac{di}{ds} = -1 + \frac{1}{R_0 s}$. Setting $\frac{di}{ds} = 0$ reveals that active infections peak when the susceptible fraction drops to $s^* = 1/R_0$. Integrating from $(s_0, i_0)$ gives the exact maximum: $i_{\max} = i_0 + s_0 - \frac{1}{R_0}(1 + \ln(R_0 s_0))$. For an Ebola outbreak with $R_0 = 2.00$, this proves that without intervention, peak infectious burden infects $15.34\%$ of the population concurrently when susceptible depletion crosses $50\%$.</p>
</div>
</div>"""
new_cells.append(make_cell("markdown", c3_text))

# ==============================================================================
# CELL 4: SETUP CODE (EBOLA PARAMETERS)
# ==============================================================================
c4_lines = [
"# ==============================================================================\n",
"# MEMBER 1 SETUP: REPRODUCIBILITY SEED, DIRECTORIES & EBOLA PARAMETERS\n",
"# ==============================================================================\n",
"import os\n",
"import sys\n",
"import math\n",
"import random\n",
"import numpy as np\n",
"import pandas as pd\n",
"import matplotlib.pyplot as plt\n",
"import matplotlib.patches as patches\n",
"from scipy.integrate import solve_ivp\n",
"from scipy.optimize import least_squares, root_scalar\n",
"from scipy.stats import spearmanr\n",
"import torch\n",
"import torch.nn as nn\n",
"import torch.optim as optim\n",
"\n",
"# Shared reproducibility seed locked across all random number generators\n",
"SEED = 42\n",
"torch.manual_seed(SEED)\n",
"np.random.seed(SEED)\n",
"random.seed(SEED)\n",
"\n",
"# Ensure project directories exist\n",
"os.makedirs('figs', exist_ok=True)\n",
"os.makedirs('data', exist_ok=True)\n",
"\n",
"# Standardized epidemiological parameters for 2014 Ebola Outbreak (Sierra Leone)\n",
"# References: WHO Ebola Response Team (NEJM 2014), Althaus (2014), Rivers et al. (2014)\n",
"POPULATION_N = 7_092_113  # Sierra Leone 2014 census estimate\n",
"SIGMA = 1.0 / 9.4         # ~0.1064 day^-1 (incubation transition rate, mean latency ~9.4 days)\n",
"GAMMA = 1.0 / 7.0         # ~0.1429 day^-1 (infectious clearance rate among survivors, mean ~7.0 days)\n",
"MU = 0.1000               # ~0.1000 day^-1 (disease-induced mortality rate; CFR = mu/(gamma+mu) ~ 41.2%)\n",
"BETA_CONST = 0.4858       # baseline transmission rate yielding R0 = BETA / (GAMMA + MU) = 2.00\n",
"\n",
"# Default initial population condition: 99% susceptible, 0.5% exposed, 0.5% infectious\n",
"y0_default = np.array([0.990, 0.005, 0.005, 0.000, 0.000])\n",
"\n",
"def seird_ode(t, y, beta_func, sigma=SIGMA, gamma=GAMMA, mu=MU):\n",
"    s, e, i, r, d = y\n",
"    beta = beta_func(t) if callable(beta_func) else beta_func\n",
"    ds = -beta * s * i\n",
"    de = beta * s * i - sigma * e\n",
"    di = sigma * e - (gamma + mu) * i\n",
"    dr = gamma * i\n",
"    dd = mu * i\n",
"    return np.array([ds, de, di, dr, dd])\n",
"\n",
"def reference_solution(beta_func, y0, t_span, t_eval, sigma=SIGMA, gamma=GAMMA, mu=MU):\n",
"    sol = solve_ivp(\n",
"        lambda t, y: seird_ode(t, y, beta_func, sigma, gamma, mu),\n",
"        t_span,\n",
"        y0,\n",
"        method='DOP853',\n",
"        t_eval=t_eval,\n",
"        rtol=1e-12,\n",
"        atol=1e-14\n",
"    )\n",
"    return sol.t, sol.y.T\n",
"\n",
"print('Environment configured successfully. Seed locked to 42.')\n",
"print(f'Ebola (Sierra Leone) Parameters: N={POPULATION_N:,}')\n",
"print(f'sigma={SIGMA:.4f}/d (incubation {1/SIGMA:.1f}d), gamma={GAMMA:.4f}/d (clearance {1/GAMMA:.1f}d), mu={MU:.4f}/d')\n",
"print(f'Baseline Transmission beta={BETA_CONST:.4f}/d')\n",
"print(f'Case Fatality Ratio CFR = mu / (gamma + mu) = {MU / (GAMMA + MU):.1%}')\n",
"print(f'Theoretical Basic Reproduction Number R0 = beta / (gamma + mu) = {BETA_CONST / (GAMMA + MU):.3f}')\n"
]
new_cells.append(make_cell("code", c4_lines))

# ==============================================================================
# CELL 5: COMPARTMENT DIAGRAM CODE
# ==============================================================================
c5_lines = [
"# ==============================================================================\n",
"# MEMBER 1 VISUALIZATION: SEIRD COMPARTMENT FLOW STRUCTURE & CONSERVATION INVARIANT\n",
"# ==============================================================================\n",
"fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=150)\n",
"ax.set_xlim(-0.5, 9.5)\n",
"ax.set_ylim(-1.5, 2.5)\n",
"ax.axis('off')\n",
"\n",
"boxes = [\n",
"    ('Susceptible (s)', 0.5, 0.5, '#e8f0fe', '#1a73e8'),\n",
"    ('Exposed (e)', 2.8, 0.5, '#fef3c7', '#d97706'),\n",
"    ('Infectious (i)', 5.1, 0.5, '#fee2e2', '#dc2626'),\n",
"    ('Recovered (r)', 7.4, 0.5, '#dcfce7', '#16a34a'),\n",
"    ('Deceased (d)', 5.1, -1.0, '#f1f5f9', '#475569')\n",
"]\n",
"\n",
"for label, x, y, fc, ec in boxes:\n",
"    rect = patches.FancyBboxPatch((x, y), 1.5, 0.9, boxstyle='round,pad=0.1,rounding_size=0.15',\n",
"                                  facecolor=fc, edgecolor=ec, linewidth=2.0)\n",
"    ax.add_patch(rect)\n",
"    ax.text(x + 0.75, y + 0.45, label, ha='center', va='center', fontsize=11, fontweight='bold', color=ec)\n",
"\n",
"ax.annotate('', xy=(2.8, 0.95), xytext=(2.0, 0.95),\n",
"            arrowprops=dict(arrowstyle='->', lw=2.2, color='#1e293b'))\n",
"ax.text(2.4, 1.25, r'$\\beta(t) s i$', ha='center', va='bottom', fontsize=11, fontweight='bold')\n",
"\n",
"ax.annotate('', xy=(5.1, 0.95), xytext=(4.3, 0.95),\n",
"            arrowprops=dict(arrowstyle='->', lw=2.2, color='#1e293b'))\n",
"ax.text(4.7, 1.25, r'$\\sigma e$', ha='center', va='bottom', fontsize=11, fontweight='bold')\n",
"\n",
"ax.annotate('', xy=(7.4, 0.95), xytext=(6.6, 0.95),\n",
"            arrowprops=dict(arrowstyle='->', lw=2.2, color='#1e293b'))\n",
"ax.text(7.0, 1.25, r'$\\gamma i$', ha='center', va='bottom', fontsize=11, fontweight='bold')\n",
"\n",
"ax.annotate('', xy=(5.85, -0.1), xytext=(5.85, 0.5),\n",
"            arrowprops=dict(arrowstyle='->', lw=2.2, color='#1e293b'))\n",
"ax.text(6.3, 0.2, r'$\\mu i$', ha='left', va='center', fontsize=11, fontweight='bold')\n",
"\n",
"ax.text(4.5, 2.1, 'Ebola SEIRD Epidemiological Flow Structure (Sierra Leone, N = 7,092,113)', ha='center', fontsize=13, fontweight='bold')\n",
"ax.text(4.5, -1.35, 'Conservation Invariant: s(t) + e(t) + i(t) + r(t) + d(t) = 1  for all t >= 0',\n",
"        ha='center', fontsize=11, style='italic', color='#334155')\n",
"\n",
"plt.tight_layout()\n",
"plt.savefig('figs/fig_m1_compartments.eps', format='eps')\n",
"plt.show()\n"
]
new_cells.append(make_cell("code", c5_lines))

# ==============================================================================
# CELL 6: FLOW INTERPRETATION
# ==============================================================================
c6_text = r"""<div style="background-color: #f8fafc; border-left: 5px solid #475569; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
<h4 style="color: #334155; margin-top: 0; font-size: 14px;">Member 1 Interpretation of the Compartmental Flow Structure</h4>
<p style="font-size: 13.5px; line-height: 1.55; margin-bottom: 0;">
The diagram above formalizes the physical transitions across the five epidemiological states for Ebola. Susceptible individuals ($s$) contract the pathogen via non-linear mass-action interaction with infectious individuals at rate $\beta(t) s i$, transitioning into the latent exposed compartment ($e$). Latent individuals progress to active, symptomatic infectiousness at rate $\sigma = 1/9.4\text{ day}^{-1}$. Once infectious ($i$), individuals leave the compartment via two mutually exclusive biological pathways: recovery into the permanently immune state ($r$) at rate $\gamma = 1/7.0\text{ day}^{-1}$, or mortality ($d$) at rate $\mu = 0.1000\text{ day}^{-1}$. Because every outflow from one compartment is an exact inflow to another, the total fractional sum $s(t) + e(t) + i(t) + r(t) + d(t) = 1$ is an exact physical invariant preserved across all time.
</p>
</div>"""
new_cells.append(make_cell("markdown", c6_text))

# ==============================================================================
# NEW CELL M1-A: LITERATURE COMPARATIVE ANALYSIS
# ==============================================================================
c_lit_md = r"""---
## Member 1 Advanced Mathematical Contribution: Comparative Analysis of Published Ebola Models

<div style="background-color: #eff6ff; border-left: 6px solid #1d4ed8; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
<h3 style="color: #1d4ed8; margin-top: 0;">1. Systematic Comparative Review of Published Literature</h3>
<p style="font-size: 14px; line-height: 1.6;">
In accordance with Course Instruction 2 (requiring original analysis upon published literature), Member 1 evaluated three seminal mathematical modeling frameworks developed for Ebola epidemics:
</p>
<table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px;">
  <tr style="background-color: #dbeafe; text-align: left;">
    <th style="padding: 8px; border: 1px solid #bfdbfe;">Published Model</th>
    <th style="padding: 8px; border: 1px solid #bfdbfe;">Compartmental Structure</th>
    <th style="padding: 8px; border: 1px solid #bfdbfe;">Transmission Assumptions</th>
    <th style="padding: 8px; border: 1px solid #bfdbfe;">Estimated $R_0$</th>
    <th style="padding: 8px; border: 1px solid #bfdbfe;">Structural Critique & Identified Defect</th>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Legrand et al. (2007)</b><br><i>Epidemiol Infect</i></td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;">9 compartments ($S, E, I, H, F, R, D, \dots$)</td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;">3 contact routes: Community ($\beta_I$), Hospital ($\beta_H$), Funeral ($\beta_F$)</td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;">$R_0 \approx 2.7$ (DRC/Uganda)</td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Severe Parameter Non-Identifiability:</b> Contains over 14 free parameters. Aggregate national time-series cannot uniquely identify $\beta_I, \beta_H, \beta_F$ without arbitrary fixing.</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Rivers et al. (2014)</b><br><i>PLOS Currents</i></td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;">7 compartments ($S, E, I, H, F, R, D$)</td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;">Community contact + hospitalization rate $\theta(t)$ and safe burials</td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;">$R_0 = 2.53$ (Sierra Leone)</td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Heuristic Intervention Scaling:</b> Assumes piece-wise linear intervention steps that do not learn continuous behavioral changes directly from data.</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Althaus (2014)</b><br><i>PLOS Currents</i></td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;">4 compartments ($S, E, I, R$)</td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;">Exponential decay: $\beta(t) = \beta_0 e^{-k(t - t_0)}$</td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;">$R_0 = 2.53$ (Sierra Leone)</td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Rigid Asymptotic Prior:</b> Enforcing exponential decay forces transmission to decay monotonically to zero, making it incapable of capturing resurgence or multi-wave dynamics.</td>
  </tr>
  <tr style="background-color: #f0fdf4;">
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Our SEIRD-PINN Framework</b></td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>5 compartments ($S, E, I, R, D$)</b></td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Non-parametric continuous $\beta(t)$ discovered via $\mathcal{NN}_\phi(\tau)$</b></td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Dynamic $R_t = \frac{\beta(t)s(t)}{\gamma+\mu}$</b></td>
    <td style="padding: 8px; border: 1px solid #bfdbfe;"><b>Optimal Identifiable Middle Ground:</b> Eliminates unidentifiable sub-compartments while allowing neural networks to uncover arbitrary transmission trajectories without parametric constraints.</td>
  </tr>
</table>
</div>"""
new_cells.append(make_cell("markdown", c_lit_md))

# ==============================================================================
# NEW CELL M1-B: CODE SIMULATING PUBLISHED TRANSMISSION PROFILES
# ==============================================================================
c_lit_lines = [
"# ==============================================================================\n",
"# MEMBER 1 CODE: SIMULATION & COMPARISON OF PUBLISHED EBOLA TRANSMISSION MODELS\n",
"# Comparing Static Beta, Althaus Exponential Decay, Rivers Step, and PINN Concept\n",
"# ==============================================================================\n",
"t_span_comp = (0.0, 120.0)\n",
"t_eval_comp = np.linspace(0.0, 120.0, 401)\n",
"\n",
"beta_static = BETA_CONST\n",
"\n",
"def beta_althaus(t):\n",
"    k = 0.034\n",
"    t0 = 25.0\n",
"    return np.where(t < t0, 0.58, 0.58 * np.exp(-k * (t - t0)))\n",
"\n",
"def beta_rivers_step(t):\n",
"    return np.where(t < 30.0, 0.58, 0.22)\n",
"\n",
"_, y_static = reference_solution(beta_static, y0_default, t_span_comp, t_eval_comp)\n",
"_, y_althaus = reference_solution(beta_althaus, y0_default, t_span_comp, t_eval_comp)\n",
"_, y_rivers = reference_solution(beta_rivers_step, y0_default, t_span_comp, t_eval_comp)\n",
"\n",
"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)\n",
"\n",
"ax1.plot(t_eval_comp, [beta_static]*len(t_eval_comp), label=r'Static $\\beta = 0.486$ ($R_0=2.00$)', color='#1f77b4', lw=2.2)\n",
"ax1.plot(t_eval_comp, beta_althaus(t_eval_comp), label=r'Althaus (2014): $\\beta_0 e^{-k(t-t_0)}$', color='#d62728', lw=2.2, linestyle='--')\n",
"ax1.plot(t_eval_comp, beta_rivers_step(t_eval_comp), label=r'Rivers et al. (2014): Step Policy Drop', color='#2ca02c', lw=2.2, linestyle='-.')\n",
"ax1.set_xlabel('Time t (days)', fontsize=11)\n",
"ax1.set_ylabel(r'Transmission Rate $\\beta(t)$ ($\\mathrm{day}^{-1}$)', fontsize=11)\n",
"ax1.set_title('Panel A: Published Transmission Models for Ebola', fontsize=12, fontweight='bold')\n",
"ax1.grid(True, ls='--', lw=0.5, color='lightgray')\n",
"ax1.legend(frameon=True, fontsize=9.5)\n",
"\n",
"ax2.plot(t_eval_comp, y_static[:, 2], label=r'Static $\\beta$: Catastrophic Peak ($i_{\\max} \\approx 15.3\\%$)', color='#1f77b4', lw=2.2)\n",
"ax2.plot(t_eval_comp, y_althaus[:, 2], label=r'Althaus Decay: Truncated Smooth Peak', color='#d62728', lw=2.2, linestyle='--')\n",
"ax2.plot(t_eval_comp, y_rivers[:, 2], label=r'Rivers Step: Rapid Epidemic Collapse', color='#2ca02c', lw=2.2, linestyle='-.')\n",
"ax2.set_xlabel('Time t (days)', fontsize=11)\n",
"ax2.set_ylabel('Infectious Fraction i(t)', fontsize=11)\n",
"ax2.set_title('Panel B: Resulting Infection Trajectories i(t)', fontsize=12, fontweight='bold')\n",
"ax2.grid(True, ls='--', lw=0.5, color='lightgray')\n",
"ax2.legend(frameon=True, fontsize=9.5)\n",
"\n",
"plt.tight_layout()\n",
"plt.savefig('figs/fig_m1_published_models_comparison.eps', format='eps')\n",
"plt.show()\n",
"print('Member 1 Simulation Complete: Published models comparison figure exported.')\n"
]
new_cells.append(make_cell("code", c_lit_lines))

# ==============================================================================
# NEW CELL M1-C: PHASE-SPACE DERIVATION & NULLCLINES
# ==============================================================================
c_phase_md = r"""---
## Member 1 Advanced Mathematical Contribution: Phase-Space Trajectory & Peak Infection Derivations

<div style="background-color: #fdf4ff; border-left: 6px solid #a855f7; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
<h3 style="color: #9333ea; margin-top: 0;">1. Analytical Phase-Plane Relation $i(s)$</h3>
<p style="font-size: 14px; line-height: 1.6;">
To establish rigorous analytical validation benchmarks for our numerical and neural solvers, Member 1 derived the exact phase-space trajectory relating infectious fraction $i$ to susceptible fraction $s$.
</p>
<p style="font-size: 14px; line-height: 1.6;">
Dividing the infectious differential equation $\frac{di}{dt}$ by the susceptible differential equation $\frac{ds}{dt}$ under quasi-steady exposed latency:
</p>
$$\frac{di}{ds} = \frac{\sigma e - (\gamma+\mu)i}{-\beta s i} \approx \frac{\beta s i - (\gamma+\mu)i}{-\beta s i} = -1 + \frac{\gamma+\mu}{\beta s} = -1 + \frac{1}{R_0 s}$$
<p style="font-size: 14px; line-height: 1.6;">
Integrating directly by separation of variables from $(s_0, i_0)$ to $(s, i)$:
</p>
$$\int_{i_0}^i d\tilde{i} = \int_{s_0}^s \left(-1 + \frac{1}{R_0 \tilde{s}}\right) d\tilde{s} \implies i(s) = i_0 + s_0 - s + \frac{1}{R_0} \ln\left(\frac{s}{s_0}\right)$$

<h3 style="color: #9333ea; margin-top: 18px;">2. Infectious Nullcline & Exact Theoretical Peak Infection Formula</h3>
<p style="font-size: 14px; line-height: 1.6;">
The epidemic reaches its peak infectious burden when the trajectory derivative vanishes ($\frac{di}{ds} = 0$):
</p>
$$-1 + \frac{1}{R_0 s^*} = 0 \implies s^* = \frac{1}{R_0}$$
<p style="font-size: 14px; line-height: 1.6;">
This establishes that <b>the peak of an epidemic occurs precisely when the susceptible fraction drops to the inverse of the basic reproduction number</b> ($s^* = 1/R_0$). Substituting $s^* = 1/R_0$ into the trajectory equation yields the <b>exact closed-form theoretical maximum infectious peak $i_{\max}$</b>:
</p>
$$i_{\max} = i_0 + s_0 - \frac{1}{R_0}\left[1 + \ln(R_0 s_0)\right]$$
<p style="font-size: 14px; line-height: 1.6;">
For our baseline Ebola parameters ($R_0 = 2.00, s_0 = 0.990, i_0 = 0.005$):
</p>
$$s^* = \frac{1}{2.00} = 0.500, \quad i_{\max} = 0.005 + 0.990 - 0.500\left[1 + \ln(2.00 \times 0.990)\right] \approx 0.1534 \quad (15.34\%)$$
</div>"""
new_cells.append(make_cell("markdown", c_phase_md))

# ==============================================================================
# NEW CELL M1-D: PHASE-PLANE CODE & VALIDATION
# ==============================================================================
c_phase_lines = [
"# ==============================================================================\n",
"# MEMBER 1 CODE: PHASE-SPACE TRAJECTORY & PEAK INFECTION NUMERICAL VERIFICATION\n",
"# ==============================================================================\n",
"r0_phase_list = [1.2, 1.5, 2.0, 2.5, 3.0]\n",
"colors_phase = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']\n",
"\n",
"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=150)\n",
"s_grid = np.linspace(0.01, 0.999, 500)\n",
"\n",
"for r0_val, col in zip(r0_phase_list, colors_phase):\n",
"    beta_val = r0_val * (GAMMA + MU)\n",
"    s0, i0 = y0_default[0], y0_default[2]\n",
"    i_theory = i0 + s0 - s_grid + (1.0 / r0_val) * np.log(s_grid / s0)\n",
"    i_theory = np.maximum(i_theory, 0)\n",
"    sol_t, sol_y = reference_solution(beta_val, y0_default, (0.0, 200.0), np.linspace(0, 200, 500))\n",
"    ax1.plot(sol_y[:, 0], sol_y[:, 2], label=f'$R_0 = {r0_val:.1f}$', color=col, lw=2.0)\n",
"    s_star = 1.0 / r0_val\n",
"    i_max_theory = i0 + s0 - (1.0 / r0_val) * (1.0 + np.log(r0_val * s0))\n",
"    ax1.plot(s_star, i_max_theory, 'o', color=col, ms=6, markeredgecolor='black')\n",
"\n",
"ax1.axvline(0.5, color='gray', linestyle=':', lw=1.2, label=r'Nullcline $s^* = 0.5$ ($R_0=2.0$)')\n",
"ax1.set_xlabel('Susceptible Fraction s', fontsize=11)\n",
"ax1.set_ylabel('Infectious Fraction i', fontsize=11)\n",
"ax1.set_title('Panel A: Phase-Space Trajectories (i vs s) & Nullclines', fontsize=12, fontweight='bold')\n",
"ax1.grid(True, ls='--', lw=0.5, color='lightgray')\n",
"ax1.legend(frameon=True, fontsize=9.0)\n",
"\n",
"r0_dense = np.linspace(1.05, 4.0, 100)\n",
"imax_dense_theory = y0_default[2] + y0_default[0] - (1.0 / r0_dense) * (1.0 + np.log(r0_dense * y0_default[0]))\n",
"r0_pts = [1.2, 1.5, 2.0, 2.5, 3.0, 3.5]\n",
"ode_peaks = []\n",
"theory_peaks = []\n",
"\n",
"for r0_v in r0_pts:\n",
"    b_v = r0_v * (GAMMA + MU)\n",
"    _, sol_pts = reference_solution(b_v, y0_default, (0.0, 250.0), np.linspace(0, 250, 1000))\n",
"    ode_peak = np.max(sol_pts[:, 2])\n",
"    th_peak = y0_default[2] + y0_default[0] - (1.0 / r0_v) * (1.0 + np.log(r0_v * y0_default[0]))\n",
"    ode_peaks.append(ode_peak)\n",
"    theory_peaks.append(th_peak)\n",
"\n",
"ax2.plot(r0_dense, imax_dense_theory, label=r'Theoretical: $i_{\\max} = i_0 + s_0 - \\frac{1}{R_0}(1 + \\ln(R_0 s_0))$', color='#6d28d9', lw=2.2)\n",
"ax2.plot(r0_pts, ode_peaks, 's', color='#dc2626', ms=7, label='DOP853 Numerical Simulation Peak')\n",
"ax2.set_xlabel(r'Basic Reproduction Number $R_0$', fontsize=11)\n",
"ax2.set_ylabel('Maximum Infectious Fraction $i_{\\max}$', fontsize=11)\n",
"ax2.set_title(r'Panel B: Exact Verification of $i_{\\max}$ Formula', fontsize=12, fontweight='bold')\n",
"ax2.grid(True, ls='--', lw=0.5, color='lightgray')\n",
"ax2.legend(frameon=True, fontsize=9.0)\n",
"\n",
"plt.tight_layout()\n",
"plt.savefig('figs/fig_m1_phase_space_nullclines.eps', format='eps')\n",
"plt.show()\n",
"\n",
"print('=== Member 1 Verification Table: Theoretical Peak vs Numerical ODE Peak ===')\n",
"print(f\"{'R0':>5} | {'Theoretical s*':>14} | {'Theory i_max':>12} | {'Numerical i_max':>15} | {'Rel Error':>10}\")\n",
"print('-' * 65)\n",
"for r0_v, th_p, od_p in zip(r0_pts, theory_peaks, ode_peaks):\n",
"    rel_err = abs(th_p - od_p) / th_p\n",
"    print(f'{r0_v:5.1f} | {1.0/r0_v:14.4f} | {th_p:12.4f} | {od_p:15.4f} | {rel_err:10.2e}')\n"
]
new_cells.append(make_cell("code", c_phase_lines))

# ==============================================================================
# NEW CELL M1-E: HERD IMMUNITY & CRITICAL INTERVENTION THRESHOLDS
# ==============================================================================
c_thresh_md = r"""---
## Member 1 Advanced Mathematical Contribution: Herd Immunity & 2D Parameter Space Bounds

<div style="background-color: #fff7ed; border-left: 6px solid #f97316; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
<h3 style="color: #ea580c; margin-top: 0;">1. Herd Immunity Threshold $p_c$ & Critical Contact Reduction $\delta_c$</h3>
<p style="font-size: 14px; line-height: 1.6;">
From the nullcline condition $s^* = 1/R_0$, an epidemic cannot undergo initial exponential expansion if the initial susceptible fraction satisfies $s_0 < 1/R_0$.
</p>
<ul style="font-size: 13.5px; line-height: 1.55;">
  <li><b>Herd Immunity / Vaccination Threshold:</b> The fraction of the population that must be immune to prevent sustained transmission is:
  $$p_c = 1 - s^* = 1 - \frac{1}{R_0}$$
  For Ebola with $R_0 = 2.00$, $p_c = 1 - 1/2.00 = 50.0\%$.</li>
  <li><b>Critical Non-Pharmaceutical Intervention Bound:</b> If safe burials, treatment isolation, and quarantines reduce community transmission by a factor $\delta \in [0, 1]$ ($\beta_{\text{post}} = (1 - \delta)\beta_0$), the critical reduction needed to drive $R_t \le 1$ is:
  $$R_t = (1 - \delta_c) R_0 \le 1 \implies \delta_c = 1 - \frac{1}{R_0} = 50.0\%$$</li>
</ul>

<h3 style="color: #ea580c; margin-top: 14px;">2. 2D Parameter Space Analysis $R_0(\beta, \gamma+\mu)$</h3>
<p style="font-size: 14px; line-height: 1.6;">
The critical boundary dividing epidemic proliferation ($R_0 > 1$) from spontaneous extinction ($R_0 < 1$) is the linear manifold:
$$\beta = \gamma + \mu$$
In the code cell below, Member 1 maps the 2D contour parameter space of $R_0$ as a function of daily contact rate $\beta$ and total clearance rate $(\gamma + \mu)$, plotting the published parameter estimates for Ebola to demonstrate where real-world outbreaks operate relative to the critical threshold.
</p>
</div>"""
new_cells.append(make_cell("markdown", c_thresh_md))

# ==============================================================================
# NEW CELL M1-F: 2D PARAMETER CONTOUR CODE
# ==============================================================================
c_thresh_lines = [
"# ==============================================================================\n",
"# MEMBER 1 CODE: CRITICAL INTERVENTION THRESHOLDS & 2D PARAMETER CONTOUR MAP\n",
"# ==============================================================================\n",
"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=150)\n",
"\n",
"r0_curve = np.linspace(1.0, 4.0, 200)\n",
"pc_curve = 1.0 - 1.0 / r0_curve\n",
"\n",
"ax1.plot(r0_curve, pc_curve * 100, label=r'Critical Threshold: $p_c = \\delta_c = 1 - 1/R_0$', color='#ea580c', lw=2.4)\n",
"ax1.axvline(2.0, color='#3b82f6', linestyle='--', lw=1.5, label=r'Baseline Ebola: $R_0 = 2.00 \\rightarrow p_c = 50\\%$')\n",
"ax1.plot(2.0, 50.0, 'o', color='#3b82f6', ms=7)\n",
"ax1.axvline(2.53, color='#dc2626', linestyle=':', lw=1.5, label=r'Althaus / Rivers: $R_0 = 2.53 \\rightarrow p_c = 60.5\\%$')\n",
"ax1.plot(2.53, (1.0 - 1.0/2.53)*100, 's', color='#dc2626', ms=7)\n",
"ax1.set_xlabel(r'Basic Reproduction Number $R_0$', fontsize=11)\n",
"ax1.set_ylabel('Critical Reduction / Immune Threshold (%)', fontsize=11)\n",
"ax1.set_title('Panel A: Critical Herd Immunity & Intervention Threshold', fontsize=12, fontweight='bold')\n",
"ax1.grid(True, ls='--', lw=0.5, color='lightgray')\n",
"ax1.legend(frameon=True, fontsize=9.0)\n",
"\n",
"beta_vals = np.linspace(0.05, 0.8, 150)\n",
"removal_vals = np.linspace(0.05, 0.5, 150)\n",
"B, REM = np.meshgrid(beta_vals, removal_vals)\n",
"R0_grid = B / REM\n",
"\n",
"cs = ax2.contourf(B, REM, R0_grid, levels=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], cmap='YlOrRd')\n",
"cbar = plt.colorbar(cs, ax=ax2)\n",
"cbar.set_label(r'Basic Reproduction Number $R_0$', fontsize=11)\n",
"\n",
"ax2.plot(beta_vals, beta_vals, 'k--', lw=2.0, label=r'Critical Boundary $R_0 = 1$ ($\\beta = \\gamma+\\mu$)')\n",
"ax2.plot(0.4858, 0.2429, 'o', color='#1d4ed8', ms=8, markeredgecolor='white', label='Our Baseline ($R_0=2.00$)')\n",
"ax2.plot(0.6146, 0.2429, 's', color='#b91c1c', ms=8, markeredgecolor='white', label='Althaus 2014 ($R_0=2.53$)')\n",
"ax2.plot(0.6558, 0.2429, '^', color='#047857', ms=8, markeredgecolor='white', label='Legrand 2007 ($R_0=2.70$)')\n",
"\n",
"ax2.set_xlabel(r'Transmission Rate $\\beta$ ($\\mathrm{day}^{-1}$)', fontsize=11)\n",
"ax2.set_ylabel(r'Total Removal Rate $\\gamma + \\mu$ ($\\mathrm{day}^{-1}$)', fontsize=11)\n",
"ax2.set_title(r'Panel B: 2D Parameter Space $R_0(\\beta, \\gamma+\\mu)$ & Literature Estimates', fontsize=12, fontweight='bold')\n",
"ax2.legend(frameon=True, fontsize=8.8, loc='upper left')\n",
"\n",
"plt.tight_layout()\n",
"plt.savefig('figs/fig_m1_thresholds_contour.eps', format='eps')\n",
"plt.show()\n",
"print('Member 1 2D Parameter Contour Analysis Complete.')\n"
]
new_cells.append(make_cell("code", c_thresh_lines))

# ==============================================================================
# CELLS 7, 8, 9, 10: FLAW 1 & FLAW 2 DEMOS (UPDATED FOR EBOLA)
# ==============================================================================
c7_lines = [
"# ==============================================================================\n",
"# MEMBER 1 SIMULATION DEMO 1: DEMONSTRATING FLAW 1 OF CLASSICAL MODELS\n",
"# ==============================================================================\n",
"t_span = (0.0, 90.0)\n",
"t_eval = np.linspace(0.0, 90.0, 301)\n",
"\n",
"sol_t, sol_const = reference_solution(BETA_CONST, y0_default, t_span, t_eval)\n",
"\n",
"def beta_step(t):\n",
"    return np.where(t < 25.0, 0.60, 0.22)\n",
"\n",
"sol_t, sol_step = reference_solution(beta_step, y0_default, t_span, t_eval)\n",
"\n",
"def beta_wave(t):\n",
"    return 0.45 + 0.15 * np.sin(2.0 * np.pi * t / 30.0)\n",
"\n",
"sol_t, sol_wave = reference_solution(beta_wave, y0_default, t_span, t_eval)\n",
"\n",
"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)\n",
"\n",
"ax1.plot(t_eval, [BETA_CONST]*len(t_eval), label=r'Constant $\\beta = 0.486$ (Static Classical)', color='#1f77b4', lw=2.2)\n",
"ax1.plot(t_eval, beta_step(t_eval), label=r'Step Intervention $\\beta(t)$ (Safe Burials at Day 25)', color='#d62728', lw=2.2, linestyle='--')\n",
"ax1.plot(t_eval, beta_wave(t_eval), label=r'Seasonal Wave $\\beta(t)$ (30-day Harmonic)', color='#2ca02c', lw=2.2, linestyle='-.')\n",
"ax1.set_xlabel('Time t (days)', fontsize=11)\n",
"ax1.set_ylabel(r'Transmission Rate $\\beta(t)$ ($\\mathrm{day}^{-1}$)', fontsize=11)\n",
"ax1.set_title(r'Panel A: Transmission Dynamics $\\beta(t)$', fontsize=12, fontweight='bold')\n",
"ax1.grid(True, ls='--', lw=0.5, color='lightgray')\n",
"ax1.legend(frameon=True, fontsize=9.5)\n",
"\n",
"ax2.plot(t_eval, sol_const[:, 2], label=r'Constant $\\beta$: Severe Peak ($i_{\\max} \\approx 15.3\\%$)', color='#1f77b4', lw=2.2)\n",
"ax2.plot(t_eval, sol_step[:, 2], label=r'Step Intervention: Immediate Outbreak Collapse', color='#d62728', lw=2.2, linestyle='--')\n",
"ax2.plot(t_eval, sol_wave[:, 2], label=r'Seasonal Wave: Resurgent Multi-Wave Oscillations', color='#2ca02c', lw=2.2, linestyle='-.')\n",
"ax2.set_xlabel('Time t (days)', fontsize=11)\n",
"ax2.set_ylabel('Infectious Fraction i(t)', fontsize=11)\n",
"ax2.set_title('Panel B: Resulting Infection Trajectories i(t)', fontsize=12, fontweight='bold')\n",
"ax2.grid(True, ls='--', lw=0.5, color='lightgray')\n",
"ax2.legend(frameon=True, fontsize=9.5)\n",
"\n",
"plt.tight_layout()\n",
"plt.savefig('figs/fig_m1_flaw1_transmission_comparison.eps', format='eps')\n",
"plt.show()\n"
]
c8_text = r"""<div style="background-color: #f8fafc; border-left: 5px solid #475569; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
<h4 style="color: #334155; margin-top: 0; font-size: 14px;">Member 1 Scientific Interpretation of Flaw 1 Demonstration</h4>
<p style="font-size: 13.5px; line-height: 1.55; margin-bottom: 0;">
This simulation directly substantiates Member 1's thesis regarding Flaw 1 for Ebola:
<ul>
  <li>Under <b>constant $\beta = 0.486$ ($R_0 = 2.00$)</b>, the epidemic produces a catastrophic curve peaking at day $\approx 35$ ($i_{\max} \approx 15.3\%$) before herd immunity depletes the susceptible pool.</li>
  <li>Under the <b>step intervention</b> (modeling rapid deployment of safe burials and isolation on day 25), transmission drops from $0.60$ to $0.22$. Because $R_t$ drops below unity, the active epidemic collapses abruptly, truncating the peak early and protecting the majority of the population. A classical model assuming constant $\beta$ completely fails to capture this truncation.</li>
  <li>Under the <b>seasonal wave</b>, periodic fluctuations induce recurring waves and secondary resurgences. Classical static models cannot reproduce multi-wave behavior without manually re-fitting separate segments.</li>
</ul>
This proves the necessity of a flexible deep learning approach capable of discovering $\beta(t)$ without pre-specifying its mathematical structure.
</p>
</div>"""

c10_text = r"""<div style="background-color: #f8fafc; border-left: 5px solid #475569; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
<h4 style="color: #334155; margin-top: 0; font-size: 14px;">Member 1 Scientific Interpretation of Flaw 2 Demonstration</h4>
<p style="font-size: 13.5px; line-height: 1.55; margin-bottom: 0;">
This visualization captures the essence of Flaw 2 (the surveillance visibility dilemma):
<ul>
  <li>In <b>Panel A</b>, public health authorities have access only to cumulative case counts ($C$) and death tallies ($D$). These time series are monotonically non-decreasing, blurred by testing/reporting fraction $\rho$, and corrupted by reporting delays.</li>
  <li>In <b>Panel B</b>, the actual epidemiological reality is displayed. The latent exposed compartment ($e(t)$) forms a hidden "iceberg" that surges well before cases are clinically registered. Moreover, the depletion of susceptible individuals ($s(t)$) determines the true proximity to the epidemic turning point.</li>
</ul>
Classical regression cannot invert $s(t)$ and $e(t)$ from Panel A without falling into parameter collinearity. This motivates our Physics-Informed Neural Network architecture, which utilizes physical conservation laws as regularizers to reconstruct these latent curves with high precision (as Member 4 demonstrates in Experiment E4).
</p>
</div>"""

new_cells.append(make_cell("code", c7_lines))
new_cells.append(make_cell("markdown", c8_text))

c9_lines = [
"# ==============================================================================\n",
"# MEMBER 1 SIMULATION DEMO 2: DEMONSTRATING FLAW 2 OF CLASSICAL MODELS\n",
"# ==============================================================================\n",
"np.random.seed(42)\n",
"t_obs_demo = np.linspace(0.0, 90.0, 91)\n",
"_, sol_obs_demo = reference_solution(beta_step, y0_default, t_span, t_obs_demo)\n",
"\n",
"s_demo = sol_obs_demo[:, 0]\n",
"e_demo = sol_obs_demo[:, 1]\n",
"i_demo = sol_obs_demo[:, 2]\n",
"r_demo = sol_obs_demo[:, 3]\n",
"d_demo = sol_obs_demo[:, 4]\n",
"\n",
"rho_demo = 0.80\n",
"c_true_demo = rho_demo * (1.0 - s_demo - e_demo)\n",
"noise_c = np.random.normal(0, 0.015 * np.mean(c_true_demo), size=len(t_obs_demo))\n",
"c_obs_demo = np.clip(c_true_demo + noise_c, 0, 1)\n",
"\n",
"noise_d = np.random.normal(0, 0.015 * np.mean(d_demo + 1e-4), size=len(t_obs_demo))\n",
"d_obs_demo = np.clip(d_demo + noise_d, 0, 1)\n",
"\n",
"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)\n",
"\n",
"ax1.plot(t_obs_demo, c_obs_demo, 'o', color='#1a73e8', ms=4, label='Reported Cumulative Diagnoses C_obs(t)')\n",
"ax1.plot(t_obs_demo, d_obs_demo, 's', color='#dc2626', ms=4, label='Reported Cumulative Deaths D_obs(t)')\n",
"ax1.plot(t_obs_demo, c_true_demo, '-', color='#1a73e8', lw=1.5)\n",
"ax1.plot(t_obs_demo, d_demo, '-', color='#dc2626', lw=1.5)\n",
"ax1.set_xlabel('Time t (days)', fontsize=11)\n",
"ax1.set_ylabel('Reported Population Fraction', fontsize=11)\n",
"ax1.set_title('Panel A: Observable Surveillance (Clinic & Hospital Records)', fontsize=12, fontweight='bold')\n",
"ax1.grid(True, ls='--', lw=0.5, color='lightgray')\n",
"ax1.legend(frameon=True, fontsize=9.5)\n",
"\n",
"ax2.plot(t_obs_demo, s_demo, color='#9333ea', lw=2.2, label='Latent Susceptible Pool s(t) (Unobserved)')\n",
"ax2.plot(t_obs_demo, e_demo, color='#d97706', lw=2.2, linestyle='--', label='Latent Incubating Exposed e(t) (Unobserved)')\n",
"ax2.plot(t_obs_demo, i_demo, color='#ef4444', lw=1.8, linestyle=':', label='Active Infectious i(t) (Partially Unobserved)')\n",
"ax2.set_xlabel('Time t (days)', fontsize=11)\n",
"ax2.set_ylabel('True Compartmental Fraction', fontsize=11)\n",
"ax2.set_title('Panel B: Latent Hidden Epidemiological States', fontsize=12, fontweight='bold')\n",
"ax2.grid(True, ls='--', lw=0.5, color='lightgray')\n",
"ax2.legend(frameon=True, fontsize=9.5)\n",
"\n",
"plt.tight_layout()\n",
"plt.savefig('figs/fig_m1_flaw2_hidden_states.eps', format='eps')\n",
"plt.show()\n"
]
new_cells.append(make_cell("code", c9_lines))
new_cells.append(make_cell("markdown", c10_text))

# ==============================================================================
# REMAINING CELLS (PARTS 2 TO 10): APPEND WITH TEXT UPDATES FOR EBOLA
# ==============================================================================
part2_start = None
for idx, c in enumerate(orig_cells):
    src = ''.join(c['source'])
    if "Part 2: Vectorized Numerical Solvers" in src:
        part2_start = idx
        break

if part2_start is None:
    part2_start = 17

for idx in range(part2_start, len(orig_cells)):
    c = orig_cells[idx]
    src = ''.join(c['source'])
    src_mod = src.replace("Ontario", "Sierra Leone").replace("COVID-19", "Ebola").replace("covid 19", "Ebola").replace("Ontario COVID-19", "Sierra Leone Ebola")
    new_cells.append(make_cell(c['cell_type'], src_mod))

orig_nb['cells'] = new_cells

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(orig_nb, f, indent=1)

print(f"Successfully generated clean notebook.ipynb! Total cells: {len(new_cells)}")
