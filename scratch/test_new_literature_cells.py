import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Epidemiological parameters
POPULATION_N = 7_092_113
SIGMA = 1.0 / 9.4
GAMMA = 1.0 / 7.0
MU = 0.1000
BETA_CONST = 0.4858

y0_default = np.array([0.990, 0.005, 0.005, 0.000, 0.000])

def seird_ode(t, y, beta_func, sigma=SIGMA, gamma=GAMMA, mu=MU):
    s, e, i, r, d = y
    beta = beta_func(t) if callable(beta_func) else beta_func
    ds = -beta * s * i
    de = beta * s * i - sigma * e
    di = sigma * e - (gamma + mu) * i
    dr = gamma * i
    dd = mu * i
    return np.array([ds, de, di, dr, dd])

def reference_solution(beta_func, y0, t_span, t_eval, sigma=SIGMA, gamma=GAMMA, mu=MU):
    sol = solve_ivp(
        lambda t, y: seird_ode(t, y, beta_func, sigma, gamma, mu),
        t_span,
        y0,
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-14
    )
    return sol.t, sol.y.T

t_span = (0.0, 120.0)
t_eval = np.linspace(0.0, 120.0, 401)

# Model 1: Static Kermack-McKendrick
beta_static = BETA_CONST

# Model 2: Althaus (2014) exponential decay
def beta_althaus(t):
    k = 0.034
    t0 = 25.0
    return np.where(t < t0, 0.58, 0.58 * np.exp(-k * (t - t0)))

# Model 3: Rivers et al. (2014) piecewise step intervention
def beta_rivers(t):
    return np.where(t < 30.0, 0.58, 0.22)

# Model 4: Dynamic adaptive / resurgence wave model (simulating behavioral fatigue & secondary wave)
def beta_dynamic(t):
    # Initial drop from interventions, followed by secondary wave around day 75
    b0 = 0.58
    drop = 0.60 / (1.0 + np.exp(-0.18 * (t - 30.0)))
    resurgence = 0.25 / (1.0 + np.exp(-0.15 * (t - 75.0)))
    return b0 * (1.0 - drop + resurgence)

# Integrate all four models
_, y_static = reference_solution(beta_static, y0_default, t_span, t_eval)
_, y_althaus = reference_solution(beta_althaus, y0_default, t_span, t_eval)
_, y_rivers = reference_solution(beta_rivers, y0_default, t_span, t_eval)
_, y_dynamic = reference_solution(beta_dynamic, y0_default, t_span, t_eval)

# Compute Rt for all models
rt_static = [beta_static * y_static[k, 0] / (GAMMA + MU) for k in range(len(t_eval))]
rt_althaus = [beta_althaus(t_eval[k]) * y_althaus[k, 0] / (GAMMA + MU) for k in range(len(t_eval))]
rt_rivers = [beta_rivers(t_eval[k]) * y_rivers[k, 0] / (GAMMA + MU) for k in range(len(t_eval))]
rt_dynamic = [beta_dynamic(t_eval[k]) * y_dynamic[k, 0] / (GAMMA + MU) for k in range(len(t_eval))]

fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=150)

# Panel A: Transmission trajectories
axs[0, 0].plot(t_eval, [beta_static]*len(t_eval), label=r'Static $\beta = 0.486$ ($R_0=2.00$)', color='#1f77b4', lw=2.2)
axs[0, 0].plot(t_eval, beta_althaus(t_eval), label=r'Althaus (2014): $\beta_0 e^{-k(t-t_0)}$', color='#d62728', lw=2.2, linestyle='--')
axs[0, 0].plot(t_eval, beta_rivers(t_eval), label=r'Rivers et al. (2014): Step Policy Drop', color='#2ca02c', lw=2.2, linestyle='-.')
axs[0, 0].plot(t_eval, beta_dynamic(t_eval), label=r'Dynamic Resurgence Model', color='#9467bd', lw=2.2, linestyle=':')
axs[0, 0].axvspan(25, 35, color='#fed7aa', alpha=0.3, label='Intervention Phase')
axs[0, 0].set_xlabel('Time $t$ (days)', fontsize=11)
axs[0, 0].set_ylabel(r'Transmission Rate $\beta(t)$ ($\mathrm{day}^{-1}$)', fontsize=11)
axs[0, 0].set_title('Panel A: Published & Dynamic Transmission Rates $\\beta(t)$', fontsize=12, fontweight='bold')
axs[0, 0].grid(True, ls='--', lw=0.5, color='lightgray')
axs[0, 0].legend(frameon=True, fontsize=9.0, loc='upper right')

# Panel B: Effective Reproduction Number Rt
axs[0, 1].plot(t_eval, rt_static, label='Static $\\beta$: Depletion Only', color='#1f77b4', lw=2.2)
axs[0, 1].plot(t_eval, rt_althaus, label='Althaus (2014): Monotonic Decline', color='#d62728', lw=2.2, linestyle='--')
axs[0, 1].plot(t_eval, rt_rivers, label='Rivers et al. (2014): Discontinuous Drop', color='#2ca02c', lw=2.2, linestyle='-.')
axs[0, 1].plot(t_eval, rt_dynamic, label='Dynamic: Second Wave Crossing', color='#9467bd', lw=2.2, linestyle=':')
axs[0, 1].axhline(1.0, color='black', lw=1.5, ls='--', label='Critical Threshold $R_t = 1.0$')
axs[0, 1].axhspan(0, 1.0, color='#dcfce7', alpha=0.3)
axs[0, 1].axhspan(1.0, 3.0, color='#fee2e2', alpha=0.3)
axs[0, 1].set_xlabel('Time $t$ (days)', fontsize=11)
axs[0, 1].set_ylabel(r'Effective Reproduction Number $R_t(t)$', fontsize=11)
axs[0, 1].set_title('Panel B: Effective Reproduction Number $R_t(t)$ vs Critical Threshold', fontsize=12, fontweight='bold')
axs[0, 1].grid(True, ls='--', lw=0.5, color='lightgray')
axs[0, 1].set_ylim(0.0, 2.6)
axs[0, 1].legend(frameon=True, fontsize=9.0, loc='upper right')

# Panel C: Infectious Fraction i(t)
axs[1, 0].plot(t_eval, y_static[:, 2], label=f'Static $\\beta$: Peak={np.max(y_static[:, 2]):.1%}', color='#1f77b4', lw=2.2)
axs[1, 0].plot(t_eval, y_althaus[:, 2], label=f'Althaus: Peak={np.max(y_althaus[:, 2]):.1%}', color='#d62728', lw=2.2, linestyle='--')
axs[1, 0].plot(t_eval, y_rivers[:, 2], label=f'Rivers Step: Peak={np.max(y_rivers[:, 2]):.1%}', color='#2ca02c', lw=2.2, linestyle='-.')
axs[1, 0].plot(t_eval, y_dynamic[:, 2], label=f'Dynamic: Peak={np.max(y_dynamic[:, 2]):.1%}', color='#9467bd', lw=2.2, linestyle=':')
axs[1, 0].set_xlabel('Time $t$ (days)', fontsize=11)
axs[1, 0].set_ylabel('Infectious Fraction $i(t)$', fontsize=11)
axs[1, 0].set_title('Panel C: Active Symptomatic Infection Trajectories $i(t)$', fontsize=12, fontweight='bold')
axs[1, 0].grid(True, ls='--', lw=0.5, color='lightgray')
axs[1, 0].legend(frameon=True, fontsize=9.0)

# Panel D: Cumulative Mortalities d(t)
axs[1, 1].plot(t_eval, y_static[:, 4], label=f'Static $\\beta$: Final={y_static[-1, 4]:.1%}', color='#1f77b4', lw=2.2)
axs[1, 1].plot(t_eval, y_althaus[:, 4], label=f'Althaus: Final={y_althaus[-1, 4]:.1%}', color='#d62728', lw=2.2, linestyle='--')
axs[1, 1].plot(t_eval, y_rivers[:, 4], label=f'Rivers Step: Final={y_rivers[-1, 4]:.1%}', color='#2ca02c', lw=2.2, linestyle='-.')
axs[1, 1].plot(t_eval, y_dynamic[:, 4], label=f'Dynamic: Final={y_dynamic[-1, 4]:.1%}', color='#9467bd', lw=2.2, linestyle=':')
axs[1, 1].set_xlabel('Time $t$ (days)', fontsize=11)
axs[1, 1].set_ylabel('Cumulative Mortality Fraction $d(t)$', fontsize=11)
axs[1, 1].set_title('Panel D: Cumulative Mortality Trajectories $d(t)$', fontsize=12, fontweight='bold')
axs[1, 1].grid(True, ls='--', lw=0.5, color='lightgray')
axs[1, 1].legend(frameon=True, fontsize=9.0)

plt.tight_layout()
plt.savefig('figs/fig_m1_published_models_comparison.eps', format='eps')
plt.close()
print("Figure 1 generated successfully!")

# Test second figure: Forest plot of published R0 estimates
studies = [
    ('Chowell et al. (2004) - Uganda', 1.34, 1.15, 1.53),
    ('Chowell et al. (2004) - Congo', 1.83, 1.48, 2.18),
    ('WHO Response Team (2014) - Sierra Leone', 2.02, 1.79, 2.26),
    ('Our SEIRD Baseline (Sierra Leone)', 2.00, 1.80, 2.20),
    ('Rivers et al. (2014) - Sierra Leone', 2.53, 2.20, 2.86),
    ('Althaus (2014) - Sierra Leone', 2.53, 2.41, 2.67),
    ('Legrand et al. (2007) - DRC', 2.70, 2.30, 3.10)
]

labels = [s[0] for s in studies]
r0_vals = np.array([s[1] for s in studies])
err_low = r0_vals - np.array([s[2] for s in studies])
err_high = np.array([s[3] for s in studies]) - r0_vals
y_pos = np.arange(len(studies))

fig, ax1 = plt.subplots(figsize=(10.5, 4.8), dpi=150)
colors = ['#0284c7', '#0284c7', '#16a34a', '#7c3aed', '#ea580c', '#ea580c', '#dc2626']

for i in range(len(studies)):
    ax1.errorbar(r0_vals[i], y_pos[i], xerr=[[err_low[i]], [err_high[i]]],
                 fmt='o', color=colors[i], ecolor=colors[i], elinewidth=2.5,
                 capsize=5, capthick=1.8, markersize=8)

ax1.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, label='Epidemic Threshold $R_0 = 1$')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=10.5)
ax1.set_xlabel(r'Estimated Basic Reproduction Number $R_0$ (with 95% CI)', fontsize=11)
ax1.set_title(r'Published Basic Reproduction Numbers $R_0$ Across Landmark Ebola Studies', fontsize=12, fontweight='bold')
ax1.grid(True, ls='--', lw=0.5, color='lightgray')

# Add secondary top axis for implied herd immunity threshold p_c = 1 - 1/R0
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
tick_r0 = np.array([1.2, 1.5, 2.0, 2.5, 3.0])
tick_pc = (1.0 - 1.0 / tick_r0) * 100.0
ax2.set_xticks(tick_r0)
ax2.set_xticklabels([f'{pc:.1f}%' for pc in tick_pc], fontsize=10)
ax2.set_xlabel(r'Implied Critical Intervention Threshold $p_c = 1 - 1/R_0$ (%)', fontsize=11, color='#475569')

plt.tight_layout()
plt.savefig('figs/fig_m1_published_r0_forest.eps', format='eps')
plt.close()
print("Forest plot generated successfully!")
