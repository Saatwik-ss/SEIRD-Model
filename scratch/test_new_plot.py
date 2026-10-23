import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

GAMMA = 1.0 / 7.0
MU = 0.1000
SIGMA = 1.0 / 9.4
BETA_CONST = 2.0 * (GAMMA + MU)
y0_default = np.array([0.990, 0.005, 0.005, 0.0, 0.0])

def seird_rhs(t, y, beta, sigma=SIGMA, gamma=GAMMA, mu=MU):
    s, e, i, r, d = y
    ds = -beta * s * i
    de = beta * s * i - sigma * e
    di = sigma * e - (gamma + mu) * i
    dr = gamma * i
    dd = mu * i
    return [ds, de, di, dr, dd]

def reference_solution(beta_val, y0, t_span, t_eval):
    sol = solve_ivp(seird_rhs, t_span, y0, args=(beta_val,), t_eval=t_eval, method="DOP853", rtol=1e-10, atol=1e-12)
    return sol.t, sol.y.T

r0_phase_list = [1.2, 1.5, 2.0, 2.5, 3.0]
colors_phase = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=150)
s_grid = np.linspace(0.01, 0.999, 500)

s0 = y0_default[0]
w0 = y0_default[1] + y0_default[2]  # Total active: e0 + i0

for r0_val, col in zip(r0_phase_list, colors_phase):
    beta_val = r0_val * (GAMMA + MU)
    w_theory = w0 + s0 - s_grid + (1.0 / r0_val) * np.log(s_grid / s0)
    w_theory = np.maximum(w_theory, 0)
    sol_t, sol_y = reference_solution(beta_val, y0_default, (0.0, 250.0), np.linspace(0, 250, 600))
    
    # Plot total active infections (e + i)
    w_sol = sol_y[:, 1] + sol_y[:, 2]
    ax1.plot(sol_y[:, 0], w_sol, label=f'$R_0 = {r0_val:.1f}$', color=col, lw=2.0)
    
    # Theoretical peak point
    s_star = 1.0 / r0_val
    w_max_theory = w0 + s0 - (1.0 / r0_val) * (1.0 + np.log(r0_val * s0))
    ax1.plot(s_star, w_max_theory, 'o', color=col, ms=6, markeredgecolor='black', zorder=5)

ax1.axvline(0.5, color='gray', linestyle=':', lw=1.2, label=r'Nullcline $s^* = 0.5$ ($R_0=2.0$)')
ax1.set_xlabel('Susceptible Fraction $s$', fontsize=11)
ax1.set_ylabel('Total Active Infected Fraction $(e + i)$', fontsize=11)
ax1.set_title('Panel A: Phase-Space Trajectories $(e+i \text{ vs } s)$ & Nullclines', fontsize=12, fontweight='bold')
ax1.grid(True, ls='--', lw=0.5, color='lightgray')
ax1.legend(frameon=True, fontsize=9.0)

# Panel B: Exact verification
r0_dense = np.linspace(1.05, 4.0, 100)
wmax_dense_theory = w0 + s0 - (1.0 / r0_dense) * (1.0 + np.log(r0_dense * s0))
r0_pts = [1.2, 1.5, 2.0, 2.5, 3.0, 3.5]
ode_peaks_w = []
ode_peaks_i = []
theory_peaks = []

for r0_v in r0_pts:
    b_v = r0_v * (GAMMA + MU)
    _, sol_pts = reference_solution(b_v, y0_default, (0.0, 300.0), np.linspace(0, 300, 2000))
    ode_w_pk = np.max(sol_pts[:, 1] + sol_pts[:, 2])
    ode_i_pk = np.max(sol_pts[:, 2])
    th_pk = w0 + s0 - (1.0 / r0_v) * (1.0 + np.log(r0_v * s0))
    ode_peaks_w.append(ode_w_pk)
    ode_peaks_i.append(ode_i_pk)
    theory_peaks.append(th_pk)

ax2.plot(r0_dense, wmax_dense_theory, label=r'Theoretical $(e+i)_{\max} = w_0 + s_0 - \frac{1}{R_0}(1 + \ln(R_0 s_0))$', color='#6d28d9', lw=2.2)
ax2.plot(r0_pts, ode_peaks_w, 's', color='#dc2626', ms=7, label=r'DOP853 Numerical Peak $(e+i)$')
ax2.plot(r0_pts, ode_peaks_i, '^', color='#2563eb', ms=6, linestyle='--', lw=1.2, label=r'DOP853 Infectious Peak $i$ only')
ax2.set_xlabel(r'Basic Reproduction Number $R_0$', fontsize=11)
ax2.set_ylabel('Peak Active Fraction', fontsize=11)
ax2.set_title(r'Panel B: Exact Verification of Peak Formula', fontsize=12, fontweight='bold')
ax2.grid(True, ls='--', lw=0.5, color='lightgray')
ax2.legend(frameon=True, fontsize=9.0)

plt.tight_layout()
plt.savefig('figs/fig_m1_phase_space_nullclines.eps', format='eps')
print("Figure saved successfully!")

print(f"{'R0':>5} | {'Theoretical s*':>14} | {'Theory (e+i)_max':>16} | {'ODE (e+i)_max':>14} | {'Rel Error':>10}")
print('-' * 70)
for r0_v, th_p, od_p in zip(r0_pts, theory_peaks, ode_peaks_w):
    rel_err = abs(th_p - od_p) / th_p
    print(f'{r0_v:5.1f} | {1.0/r0_v:14.4f} | {th_p:16.4f} | {od_p:14.4f} | {rel_err:10.2e}')
