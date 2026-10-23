import numpy as np
from scipy.integrate import solve_ivp

GAMMA = 1.0 / 7.0
MU = 0.1000
SIGMA = 1.0 / 9.4
y0_default = np.array([0.990, 0.005, 0.005, 0.0, 0.0])

def seird_rhs(t, y, beta, sigma=SIGMA, gamma=GAMMA, mu=MU):
    s, e, i, r, d = y
    ds = -beta * s * i
    de = beta * s * i - sigma * e
    di = sigma * e - (gamma + mu) * i
    dr = gamma * i
    dd = mu * i
    return [ds, de, di, dr, dd]

r0_pts = [1.2, 1.5, 2.0, 2.5, 3.0, 3.5]
for r0 in r0_pts:
    b = r0 * (GAMMA + MU)
    sol = solve_ivp(seird_rhs, (0, 300), y0_default, args=(b,), dense_output=True, t_eval=np.linspace(0, 300, 2000))
    s_sol = sol.y[0]
    i_sol = sol.y[2]
    idx_max = np.argmax(i_sol)
    ode_s_at_peak = s_sol[idx_max]
    ode_peak = i_sol[idx_max]
    
    s_star_sir = 1.0 / r0
    s0, i0 = y0_default[0], y0_default[2]
    theory_peak_sir = i0 + s0 - (1.0 / r0) * (1.0 + np.log(r0 * s0))
    rel_err = abs(theory_peak_sir - ode_peak) / theory_peak_sir
    print(f"R0={r0:.1f} | s*={s_star_sir:.4f}, s_actual={ode_s_at_peak:.4f} | i_theory={theory_peak_sir:.4f}, i_actual={ode_peak:.4f} | err={rel_err*100:.2f}%")
