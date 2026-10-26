"""
seird_pinn.py
-------------
Module 4 (M4): physics-informed neural network (PINN) for the SEIRD model with a
time-varying transmission rate beta(t).

Model, in counts (N = population size):
    dS/dt = -beta(t) S I / N
    dE/dt =  beta(t) S I / N - sigma E
    dI/dt =  sigma E - (gamma + mu) I
    dR/dt =  gamma I
    dD/dt =  mu I

Surveillance reports cumulative cases and cumulative deaths, so the PINN uses the
cumulative number of symptomatic cases C = I + R + D in place of R:
    dC/dt = sigma E,    R = C - I - D,    S = N - E - C.
Writing S this way makes S + E + I + R + D = N hold exactly, for any network
weights, and makes both observed series (C and D) states of the network.

All states are measured in units of a characteristic count K (for real data, the
largest cumulative case count in the training window) and time is rescaled to
tau = (t - t0) / T in [0, 1], so every quantity the optimiser sees is O(1).

Unknowns learned together:
    * the states e(tau), i(tau), c(tau), d(tau)          (state network)
    * beta(tau) > 0                                     (beta network, Softplus)
    * mu, through the case-fatality ratio CFR = mu / (gamma + mu)
    * the initial conditions e0, i0, c0, d0
sigma and gamma are fixed from the literature (project conventions), which keeps
the problem identifiable.
"""

import math
import random
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

SIGMA = 1.0 / 9.4   # incubation rate (day^-1), WHO Ebola Response Team 2014
GAMMA = 1.0 / 7.0   # recovery rate (day^-1)
MU = 0.1            # disease-induced death rate (day^-1), project convention
DTYPE = torch.float64


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _inv_softplus(y):
    y = float(y)
    return y + math.log(-math.expm1(-y))


class FourierFeatures(nn.Module):
    """x in [-1, 1] -> (x, sin(pi k x), cos(pi k x)) for k = 1..n_freq.
    Plain tanh networks learn slowly varying functions first ("spectral bias");
    these extra inputs make oscillations over the window as easy to represent as trends."""

    def __init__(self, n_freq):
        super().__init__()
        self.register_buffer('k', math.pi * torch.arange(1, n_freq + 1, dtype=DTYPE).view(1, -1))

    def forward(self, x):
        return torch.cat([x, torch.sin(x * self.k), torch.cos(x * self.k)], dim=1)


def _mlp(n_out, hidden, layers, n_freq=0):
    n_in = 1 + 2 * n_freq
    mods = [FourierFeatures(n_freq)] if n_freq > 0 else []
    mods += [nn.Linear(n_in, hidden), nn.Tanh()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.Tanh()]
    mods.append(nn.Linear(hidden, n_out))
    net = nn.Sequential(*mods)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return net


# =============================================================================
# Model
# =============================================================================
class SEIRDPINN(nn.Module):
    """
    State network g(tau) with 4 outputs, x = (e, i, c, d) in units of K:
        e = e0 exp(tau g_1),  i = i0 exp(tau g_2),  c = c0 exp(tau g_3),  d = d0 + tau g_4.
        The factor tau makes x(0) = x0 exactly (hard initial condition), and the
        exponentials keep e, i, c positive while they grow over orders of magnitude.
    Beta network:   beta(tau) = Softplus(h(tau) + b),  b chosen so beta starts at beta_init.
    CFR:            CFR = lo + (hi - lo) sigmoid(theta),  mu = gamma * CFR / (1 - CFR).
        Keeping CFR in a plausible range (default 20-80%; published Ebola values
        are 40-70%) blocks a spurious solution in which mu -> 0 while I grows
        without bound so that mu * I still matches the deaths.
    """

    def __init__(self, N, K, T, x0, beta_init=0.3, cfr_init=None, learn_ic=True,
                 learn_mu=True, beta_const=None, hidden=48, layers=3,
                 beta_hidden=32, beta_layers=3, sigma=SIGMA, gamma=GAMMA,
                 relative_residuals=True, cfr_bounds=(0.2, 0.8), n_freq=0):
        super().__init__()
        self.N, self.K, self.T = float(N), float(K), float(T)
        self.sigma, self.gamma = float(sigma), float(gamma)
        self.relative_residuals = relative_residuals
        self.cfr_lo, self.cfr_hi = cfr_bounds
        self.state_net = _mlp(4, hidden, layers, n_freq)
        self.beta_const = beta_const
        if beta_const is None:
            self.beta_net = _mlp(1, beta_hidden, beta_layers, n_freq)
            self.beta_bias = nn.Parameter(torch.tensor(_inv_softplus(beta_init), dtype=DTYPE))
        cfr0 = MU / (GAMMA + MU) if cfr_init is None else cfr_init
        q = (min(max(cfr0, self.cfr_lo + 1e-3), self.cfr_hi - 1e-3) - self.cfr_lo) / (self.cfr_hi - self.cfr_lo)
        theta = torch.tensor(math.log(q / (1.0 - q)), dtype=DTYPE)
        self.cfr_logit = nn.Parameter(theta) if learn_mu else nn.Parameter(theta, requires_grad=False)
        # initial conditions stored through softplus so they stay positive
        raw = torch.tensor([_inv_softplus(max(v, 1e-8)) for v in x0], dtype=DTYPE)
        self.x0_raw = nn.Parameter(raw) if learn_ic else nn.Parameter(raw, requires_grad=False)
        self.to(DTYPE)

    # ---- parameters -------------------------------------------------------
    @property
    def cfr(self):
        return self.cfr_lo + (self.cfr_hi - self.cfr_lo) * torch.sigmoid(self.cfr_logit)

    @property
    def mu(self):
        cfr = self.cfr
        return self.gamma * cfr / (1.0 - cfr)

    @property
    def x0(self):
        return nn.functional.softplus(self.x0_raw)

    # ---- networks ---------------------------------------------------------
    @staticmethod
    def _states_from(net, x0, tau):
        g = tau * net(2.0 * tau - 1.0)
        eic = x0[:3] * torch.exp(g[:, :3])
        d = x0[3] + g[:, 3:4]
        return torch.cat([eic, d], dim=1)

    def states(self, tau):
        """(e, i, c, d) in units of K, shape (n, 4)."""
        return self._states_from(self.state_net, self.x0, tau)


    def beta(self, tau):
        if self.beta_const is not None:
            return torch.full_like(tau, float(self.beta_const))
        return nn.functional.softplus(self.beta_net(2.0 * tau - 1.0) + self.beta_bias)

    def s_frac(self, x):
        """Susceptible fraction S/N = 1 - (E + C)/N."""
        return 1.0 - self.K * (x[:, 0:1] + x[:, 2:3]) / self.N

    # ---- physics residual -------------------------------------------------
    def residuals(self, tau):
        """
        Relative residuals of the scaled ODEs, each of shape (n, 1):
            r_e = [de/dtau - T f_e] / e,   r_i = [di/dtau - T f_i] / i,
            r_c = [dc/dtau - T f_c] / c,   r_d = [dd/dtau - T f_d] / c.
        Dividing by the size of the state makes a residual of 0.01 mean "1% error
        per training window" whether the epidemic is at 10 cases or 10,000, so the
        early, small phase of an outbreak is not ignored. Deaths are scaled by c
        because d starts at zero and d ~ CFR * c.
        """
        tau = tau.clone().requires_grad_(True)
        x = self.states(tau)
        dx = [torch.autograd.grad(x[:, k].sum(), tau, create_graph=True)[0] for k in range(4)]
        e, i, c = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        beta = self.beta(tau)
        s = self.s_frac(x)
        mu = self.mu
        T = self.T
        r_e = dx[0] - T * (beta * s * i - self.sigma * e)
        r_i = dx[1] - T * (self.sigma * e - (self.gamma + mu) * i)
        r_c = dx[2] - T * (self.sigma * e)
        r_d = dx[3] - T * (mu * i)
        if self.relative_residuals:
            r_e, r_i, r_c, r_d = r_e / e, r_i / i, r_c / c, r_d / c
        return r_e, r_i, r_c, r_d, beta, tau

    # ---- convenience ------------------------------------------------------
    @torch.no_grad()
    def predict(self, t, t0):
        """Counts S, E, I, R, D, C and beta at times t (days)."""
        tau = torch.tensor((np.asarray(t, float) - t0) / self.T, dtype=DTYPE).view(-1, 1)
        x = self.states(tau).numpy() * self.K
        e, i, c, d = x.T
        out = {
            'E': e, 'I': i, 'C': c, 'D': d,
            'R': c - i - d,
            'S': self.N - e - c,
            'beta': self.beta(tau).numpy().ravel(),
        }
        return out


# =============================================================================
# Training
# =============================================================================
@dataclass
class TrainConfig:
    adam_iters: int = 6000
    adam_lr: float = 2e-3
    lbfgs_iters: int = 1500
    n_colloc: int = 200
    w_data: float = 1.0
    w_deaths: float = 1.0
    w_ode: float = 1.0
    w_smooth: float = 1e-4
    seed: int = 42
    verbose: bool = False


def train_pinn(model, tau_obs, y_obs, cfg, mask=None):
    """
    Fit the PINN.

    tau_obs : (n,) rescaled observation times
    y_obs   : (n, 2) observed cumulative cases and deaths in units of K
              (NaN = missing; a column of NaN means that series is not observed)
    Loss    : w_data * [ MSE(cases) + w_deaths * MSE(deaths) ]
              + w_ode * mean of squared ODE residuals over collocation points
              + w_smooth * mean (d beta / d tau)^2
    Optimiser: Adam with random collocation points, then L-BFGS on a fixed grid.
    """
    set_seeds(cfg.seed)
    tau_obs = torch.tensor(np.asarray(tau_obs, float), dtype=DTYPE).view(-1, 1)
    y_obs = torch.tensor(np.asarray(y_obs, float), dtype=DTYPE)
    has_c = ~torch.isnan(y_obs[:, 0])
    has_d = ~torch.isnan(y_obs[:, 1])
    y_c = y_obs[has_c, 0]
    y_d = y_obs[has_d, 1]
    history = []

    def loss_fn(tau_c):
        x = model.states(tau_obs)
        l_data = torch.zeros((), dtype=DTYPE)
        if has_c.any():
            l_data = l_data + torch.mean((x[has_c, 2] - y_c) ** 2)
        if has_d.any():
            l_data = l_data + cfg.w_deaths * torch.mean((x[has_d, 3] - y_d) ** 2)
        r_e, r_i, r_c, r_d, beta, tau_g = model.residuals(tau_c)
        l_ode = torch.mean(r_e ** 2 + r_i ** 2 + r_c ** 2 + r_d ** 2)
        if model.beta_const is None and cfg.w_smooth > 0:
            db = torch.autograd.grad(beta.sum(), tau_g, create_graph=True)[0]
            l_smooth = torch.mean(db ** 2)
        else:
            l_smooth = torch.zeros((), dtype=DTYPE)
        total = cfg.w_data * l_data + cfg.w_ode * l_ode + cfg.w_smooth * l_smooth
        return total, l_data, l_ode, l_smooth

    start = time.time()
    params = [p for p in model.parameters() if p.requires_grad]
    adam = torch.optim.Adam(params, lr=cfg.adam_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=max(cfg.adam_iters, 1), eta_min=cfg.adam_lr * 0.05)
    for it in range(cfg.adam_iters):
        adam.zero_grad()
        tau_c = torch.rand((cfg.n_colloc, 1), dtype=DTYPE)
        total, l_data, l_ode, l_smooth = loss_fn(tau_c)
        total.backward()
        adam.step()
        sched.step()
        if it % 200 == 0 or it == cfg.adam_iters - 1:
            history.append(('adam', it, total.item(), l_data.item(), l_ode.item(), l_smooth.item()))
            if cfg.verbose and it % 2000 == 0:
                print(f"    adam {it:5d} | loss {total.item():.3e} | data {l_data.item():.3e} | ode {l_ode.item():.3e}")

    if cfg.lbfgs_iters > 0:
        tau_fixed = torch.linspace(0.0, 1.0, cfg.n_colloc, dtype=DTYPE).view(-1, 1)
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.lbfgs_iters, history_size=50,
                                  tolerance_grad=1e-12, tolerance_change=1e-14,
                                  line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            total, _, _, _ = loss_fn(tau_fixed)
            total.backward()
            return total

        lbfgs.step(closure)
        total, l_data, l_ode, l_smooth = loss_fn(tau_fixed)
        history.append(('lbfgs', cfg.lbfgs_iters, total.item(), l_data.item(), l_ode.item(), l_smooth.item()))
        if cfg.verbose:
            print(f"    lbfgs done | loss {total.item():.3e} | data {l_data.item():.3e} | ode {l_ode.item():.3e}")

    return {'history': history, 'seconds': time.time() - start}


# =============================================================================
# Reference ODE solutions (used for data generation and for forecasting)
# =============================================================================
def seird_counts_rhs(t, y, beta_fn, mu, N, sigma=SIGMA, gamma=GAMMA):
    """Right-hand side in counts for (S, E, I, R, D, C)."""
    S, E, I, R, D, C = y
    b = beta_fn(t)
    inf = b * S * I / N
    return [-inf, inf - sigma * E, sigma * E - (gamma + mu) * I, gamma * I, mu * I, sigma * E]


def simulate_counts(beta_fn, y0, t_eval, mu=MU, N=1.0, sigma=SIGMA, gamma=GAMMA):
    """
    DOP853 solution (rtol 1e-12, atol 1e-14) of the SEIRD model with the
    cumulative-case state C appended. y0 = (S, E, I, R, D, C).
    """
    sol = solve_ivp(lambda t, y: seird_counts_rhs(t, y, beta_fn, mu, N, sigma, gamma),
                    (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method='DOP853',
                    rtol=1e-12, atol=1e-14 * max(N, 1.0))
    S, E, I, R, D, C = sol.y
    return {'S': S, 'E': E, 'I': I, 'R': R, 'D': D, 'C': C}


def forecast(model, t0, t_future, beta_window_days=14):
    """
    Forecast beyond the training window.

    Starts from the PINN state at the end of training (tau = 1), keeps beta fixed at
    its mean over the last `beta_window_days` days of training, uses the learned mu,
    and integrates the SEIRD ODE with DOP853.
    """
    t_end = t0 + model.T
    tt = np.linspace(t_end - beta_window_days, t_end, 50)
    p_end = model.predict([t_end], t0)
    beta_f = float(np.mean(model.predict(tt, t0)['beta']))
    mu = model.mu.item()
    y0 = [p_end['S'][0], p_end['E'][0], p_end['I'][0], p_end['R'][0], p_end['D'][0], p_end['C'][0]]
    t_eval = np.concatenate([[t_end], np.asarray(t_future, float)])
    sim = simulate_counts(lambda t: beta_f, y0, t_eval, mu=mu, N=model.N)
    return {k: v[1:] for k, v in sim.items()} | {'beta_forecast': beta_f, 'mu': mu}


# =============================================================================
# Helpers for fitting observed series
# =============================================================================
def fit_observed(t_obs, cases, deaths, N, cfg, t0=None, T=None, K=None, n_seeds=1,
                 model_kwargs=None):
    """
    Build and train PINN(s) on observed cumulative cases/deaths (counts).
    Returns a list of (model, info) for each seed, plus the scaling used.
    """
    t_obs = np.asarray(t_obs, float)
    cases = np.asarray(cases, float)
    deaths = np.asarray(deaths, float)
    t0 = t_obs[0] if t0 is None else t0
    T = (t_obs[-1] - t0) if T is None else T
    K = np.nanmax(cases) if K is None else K
    # initial-condition guesses: E0 and I0 from the first two weeks of incidence
    early = t_obs <= t0 + 14
    # floor relative to the size of the data, so this works for counts and for fractions
    inc = max((cases[early][-1] - cases[0]) / max(t_obs[early][-1] - t0, 1.0), 1e-4 * np.nanmax(cases))
    x0 = [inc / SIGMA / K, inc / (GAMMA + MU) / K, cases[0] / K,
          (deaths[0] if not np.isnan(deaths[0]) else 0.0) / K]
    # start the case-fatality ratio at the observed deaths/cases ratio
    okc, okd = ~np.isnan(cases), ~np.isnan(deaths)
    cfr0 = float(np.clip(deaths[okd][-1] / cases[okc][-1], 0.05, 0.9)) if okd.any() else None
    kw = dict(beta_init=GAMMA + MU, learn_ic=True, learn_mu=True, cfr_init=cfr0)
    kw.update(model_kwargs or {})
    tau = (t_obs - t0) / T
    y = np.column_stack([cases / K, deaths / K])
    fits = []
    for k in range(n_seeds):
        c = TrainConfig(**{**asdict(cfg), 'seed': cfg.seed + k})
        set_seeds(c.seed)
        model = SEIRDPINN(N=N, K=K, T=T, x0=x0, **kw)
        info = train_pinn(model, tau, y, c)
        fits.append((model, info))
    return fits, {'t0': t0, 'T': T, 'K': K}
