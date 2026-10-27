"""
pinn_experiments.py
-------------------
Module 4 (M4) experiments for the SEIRD-PINN in seird_pinn.py.

    E2   forward problem: PINN vs the DOP853 reference solution (3 seeds)
    E3   inverse problem: recover beta(t) (intervention step) and mu from noisy
         cumulative cases and deaths only, 1% and 5% noise, 5 seeds
    E4   hidden states: s(t), e(t), i(t) recovered in the same E3 runs
    E6   ablation: physics weight w_ode and the design choices (relative residuals,
         Fourier features, L-BFGS stage) on the E3 step problem, 2 seeds
    E7   real data, WHO 2014-15 Sierra Leone (laboratory-confirmed): rolling
         4-week forecasts from 8 test origins, PINN ensemble (3 seeds) vs the
         constant-beta and piecewise-beta least-squares baselines
    E5   re-run of Module 5's held-out test (25 May - 21 Sep 2014, days 90-119)
         with this PINN; Module 5's two baselines are kept exactly as written

Every run trains with one CPU thread; independent runs execute in parallel
processes. Results are saved to results/m4_*.json and figures to figs/ (.eps and
.png). run_* functions reuse saved results unless force=True, so the notebook can
show the outputs without retraining; `python pinn_experiments.py all` reruns all.

Settings were chosen before looking at any test data:
    * synthetic settings on a separate tuning seed (seed 1), never on seeds 42-46
    * real-data settings (smoothing weight, Fourier features on/off, beta window)
      on validation forecasts only: the three E7 validation origins (24 Oct,
      21 Nov, 19 Dec 2014) and an E5 fit on days 0-59 scored on days 60-89.
      The rule was the lowest mean relative error of the case forecast.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

import seird_pinn as sp

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(ROOT, 'figs')
RESULTS_DIR = os.path.join(ROOT, 'results')
N_SL = 7_092_113
MAX_WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))   # one job per physical core

# ---- settings (see module docstring for how they were chosen) --------------
SYN_T = 120.0
SYN_Y0 = [0.990, 0.005, 0.005, 0.0, 0.0, 0.005]          # S, E, I, R, D, C (fractions)
SYN_BETA_CONST = 0.4858                                  # R0 = 2.00, project convention
SYN_STEP_DAY = 30.0
PROFILES = {
    'step': lambda t: np.where(np.asarray(t) < SYN_STEP_DAY, 0.55, 0.25),
}
PROFILE_LABELS = {'step': r'Step: $\beta=0.55 \to 0.25$ at day 30'}
SEEDS = [42, 43, 44, 45, 46]

CFG_FORWARD = sp.TrainConfig(adam_iters=2000, lbfgs_iters=4000, w_data=0.0)
CFG_INVERSE = sp.TrainConfig(adam_iters=2000, lbfgs_iters=6000, w_smooth=1e-4)
INVERSE_MODEL_KW = {'n_freq': 4}       # Fourier features (see E6)
CFG_REAL = sp.TrainConfig(adam_iters=2000, lbfgs_iters=2500, w_smooth=1e-3)
REAL_MODEL_KW = {'n_freq': 4}
REAL_BETA_WINDOW = 14        # days of beta averaged for the forecast
E7_WINDOW = 56               # training window (days) for each rolling origin
E7_HORIZON = 28              # forecast horizon (days)
E7_VAL_ORIGINS = [56, 84, 112]                       # 24 Oct, 21 Nov, 19 Dec 2014
E7_TEST_ORIGINS = [140, 168, 196, 224, 252, 280, 308, 336]   # 16 Jan - 31 Jul 2015
E7_START = '2014-08-29'


def _ensure_dirs():
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _save_json(name, obj):
    _ensure_dirs()
    with open(os.path.join(RESULTS_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, default=_to_jsonable)


def _load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None


def _to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _pool_map(fn, jobs):
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as ex:
        return list(ex.map(fn, jobs))


def _worker_init():
    import torch
    torch.set_num_threads(1)


def _rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def _cfg(base, **kw):
    return sp.TrainConfig(**{**asdict(base), **kw})


def save_figure(fig, name):
    _ensure_dirs()
    fig.savefig(os.path.join(FIGS_DIR, name + '.eps'), format='eps', bbox_inches='tight')
    fig.savefig(os.path.join(FIGS_DIR, name + '.png'), format='png', dpi=200, bbox_inches='tight')


# =============================================================================
# Synthetic data
# =============================================================================
def synthetic_truth(profile, t=None):
    t = np.linspace(0.0, SYN_T, int(SYN_T) + 1) if t is None else t
    beta_fn = (lambda tt: SYN_BETA_CONST) if profile == 'constant' else PROFILES[profile]
    ref = sp.simulate_counts(beta_fn, SYN_Y0, t, mu=sp.MU, N=1.0)
    ref['beta'] = np.full_like(t, SYN_BETA_CONST) if profile == 'constant' else PROFILES[profile](t)
    ref['t'] = t
    return ref


def synthetic_observations(ref, noise, seed):
    """Cumulative cases and deaths with independent multiplicative Gaussian noise."""
    rng = np.random.default_rng(seed)
    c = ref['C'] * (1.0 + noise * rng.standard_normal(len(ref['t'])))
    d = ref['D'] * (1.0 + noise * rng.standard_normal(len(ref['t'])))
    return c, d


# =============================================================================
# E2: forward problem
# =============================================================================
def _job_forward(seed, relative_residuals=True):
    _worker_init()
    ref = synthetic_truth('constant')
    t = ref['t']
    x0 = [SYN_Y0[1], SYN_Y0[2], SYN_Y0[5], SYN_Y0[4]]
    sp.set_seeds(seed)
    model = sp.SEIRDPINN(N=1.0, K=1.0, T=SYN_T, x0=x0, beta_const=SYN_BETA_CONST,
                         learn_ic=False, learn_mu=False, relative_residuals=relative_residuals)
    info = sp.train_pinn(model, np.array([0.0]), np.array([[np.nan, np.nan]]), _cfg(CFG_FORWARD, seed=seed))
    p = model.predict(t, 0.0)
    return {'seed': seed, 'seconds': info['seconds'],
            'rel_l2': {k: _rel_l2(p[k], ref[k]) for k in 'SEIRD'},
            'max_abs': {k: float(np.max(np.abs(p[k] - ref[k]))) for k in 'SEIRD'},
            'conservation': float(np.max(np.abs(p['S'] + p['E'] + p['I'] + p['R'] + p['D'] - 1.0))),
            'pred': {k: p[k] for k in 'SEIRD'}}


def run_e2(force=False):
    res = None if force else _load_json('m4_e2_forward.json')
    if res is None:
        runs = _pool_map(_job_forward, SEEDS[:3])
        ref = synthetic_truth('constant')
        res = {'runs': runs, 'truth': {k: ref[k] for k in ['t', 'S', 'E', 'I', 'R', 'D']},
               'config': asdict(CFG_FORWARD)}
        _save_json('m4_e2_forward.json', res)
    return res


def _job_forward_absolute(seed):
    return _job_forward(seed, relative_residuals=False)


def run_e2_absolute(force=False):
    """E6 companion: the E2 forward problem with plain (absolute) residuals."""
    res = None if force else _load_json('m4_e6_forward_absolute.json')
    if res is None:
        runs = _pool_map(_job_forward_absolute, SEEDS[:3])
        for r in runs:
            r.pop('pred')
        res = {'runs': runs, 'config': asdict(CFG_FORWARD)}
        _save_json('m4_e6_forward_absolute.json', res)
    return res


# =============================================================================
# E3 / E4: inverse problem from cases and deaths only
# =============================================================================
def _job_inverse(job):
    _worker_init()
    profile, noise, seed, overrides, model_kw = job
    ref = synthetic_truth(profile)
    t = ref['t']
    c_obs, d_obs = synthetic_observations(ref, noise, seed)
    cfg = _cfg(CFG_INVERSE, seed=seed, **overrides)
    fits, sc = sp.fit_observed(t, c_obs, d_obs, N=1.0, cfg=cfg, t0=0.0, T=SYN_T,
                               K=float(np.max(c_obs)), model_kwargs={**INVERSE_MODEL_KW, **model_kw})
    model, info = fits[0]
    p = model.predict(t, 0.0)
    b_true, b_pred = ref['beta'], p['beta']
    away = np.abs(t - SYN_STEP_DAY) > 5 if profile == 'step' else np.ones_like(t, bool)
    return {
        'profile': profile, 'noise': noise, 'seed': seed, 'overrides': overrides, 'model_kw': model_kw,
        'seconds': info['seconds'],
        'beta_rmse': float(np.sqrt(np.mean((b_pred - b_true) ** 2))),
        'beta_rel_l2': _rel_l2(b_pred, b_true),
        'beta_rmse_away_from_jump': float(np.sqrt(np.mean((b_pred - b_true)[away] ** 2))),
        'mu_hat': model.mu.item(), 'mu_rel_err': abs(model.mu.item() - sp.MU) / sp.MU,
        'state_rel_l2': {k: _rel_l2(p[k], ref[k]) for k in 'SEIRD'},
        'state_rmse': {k: float(np.sqrt(np.mean((p[k] - ref[k]) ** 2))) for k in 'SEIRD'},
        'beta_pred': b_pred, 'S_pred': p['S'], 'E_pred': p['E'], 'I_pred': p['I'],
        'c_obs': c_obs, 'd_obs': d_obs, 'final_loss': info['history'][-1][2],
    }


def run_e3_e4(force=False):
    res = None if force else _load_json('m4_e3_e4_inverse.json')
    if res is None:
        jobs = [(p, n, s, {}, {}) for p in PROFILES for n in (0.01, 0.05) for s in SEEDS]
        runs = _pool_map(_job_inverse, jobs)
        truth = {p: {k: synthetic_truth(p)[k] for k in ['t', 'beta', 'S', 'E', 'I', 'R', 'D', 'C']} for p in PROFILES}
        res = {'runs': runs, 'truth': truth, 'config': asdict(CFG_INVERSE)}
        _save_json('m4_e3_e4_inverse.json', res)
    return res


def summarise_inverse(res):
    rows = []
    df = pd.DataFrame(res['runs'])
    df = df[df['noise'] > 0]
    for (p, n), g in df.groupby(['profile', 'noise'], sort=False):
        row = {'Profile': p, 'Noise': f"{int(round(n * 100))}%", 'Seeds': len(g)}
        for col, lab in [('beta_rmse', 'beta RMSE'), ('beta_rel_l2', 'beta rel. L2'),
                         ('beta_rmse_away_from_jump', 'beta RMSE (|t-30|>5)'), ('mu_rel_err', 'mu rel. err')]:
            row[lab] = f"{g[col].mean():.4f} ± {g[col].std(ddof=1):.4f}"
        for k in 'SEI':
            v = g['state_rel_l2'].apply(lambda d: d[k])
            row[f'{k.lower()}(t) rel. L2'] = f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# E6: ablations on the E3 step problem (5% noise)
# =============================================================================
ABLATION_SEEDS = [42, 43]


def run_ablation(force=False):
    res = None if force else _load_json('m4_e6_ablation.json')
    if res is None:
        # the full method's runs are taken from E3 (same problem, same seeds)
        jobs = [('step', 0.05, s, {'w_ode': 10.0}, {}) for s in ABLATION_SEEDS]
        jobs += [('step', 0.05, s, {}, {'relative_residuals': False}) for s in ABLATION_SEEDS]
        jobs += [('step', 0.05, s, {'lbfgs_iters': 0, 'adam_iters': 8000}, {}) for s in ABLATION_SEEDS]
        jobs += [('step', 0.05, s, {}, {'n_freq': 0}) for s in ABLATION_SEEDS]
        e3 = run_e3_e4()
        defaults = [dict(r) for r in e3['runs']
                    if r['profile'] == 'step' and r['noise'] == 0.05 and r['seed'] in ABLATION_SEEDS]
        runs = defaults + _pool_map(_job_inverse, jobs)
        for r in runs:
            for k in ['beta_pred', 'S_pred', 'E_pred', 'I_pred', 'c_obs', 'd_obs']:
                r.pop(k, None)
        res = {'runs': runs}
        _save_json('m4_e6_ablation.json', res)
    return res


def summarise_ablation(res):
    df = pd.DataFrame(res['runs'])

    def label(r):
        if r['model_kw'].get('relative_residuals') is False:
            return 'Absolute residuals (no rescaling)'
        if r['model_kw'].get('n_freq') == 0:
            return 'No Fourier features'
        o = r['overrides']
        if o.get('lbfgs_iters') == 0:
            return 'Adam only (8000 its, no L-BFGS)'
        if 'w_ode' in o:
            return f"Physics weight w_ode = {o['w_ode']:g} (instead of 1)"
        return 'Full method'

    df['setting'] = df.apply(label, axis=1)
    df['e_rel'] = df['state_rel_l2'].apply(lambda d: d['E'])
    df['s_rel'] = df['state_rel_l2'].apply(lambda d: d['S'])
    out = df.groupby('setting', sort=False).agg(
        beta_rmse=('beta_rmse', 'mean'), beta_rmse_sd=('beta_rmse', 'std'),
        e_rel_l2=('e_rel', 'mean'), s_rel_l2=('s_rel', 'mean'),
        mu_rel_err=('mu_rel_err', 'mean'), seconds=('seconds', 'mean')).reset_index()
    return out


# =============================================================================
# Real data: WHO 2014-15 Sierra Leone, rolling forecasts (E7)
# =============================================================================
def load_who_series():
    from data_pipeline_who import load_sierra_leone_series
    df = load_sierra_leone_series()
    df['cases'] = df['cumulative_cases'].where(df['cases_used'])
    df['deaths'] = df['cumulative_deaths'].where(df['deaths_used'])
    return df


def _ls_fit(t_tr, c_tr, d_tr, t0, t_fut, n_seg):
    """
    Least-squares SEIRD with n_seg equal-length constant-beta segments.
    Unknowns: beta_1..beta_n, mu, E0, I0 (C0 and D0 are the first reports).
    Residuals are scaled by the mean of each series; a small penalty on
    differences between neighbouring betas, as in Module 5's piecewise baseline.
    """
    T = t_tr[-1] - t0
    edges = t0 + T * np.arange(1, n_seg) / n_seg
    okc, okd = ~np.isnan(c_tr), ~np.isnan(d_tr)
    c0 = c_tr[okc][0]
    d0 = d_tr[okd][0] if okd.any() else 0.0
    inc = max((c_tr[okc][min(3, okc.sum() - 1)] - c0) / max(t_tr[okc][min(3, okc.sum() - 1)] - t0, 1.0), 1.0)

    def beta_fn(p):
        b = p[:n_seg]
        return lambda t: b[np.searchsorted(edges, t, side='right')]

    def simulate(p, t_eval):
        mu, e0, i0 = p[n_seg], p[n_seg + 1], p[n_seg + 2]
        y0 = [N_SL - e0 - c0, e0, i0, c0 - i0 - d0, d0, c0]
        grid = np.union1d([t0], t_eval)
        sim = sp.simulate_counts(beta_fn(p), y0, grid, mu=mu, N=N_SL)
        idx = np.searchsorted(grid, t_eval)
        return {k: v[idx] for k, v in sim.items()}

    def res(p):
        sim = simulate(p, t_tr)
        r = [(sim['C'][okc] - c_tr[okc]) / np.mean(c_tr[okc])]
        if okd.any():
            r.append((sim['D'][okd] - d_tr[okd]) / np.mean(d_tr[okd]))
        if n_seg > 1:
            r.append(0.5 * np.diff(p[:n_seg]))
        return np.concatenate(r)

    # same CFR range as the PINN (20-80%), and I0 <= C0 - D0 so that R0 >= 0
    mu_lo, mu_hi = sp.GAMMA * 0.2 / 0.8, sp.GAMMA * 0.8 / 0.2
    i_hi = max(c0 - d0, 1.0)
    p0 = np.r_[np.full(n_seg, sp.GAMMA + sp.MU), sp.MU, inc / sp.SIGMA, min(inc / (sp.GAMMA + sp.MU), 0.9 * i_hi)]
    lo = np.r_[np.full(n_seg, 1e-3), mu_lo, 0.0, 0.0]
    hi = np.r_[np.full(n_seg, 2.0), mu_hi, 50 * inc / sp.SIGMA + 10, i_hi]
    fit = least_squares(res, p0, bounds=(lo, hi))
    sim = simulate(fit.x, np.asarray(t_fut, float))
    return {'C': sim['C'], 'D': sim['D'], 'params': fit.x.tolist()}


def _job_real_pinn(job):
    _worker_init()
    t_tr, c_tr, d_tr, t0, T, t_fut, seed, overrides, beta_window = job[:9]
    model_kw = job[9] if len(job) > 9 else REAL_MODEL_KW
    cfg = _cfg(CFG_REAL, seed=seed, **overrides)
    fits, sc = sp.fit_observed(t_tr, c_tr, d_tr, N=N_SL, cfg=cfg, t0=t0, T=T, model_kwargs=model_kw)
    model, info = fits[0]
    # beta_window may be a list: one trained model, several forecast assumptions
    windows = list(beta_window) if isinstance(beta_window, (list, tuple)) else [beta_window]
    fcs = {w: sp.forecast(model, t0, t_fut, beta_window_days=w) for w in windows}
    fc = fcs[windows[0]]
    t_grid = np.linspace(t0, t0 + T, 120)
    p = model.predict(t_grid, t0)
    return {'seed': seed, 'seconds': info['seconds'], 'C': fc['C'], 'D': fc['D'],
            'beta_forecast': fc['beta_forecast'], 'mu': fc['mu'], 'cfr': model.cfr.item(),
            'by_window': {str(w): {'C': f['C'], 'D': f['D'], 'beta_forecast': f['beta_forecast']} for w, f in fcs.items()},
            't_grid': t_grid, 'beta_grid': p['beta'], 'C_grid': p['C'], 'D_grid': p['D'],
            'E_grid': p['E'], 'I_grid': p['I']}


def _job_baselines(job):
    t_tr, c_tr, d_tr, t0, t_fut = job
    out = {}
    for name, n_seg in [('constant_beta', 1), ('piecewise_beta', 2)]:
        out[name] = _ls_fit(t_tr, c_tr, d_tr, t0, t_fut, n_seg)
    return out


def _forecast_metrics(obs, pred, base):
    """Errors over the horizon (report dates only). `base` = last training value."""
    ok = ~np.isnan(obs)
    if not ok.any():
        return None
    o, p = obs[ok], pred[ok]
    new_obs = o[-1] - base
    new_pred = p[-1] - base
    return {'mae': float(np.mean(np.abs(p - o))), 'rmse': float(np.sqrt(np.mean((p - o) ** 2))),
            'rel_err_pct': float(np.mean(np.abs(p - o)) / np.mean(o) * 100),
            'new_obs': float(new_obs), 'new_pred': float(new_pred),
            'new_abs_err': float(abs(new_pred - new_obs))}


def run_e7(origins=None, force=False, n_seeds=3, overrides=None, beta_window=REAL_BETA_WINDOW,
           tag='m4_e7_rolling.json'):
    """Rolling-origin forecasts on the WHO 2014-15 Sierra Leone series."""
    res = None if force else _load_json(tag)
    if res is not None:
        return res
    df = load_who_series()
    origins = E7_TEST_ORIGINS if origins is None else origins
    overrides = overrides or {}
    pinn_jobs, base_jobs, meta = [], [], []
    for o in origins:
        tr = df[(df['day'] > o - E7_WINDOW) & (df['day'] <= o)]
        te = df[(df['day'] > o) & (df['day'] <= o + E7_HORIZON)]
        t_tr = tr['day'].values.astype(float)
        t0 = float(o - E7_WINDOW)
        # start the window at the first report inside it
        t0 = float(t_tr[0])
        T = float(o) - t0
        t_fut = te['day'].values.astype(float)
        c_tr, d_tr = tr['cases'].values, tr['deaths'].values
        meta.append({'origin': o, 'date': df.loc[df['day'] <= o, 'date'].iloc[-1], 't0': t0, 'T': T,
                     't_train': t_tr, 'c_train': c_tr, 'd_train': d_tr, 't_test': t_fut,
                     'c_test': te['cases'].values, 'd_test': te['deaths'].values,
                     'split': 'validation' if o in E7_VAL_ORIGINS else 'test'})
        for s in SEEDS[:n_seeds]:
            pinn_jobs.append((t_tr, c_tr, d_tr, t0, T, t_fut, s, overrides, beta_window))
        base_jobs.append((t_tr, c_tr, d_tr, t0, t_fut))
    pinn_runs = _pool_map(_job_real_pinn, pinn_jobs)
    base_runs = _pool_map(_job_baselines, base_jobs)
    out = []
    for k, m in enumerate(meta):
        runs = pinn_runs[k * n_seeds:(k + 1) * n_seeds]
        ens_c = np.median([r['C'] for r in runs], axis=0)
        ens_d = np.median([r['D'] for r in runs], axis=0)
        c_last = m['c_train'][~np.isnan(m['c_train'])][-1]
        d_last = m['d_train'][~np.isnan(m['d_train'])][-1]
        preds = {'pinn': {'C': ens_c, 'D': ens_d}, **base_runs[k]}
        metrics = {name: {'cases': _forecast_metrics(m['c_test'], pr['C'], c_last),
                          'deaths': _forecast_metrics(m['d_test'], pr['D'], d_last)}
                   for name, pr in preds.items()}
        out.append({**m, 'pinn_runs': runs, 'preds': {n: {'C': p['C'], 'D': p['D']} for n, p in preds.items()},
                    'metrics': metrics})
    res = {'origins': out, 'config': asdict(CFG_REAL), 'overrides': overrides, 'beta_window': beta_window,
           'window': E7_WINDOW, 'horizon': E7_HORIZON, 'n_seeds': n_seeds}
    _save_json(tag, res)
    return res


def summarise_e7(res, split='test'):
    rows = []
    origins = [o for o in res['origins'] if o['split'] == split]
    for name, lab in [('pinn', f"PINN ensemble ({res['n_seeds']} seeds)"), ('piecewise_beta', 'Piecewise-beta LS'),
                      ('constant_beta', 'Constant-beta LS')]:
        row = {'Method': lab}
        for tgt in ['cases', 'deaths']:
            ms = [o['metrics'][name][tgt] for o in origins if o['metrics'][name][tgt] is not None]
            row[f'{tgt} MAE'] = np.mean([m['mae'] for m in ms])
            row[f'{tgt} RMSE'] = np.mean([m['rmse'] for m in ms])
            row[f'{tgt} rel. err (%)'] = np.mean([m['rel_err_pct'] for m in ms])
            row[f'{tgt} 4-wk new, abs err'] = np.mean([m['new_abs_err'] for m in ms])
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# E5: Module 5's held-out test (cmrivers data, 25 May - 21 Sep 2014)
# =============================================================================
E5_TRAIN_END = 89
E5_VAL_END = 59


def load_e5_reports():
    """Real report dates (no interpolation) from Module 5's raw file, days 0-119."""
    raw = pd.read_csv(os.path.join(ROOT, 'data', 'country_timeseries_raw.csv'))
    raw['Date'] = pd.to_datetime(raw['Date'], format='%m/%d/%Y')
    raw = raw.dropna(subset=['Cases_SierraLeone', 'Deaths_SierraLeone']).sort_values('Date')
    raw['day'] = (raw['Date'] - pd.Timestamp('2014-05-25')).dt.days
    raw = raw[(raw['day'] >= 0) & (raw['day'] <= 119)].copy()
    keep = np.ones(len(raw), bool)
    for col in ['Cases_SierraLeone', 'Deaths_SierraLeone']:
        v = raw[col].values.astype(float)
        keep &= v <= np.minimum.accumulate(v[::-1])[::-1]
    return raw[keep].reset_index(drop=True)


def m5_baselines(train_end=E5_TRAIN_END, fit_to='reports'):
    """
    Module 5's constant-beta and piecewise-beta baselines (same models, starting
    values, residual scaling and solver as Module 5).
    fit_to='reports': fitted to the real WHO report dates up to train_end (main E5 result).
    fit_to='daily':   fitted to Module 5's interpolated daily series, as Module 5 did.
    """
    daily = pd.read_csv(os.path.join(ROOT, 'data', 'outbreak_data.csv'))
    N = float(N_SL)
    t_all = daily['day'].values.astype(float)
    if fit_to == 'reports':
        tr = load_e5_reports()
        tr = tr[tr['day'] <= train_end]
        t_tr = tr['day'].values.astype(float)
        c_tr, d_tr = tr['Cases_SierraLeone'].values.astype(float), tr['Deaths_SierraLeone'].values.astype(float)
    else:
        tr = daily[daily['day'] <= train_end]
        t_tr = tr['day'].values.astype(float)
        c_tr, d_tr = tr['cumulative_cases'].values.astype(float), tr['cumulative_deaths'].values.astype(float)
    i0, e0, d0 = max(11.0 / N, 1e-6), max(16.0 / N, 1e-6), 5.0 / N
    y0 = [1.0 - (e0 + i0 + d0), e0, i0, 0.0, d0]

    def solve(beta, t_eval):
        # Module 5's solver: 5 fractions, DOP853, rtol 1e-10, atol 1e-12, starting at day 0
        f = beta if callable(beta) else (lambda t: beta)

        def rhs(t, y):
            s, e, i, r, d = y
            b = f(t)
            return [-b * s * i, b * s * i - sp.SIGMA * e, sp.SIGMA * e - (sp.GAMMA + sp.MU) * i,
                    sp.GAMMA * i, sp.MU * i]

        grid = np.union1d([0.0], t_eval)
        sol = solve_ivp(rhs, (grid[0], grid[-1]), y0, t_eval=grid, method='DOP853', rtol=1e-10, atol=1e-12)
        y = sol.y.T[np.searchsorted(grid, t_eval)]
        return (1.0 - y[:, 0] - y[:, 1]) * N, y[:, 4] * N

    def pw(p):
        return lambda t: p[0] if t < 30.0 else (p[1] if t < 60.0 else p[2])

    def res_c(p):
        c, d = solve(p[0], t_tr)
        return np.concatenate([(c - c_tr) / (np.mean(c_tr) + 1.0), (d - d_tr) / (np.mean(d_tr) + 1.0)])

    def res_pw(p):
        c, d = solve(pw(p), t_tr)
        return np.concatenate([(c - c_tr) / (np.mean(c_tr) + 1.0), (d - d_tr) / (np.mean(d_tr) + 1.0),
                               0.5 * np.array([p[1] - p[0], p[2] - p[1]])])

    bc = least_squares(res_c, x0=[0.25], bounds=(0.01, 1.5)).x
    bp = least_squares(res_pw, x0=[0.30, 0.25, 0.20], bounds=(0.01, 1.5)).x
    c_c, d_c = solve(bc[0], t_all)
    c_p, d_p = solve(pw(bp), t_all)
    return {'t': t_all, 'constant_beta': {'C': c_c, 'D': d_c, 'beta': bc.tolist()},
            'piecewise_beta': {'C': c_p, 'D': d_p, 'beta': bp.tolist()}}


def _e5_metrics(y, yh):
    r = yh - y
    return {'MAE': float(np.mean(np.abs(r))), 'RMSE': float(np.sqrt(np.mean(r ** 2))),
            'RelErr(%)': float(np.mean(np.abs(r)) / np.mean(y) * 100),
            'MAPE(%)': float(np.mean(np.abs(r / y)) * 100), 'MaxErr': float(np.max(np.abs(r))),
            'R2': float(1 - np.sum(r ** 2) / np.sum((y - np.mean(y)) ** 2))}


def run_e5(force=False, train_end=E5_TRAIN_END, n_seeds=5, overrides=None, beta_window=REAL_BETA_WINDOW,
           tag='m4_e5_heldout.json'):
    """
    PINN trained on the real report dates up to `train_end`, forecasting the next 30 days.
    Main result ('table'): every method fitted to the real report dates and scored on the
    real report dates in the test window. 'table_interpolated' keeps Module 5's original
    protocol (baselines fitted to, and all methods scored on, the interpolated daily series).
    A saved PINN run is reused unless force=True; the tables are always recomputed.
    """
    cached = None if force else _load_json(tag)
    rep = load_e5_reports()
    daily = pd.read_csv(os.path.join(ROOT, 'data', 'outbreak_data.csv'))
    tr = rep[rep['day'] <= train_end]
    t_tr = tr['day'].values.astype(float)
    test = daily[(daily['day'] > train_end) & (daily['day'] <= train_end + 30)]
    t_fut = test['day'].values.astype(float)
    if cached is not None:
        runs = cached['pinn_runs']
    else:
        jobs = [(t_tr, tr['Cases_SierraLeone'].values.astype(float), tr['Deaths_SierraLeone'].values.astype(float),
                 0.0, float(train_end), t_fut, s, overrides or {}, beta_window) for s in SEEDS[:n_seeds]]
        runs = _pool_map(_job_real_pinn, jobs)
    ens_c = np.median([np.asarray(r['C']) for r in runs], axis=0)
    ens_d = np.median([np.asarray(r['D']) for r in runs], axis=0)

    rep_te = rep[(rep['day'] > train_end) & (rep['day'] <= train_end + 30)]
    idx = np.searchsorted(t_fut, rep_te['day'].values.astype(float))
    obs = {'Cases': rep_te['Cases_SierraLeone'].values.astype(float), 'Deaths': rep_te['Deaths_SierraLeone'].values.astype(float)}
    base = m5_baselines(train_end, fit_to='reports')
    mask = (base['t'] > train_end) & (base['t'] <= train_end + 30)
    preds = {'Constant-beta LS (Module 5)': base['constant_beta'], 'Piecewise-beta LS (Module 5)': base['piecewise_beta']}
    table = {}
    for name, b in preds.items():
        for tgt, key in [('Cases', 'C'), ('Deaths', 'D')]:
            table[(name, tgt)] = _e5_metrics(obs[tgt], np.asarray(b[key])[mask][idx])
    for tgt, ens in [('Cases', ens_c), ('Deaths', ens_d)]:
        table[('PINN ensemble (Module 4)', tgt)] = _e5_metrics(obs[tgt], ens[idx])
    for r in runs:
        for tgt, key in [('Cases', 'C'), ('Deaths', 'D')]:
            table[(f'PINN seed {r["seed"]}', tgt)] = _e5_metrics(obs[tgt], np.asarray(r[key])[idx])

    base_d = m5_baselines(train_end, fit_to='daily')
    y_daily = {'Cases': test['cumulative_cases'].values.astype(float), 'Deaths': test['cumulative_deaths'].values.astype(float)}
    table_interp = {}
    for name, b in [('Constant-beta LS (Module 5)', base_d['constant_beta']), ('Piecewise-beta LS (Module 5)', base_d['piecewise_beta'])]:
        for tgt, key in [('Cases', 'C'), ('Deaths', 'D')]:
            table_interp[(name, tgt)] = _e5_metrics(y_daily[tgt], np.asarray(b[key])[mask])
    for tgt, ens in [('Cases', ens_c), ('Deaths', ens_d)]:
        table_interp[('PINN ensemble (Module 4)', tgt)] = _e5_metrics(y_daily[tgt], ens)

    res = {'train_end': train_end, 'protocol': 'fitted to and scored on real report dates',
           'n_test_reports': int(len(rep_te)), 'n_train_reports': int(len(tr)),
           'table': [{'Model': m, 'Target': t, **v} for (m, t), v in table.items()],
           'table_interpolated': [{'Model': m, 'Target': t, **v} for (m, t), v in table_interp.items()],
           'pinn_runs': runs, 't_test': t_fut, 'ens_C': ens_c, 'ens_D': ens_d,
           'baselines': {k: base[k] for k in ['t', 'constant_beta', 'piecewise_beta']},
           'baselines_interpolated_fit': {k: base_d[k] for k in ['constant_beta', 'piecewise_beta']},
           'train_reports': {'day': t_tr, 'C': tr['Cases_SierraLeone'].values, 'D': tr['Deaths_SierraLeone'].values},
           'test_reports': {'day': rep_te['day'].values, 'C': rep_te['Cases_SierraLeone'].values,
                            'D': rep_te['Deaths_SierraLeone'].values},
           'config': asdict(CFG_REAL), 'overrides': overrides or {}, 'beta_window': beta_window}
    _save_json(tag, res)
    return res

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    force = '--force' in sys.argv
    t0 = time.time()
    if what in ('e2', 'all'):
        run_e2(force=force)
    if what in ('e3', 'all'):
        run_e3_e4(force=force)
    if what in ('e6', 'all'):
        run_ablation(force=force)
        run_e2_absolute(force=force)
    if what in ('e7', 'all'):
        run_e7(force=force)
    if what in ('e5', 'all'):
        run_e5(force=force)
    print(f"[pinn_experiments] done '{what}' in {time.time() - t0:.0f} s")
