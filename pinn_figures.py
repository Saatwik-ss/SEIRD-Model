"""
pinn_figures.py
---------------
Figures for the Module 4 PINN experiments (pinn_experiments.py). Every figure is
saved as .eps (no transparency, per the project conventions) and .png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pinn_experiments as px

COLORS = {'S': '#1f77b4', 'E': '#ff7f0e', 'I': '#d62728', 'R': '#2ca02c', 'D': '#6b6b6b'}
METHOD_STYLE = {
    'pinn': ('PINN ensemble', '#7b2d8e', '-'),
    'piecewise_beta': (r'Piecewise-$\beta$ LS', '#2a9d3a', '--'),
    'constant_beta': (r'Constant-$\beta$ LS', '#1f5fbf', ':'),
}
LIGHT = '#d9c6e0'   # solid light purple for seed bands (EPS has no alpha)


def fig_e2(res):
    t = np.array(res['truth']['t'])
    run = res['runs'][0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for k in 'SEIRD':
        ax[0].plot(t, res['truth'][k], color=COLORS[k], lw=2.2, label=f'{k.lower()}(t) DOP853')
        ax[0].plot(t[::6], np.array(run['pred'][k])[::6], 'o', color=COLORS[k], ms=4, mfc='white',
                   label=f'{k.lower()}(t) PINN')
        err = np.abs(np.array(run['pred'][k]) - np.array(res['truth'][k]))
        ax[1].semilogy(t, np.maximum(err, 1e-12), color=COLORS[k], lw=1.6, label=f'|error| {k.lower()}(t)')
    ax[0].set(xlabel='Time (days)', ylabel='Fraction of population',
              title=r'E2: forward PINN vs DOP853 ($\beta=0.4858$, $R_0=2$)')
    ax[1].set(xlabel='Time (days)', ylabel='Absolute error', title='E2: pointwise error (seed 42)')
    ax[0].legend(ncol=2, fontsize=8)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(True, ls=':', color='#bbbbbb')
    fig.tight_layout()
    px.save_figure(fig, 'fig_e2_forward_pinn')
    return fig


def fig_e3(res):
    fig, a = plt.subplots(1, 1, figsize=(8, 4.6))
    runs = pd.DataFrame(res['runs'])
    for prof in px.PROFILES:
        t = np.array(res['truth'][prof]['t'])
        for noise, col, ls in [(0.01, '#1f5fbf', '--'), (0.05, '#c0392b', '-.')]:
            preds = np.array([np.array(r) for r in runs[(runs.profile == prof) & (runs.noise == noise)]['beta_pred']])
            lo, hi = preds.min(axis=0), preds.max(axis=0)
            if noise == 0.05:
                a.fill_between(t, lo, hi, color='#f3c6c0', lw=0, label='5% noise: range over 5 seeds')
            a.plot(t, preds.mean(axis=0), ls, color=col, lw=2, label=f'{int(noise * 100)}% noise: mean of 5 seeds')
        a.plot(t, res['truth'][prof]['beta'], 'k-', lw=2.2, label=r'true $\beta(t)$')
        a.set(xlabel='Time (days)', ylabel=r'$\beta(t)$ (day$^{-1}$)', title='E3: ' + px.PROFILE_LABELS[prof])
        a.grid(True, ls=':', color='#bbbbbb')
        a.legend(fontsize=8)
    fig.tight_layout()
    px.save_figure(fig, 'fig_e3_inverse_beta')
    return fig


def fig_e4(res, profile='step'):
    runs = pd.DataFrame(res['runs'])
    truth = res['truth'][profile]
    t = np.array(truth['t'])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, k, lab in zip(ax, ['S', 'E', 'I'], ['Susceptible $s(t)$', 'Exposed $e(t)$', 'Infectious $i(t)$']):
        a.plot(t, truth[k], 'k-', lw=2.2, label='true (never observed)')
        for noise, col, ls in [(0.01, '#1f5fbf', '--'), (0.05, '#c0392b', '-.')]:
            preds = np.array([np.array(r) for r in runs[(runs.profile == profile) & (runs.noise == noise)][f'{k}_pred']])
            a.plot(t, preds.mean(axis=0), ls, color=col, lw=2, label=f'PINN, {int(noise * 100)}% noise (mean of 5)')
        a.set(xlabel='Time (days)', ylabel='Fraction of population', title=f'E4: hidden {lab}')
        a.grid(True, ls=':', color='#bbbbbb')
        a.legend(fontsize=8)
    fig.suptitle('Hidden states recovered from noisy cumulative cases and deaths only (step-change scenario)', fontsize=11)
    fig.tight_layout()
    px.save_figure(fig, 'fig_e4_hidden_states')
    return fig


def fig_ablation(summary):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    y = np.arange(len(summary))
    ax[0].barh(y, summary['beta_rmse'], xerr=summary['beta_rmse_sd'], color='#7b2d8e', edgecolor='black')
    ax[1].barh(y, summary['e_rel_l2'], color='#ff7f0e', edgecolor='black')
    for a in ax:
        a.set_yticks(y)
        a.set_yticklabels(summary['setting'], fontsize=9)
        a.invert_yaxis()
        a.grid(True, ls=':', color='#bbbbbb', axis='x')
        a.set_xscale('log')
    ax[0].set(xlabel=r'$\beta(t)$ RMSE (day$^{-1}$), mean of 2 seeds', title='E6: transmission-rate error')
    ax[1].set(xlabel='hidden $e(t)$ relative $L_2$ error', title='E6: hidden-state error')
    fig.suptitle('Ablation on the E3 step problem (5% noise)', fontsize=11)
    fig.tight_layout()
    px.save_figure(fig, 'fig_e6_ablations')
    return fig


def fig_e7(res, who):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    a0 = fig.add_subplot(gs[0, :])
    a1 = fig.add_subplot(gs[1, 0])
    a2 = fig.add_subplot(gs[1, 1])
    who_c = who[who['cases_used']]
    a0.plot(who_c['day'], who_c['cumulative_cases'], 'k.', ms=4, label='WHO confirmed cases (report dates)')
    for k, o in enumerate(res['origins']):
        t_f = np.array(o['t_test'])
        for name in ['pinn', 'piecewise_beta', 'constant_beta']:
            lab, col, ls = METHOD_STYLE[name]
            a0.plot(t_f, o['preds'][name]['C'], ls, color=col, lw=1.8, label=lab if k == 0 else None)
        a0.axvline(o['origin'], color='#cccccc', lw=0.8, zorder=0)
        # learned beta / R_t from the seed ensemble
        tg = np.array(o['pinn_runs'][0]['t_grid'])
        b = np.median([r['beta_grid'] for r in o['pinn_runs']], axis=0)
        a1.plot(tg, b / (px.sp.GAMMA + np.median([r['mu'] for r in o['pinn_runs']])), color='#7b2d8e', lw=1.5)
    a0.set(xlabel='Days since 29 Aug 2014', ylabel='Cumulative confirmed cases',
           title='E7: rolling 4-week forecasts from 8 test origins (grey lines), Sierra Leone 2014-15')
    a0.legend(fontsize=9, loc='lower right')
    a1.axhline(1.0, color='k', lw=1, ls='--')
    a1.set(xlabel='Days since 29 Aug 2014', ylabel=r'$R_t = \beta(t)\,s(t)/(\gamma+\mu)$',
           title=r'E7: PINN reproduction number in each 8-week training window')
    summ = px.summarise_e7(res, 'test')
    names = ['pinn', 'piecewise_beta', 'constant_beta']
    x = np.arange(len(names))
    a2.bar(x - 0.2, summ['cases MAE'], 0.4, color='#7b2d8e', edgecolor='black', label='cases MAE')
    a2.bar(x + 0.2, summ['deaths MAE'], 0.4, color='#bbbbbb', edgecolor='black', label='deaths MAE')
    a2.set_xticks(x)
    a2.set_xticklabels([METHOD_STYLE[n][0] for n in names], fontsize=9)
    n_test = sum(o['split'] == 'test' for o in res['origins'])
    a2.set_yscale('log')
    a2.set(ylabel='MAE over 4-week horizon (people, log scale)', title=f'E7: mean error over {n_test} test origins')
    a2.legend(fontsize=9)
    for a in [a0, a1, a2]:
        a.grid(True, ls=':', color='#bbbbbb')
    fig.tight_layout()
    px.save_figure(fig, 'fig_e7_rolling_forecasts')
    return fig


def fig_e5(res):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    tr, te = res['train_reports'], res['test_reports']
    t_te = np.array(res['t_test'])
    base = res['baselines']
    tb = np.array(base['t'])
    for a, key, lab in [(ax[0], 'C', 'Cumulative cases'), (ax[1], 'D', 'Cumulative deaths')]:
        a.plot(tr['day'], tr[key], 'ko', ms=4, label='training reports (days 0-89)')
        a.plot(te['day'], te[key], 's', color='#c0392b', ms=5, label='held-out reports (days 90-119)')
        for name in ['constant_beta', 'piecewise_beta']:
            l, col, ls = METHOD_STYLE[name]
            a.plot(tb, base[name][key], ls, color=col, lw=1.8, label=l + ' (Module 5)')
        runs = np.array([r[key] for r in res['pinn_runs']])
        a.fill_between(t_te, runs.min(axis=0), runs.max(axis=0), color=LIGHT, lw=0, label='PINN seed range')
        a.plot(t_te, res[f'ens_{key}'], '-', color='#7b2d8e', lw=2.2, label='PINN ensemble forecast')
        r0 = res['pinn_runs'][0]
        a.plot(r0['t_grid'], np.median([r[f'{key}_grid'] for r in res['pinn_runs']], axis=0), '-', color='#7b2d8e', lw=1.2)
        a.axvline(89.5, color='#888888', ls=':', lw=1)
        a.set(xlabel='Days since 25 May 2014', ylabel=lab, title=f'E5: {lab.lower()}, 30-day held-out forecast')
        a.grid(True, ls=':', color='#bbbbbb')
        a.legend(fontsize=7.5, loc='upper left')
    r = res['pinn_runs']
    tg = np.array(r[0]['t_grid'])
    for run in r:
        ax[2].plot(tg, np.array(run['beta_grid']) / (px.sp.GAMMA + run['mu']), color='#b48ec2', lw=1)
    ax[2].plot(tg, np.median([np.array(run['beta_grid']) / (px.sp.GAMMA + run['mu']) for run in r], axis=0),
               color='#7b2d8e', lw=2.2, label='median of 5 seeds')
    ax[2].axhline(1.0, color='k', ls='--', lw=1)
    ax[2].set(xlabel='Days since 25 May 2014', ylabel=r'$R_t$', title=r'E5: PINN $R_t=\beta(t)s(t)/(\gamma+\mu)$, training window')
    ax[2].grid(True, ls=':', color='#bbbbbb')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    px.save_figure(fig, 'fig_e5_real_forecast')
    return fig
