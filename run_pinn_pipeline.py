"""
run_pinn_pipeline.py
--------------------
Master execution pipeline for Physics-Informed Neural Network (PINN) experiments
on the SEIRD epidemic model and the 2014 Sierra Leone Ebola Outbreak.

Usage:
    python run_pinn_pipeline.py --task synthetic --epochs 1200
    python run_pinn_pipeline.py --task real --epochs 2000
    python run_pinn_pipeline.py --task all
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import torch
import torch.optim as optim

from pinn_model import (
    SEIRDStateNet, ScaledSEIRDEpidemicNet, BetaNet, SEIRDPINN,
    SIGMA_DEFAULT, GAMMA_DEFAULT, MU_DEFAULT
)
from data_pipeline import (
    process_sierra_leone_data, POPULATION_SIERRA_LEONE,
    PROCESSED_DATA_PATH
)

FIGS_DIR = os.path.join(os.path.dirname(__file__), 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seird_ode_system(t, y, beta_func_or_val, sigma=SIGMA_DEFAULT, gamma=GAMMA_DEFAULT, mu=MU_DEFAULT):
    s, e, i, r, d = y
    beta = beta_func_or_val(t) if callable(beta_func_or_val) else beta_func_or_val
    ds = -beta * s * i
    de =  beta * s * i - sigma * e
    di =  sigma * e - (gamma + mu) * i
    dr =  gamma * i
    dd =  mu * i
    return [ds, de, di, dr, dd]

def solve_seird_dop853(beta_val_or_func, y0, t_eval, sigma=SIGMA_DEFAULT, gamma=GAMMA_DEFAULT, mu=MU_DEFAULT):
    t_span = (t_eval[0], t_eval[-1])
    sol = solve_ivp(
        fun=lambda t, y: seird_ode_system(t, y, beta_val_or_func, sigma, gamma, mu),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12
    )
    return sol.y.T

def save_clean_figure(fig, filename_base):
    eps_path = os.path.join(FIGS_DIR, f"{filename_base}.eps")
    png_path = os.path.join(FIGS_DIR, f"{filename_base}.png")
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[pipeline] Saved plots: {eps_path} and {png_path}")

# -------------------------------------------------------------------------
# Experiment E2: Forward PINN Integration
# -------------------------------------------------------------------------
def run_experiment_e2(epochs=1200, lr=1.5e-3, n_colloc=250):
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT E2: Forward PINN Integration vs DOP853")
    print("="*70)
    set_seeds(42)

    T_max = 60.0
    beta_const = 0.4858
    y0 = [0.998, 0.001, 0.001, 0.0, 0.0]
    t_obs = np.linspace(0.0, T_max, 61)
    
    y_true = solve_seird_dop853(beta_const, y0, t_obs)
    
    tau_obs = torch.tensor(t_obs / T_max, dtype=torch.float32).view(-1, 1)
    y_obs_tensor = torch.tensor(y_true, dtype=torch.float32)
    y0_tensor = torch.tensor(y0, dtype=torch.float32).view(1, -1)

    state_net = SEIRDStateNet(hidden_dim=48, num_layers=3)
    pinn = SEIRDPINN(state_net=state_net, T_max=T_max)
    optimizer = optim.Adam(state_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        tau_colloc = torch.rand((n_colloc, 1), dtype=torch.float32)
        loss, loss_dict = pinn.compute_loss(
            tau_obs=tau_obs,
            y_obs=y_obs_tensor,
            tau_colloc=tau_colloc,
            y0=y0_tensor,
            obs_mode='full',
            constant_beta=beta_const,
            w_data=15.0,
            w_ode=1.0,
            w_ic=15.0,
            w_reg=0.0
        )
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 400 == 0 or (epoch + 1) == epochs:
            print(f"  Epoch [{epoch+1:4d}/{epochs}] | Loss: {loss.item():.5e}")

    with torch.no_grad():
        y_pred = pinn.compute_states(tau_obs).cpu().numpy()

    compartments = ['Susceptible $s$', 'Exposed $e$', 'Infectious $i$', 'Recovered $r$', 'Deceased $d$']
    e2_metrics = []
    for j in range(5):
        l2_norm_true = np.linalg.norm(y_true[:, j])
        l2_err = np.linalg.norm(y_pred[:, j] - y_true[:, j]) / (l2_norm_true if l2_norm_true > 1e-12 else 1.0)
        rmse = np.sqrt(np.mean((y_pred[:, j] - y_true[:, j])**2))
        e2_metrics.append({'Compartment': compartments[j], 'Relative L2 Error': l2_err, 'RMSE': rmse})
        print(f"  {compartments[j]:<20} | Rel L2 Error: {l2_err:.4e} | RMSE: {rmse:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
    for j in range(5):
        axes[0].plot(t_obs, y_true[:, j], label=f"{compartments[j]} (DOP853)", color=colors[j], linewidth=2.0)
        axes[0].plot(t_obs, y_pred[:, j], '--', label=f"{compartments[j]} (PINN)", color=colors[j], linewidth=1.5)
    axes[0].set_title("E2: Forward State Trajectories (Truth vs PINN)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Time (days)", fontsize=11)
    axes[0].set_ylabel("Population Fraction", fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', fontsize=8, ncol=2)

    for j in range(5):
        err = np.abs(y_pred[:, j] - y_true[:, j])
        axes[1].semilogy(t_obs, err, label=f"|Err| {compartments[j]}", color=colors[j], linewidth=1.8)
    axes[1].set_title("E2: Absolute Error vs Time", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Time (days)", fontsize=11)
    axes[1].set_ylabel("Absolute Error (log scale)", fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower right', fontsize=8)

    save_clean_figure(fig, 'fig_e2_forward_pinn')
    return e2_metrics

# -------------------------------------------------------------------------
# Experiment E3: Inverse beta(t) Recovery Under Noise
# -------------------------------------------------------------------------
def run_experiment_e3(epochs=1500, lr=1.5e-3, n_colloc=300):
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT E3: Inverse Dynamic beta(t) Recovery Under Noise")
    print("="*70)
    set_seeds(42)

    T_max = 60.0
    y0 = [0.998, 0.001, 0.001, 0.0, 0.0]
    t_obs = np.linspace(0.0, T_max, 61)
    tau_obs = torch.tensor(t_obs / T_max, dtype=torch.float32).view(-1, 1)
    y0_tensor = torch.tensor(y0, dtype=torch.float32).view(1, -1)

    profiles = {
        'Step Intervention': lambda t: np.where(t < 25.0, 0.45, 0.18),
        'Seasonal Sinusoid': lambda t: 0.32 + 0.12 * np.sin(2.0 * np.pi * t / 30.0)
    }

    results = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_idx = 0

    for name, b_func in profiles.items():
        y_clean = solve_seird_dop853(b_func, y0, t_obs)
        b_true = b_func(t_obs)

        for noise_lvl in [0.01, 0.05]:
            set_seeds(42)
            noise = np.random.normal(0.0, noise_lvl * np.std(y_clean, axis=0, keepdims=True), size=y_clean.shape)
            y_noisy = np.clip(y_clean + noise, 0.0, 1.0)
            y_obs_tensor = torch.tensor(y_noisy, dtype=torch.float32)

            state_net = SEIRDStateNet(hidden_dim=48)
            beta_net = BetaNet(hidden_dim=32)
            pinn = SEIRDPINN(state_net=state_net, beta_net=beta_net, T_max=T_max)
            optimizer = optim.Adam(list(state_net.parameters()) + list(beta_net.parameters()), lr=lr)

            for epoch in range(epochs):
                optimizer.zero_grad()
                tau_colloc = torch.rand((n_colloc, 1), dtype=torch.float32)
                loss, _ = pinn.compute_loss(
                    tau_obs=tau_obs,
                    y_obs=y_obs_tensor,
                    tau_colloc=tau_colloc,
                    y0=y0_tensor,
                    obs_mode='full',
                    w_data=20.0,
                    w_ode=1.0,
                    w_ic=20.0,
                    w_reg=0.02
                )
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                b_pred = beta_net(tau_obs).cpu().numpy().flatten()

            rmse = np.sqrt(np.mean((b_pred - b_true)**2))
            rel_l2 = np.linalg.norm(b_pred - b_true) / np.linalg.norm(b_true) * 100.0
            results.append({
                'Profile': name,
                'Noise Level': f"{int(noise_lvl*100)}%",
                'beta RMSE': f"{rmse:.4f}",
                'Relative L2 Error (%)': f"{rel_l2:.2f}%"
            })
            print(f"  Profile: {name:<18} | Noise: {int(noise_lvl*100)}% | RMSE: {rmse:.4f} | Rel L2: {rel_l2:.2f}%")

            if noise_lvl == 0.05:
                ax = axes[plot_idx]
                ax.plot(t_obs, b_true, 'k-', linewidth=2.5, label='True β(t)')
                ax.plot(t_obs, b_pred, 'r--', linewidth=2.0, label='Inverted β(t) (5% Noise)')
                ax.set_title(f"E3: {name} (5% Noise)", fontsize=11, fontweight='bold')
                ax.set_xlabel("Time (days)", fontsize=10)
                ax.set_ylabel("Transmission Rate β(t) [day⁻¹]", fontsize=10)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='best', fontsize=9)
                plot_idx += 1

    save_clean_figure(fig, 'fig_e3_inverse_beta')
    return pd.DataFrame(results)

# -------------------------------------------------------------------------
# Experiment E4: Hidden Latent State Recovery
# -------------------------------------------------------------------------
def run_experiment_e4(epochs=1800, lr=1.5e-3, n_colloc=300):
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT E4: Latent Hidden-State Recovery (Cases & Deaths Only)")
    print("="*70)
    set_seeds(42)

    T_max = 60.0
    beta_const = 0.4858
    y0 = [0.998, 0.001, 0.001, 0.0, 0.0]
    t_obs = np.linspace(0.0, T_max, 61)
    
    y_full = solve_seird_dop853(beta_const, y0, t_obs)
    s_true = y_full[:, 0]
    e_true = y_full[:, 1]
    
    cases_obs = 1.0 - s_true - e_true
    deaths_obs = y_full[:, 4]
    
    cases_noisy = cases_obs + np.random.normal(0, 0.02 * np.std(cases_obs), size=cases_obs.shape)
    deaths_noisy = deaths_obs + np.random.normal(0, 0.02 * np.std(deaths_obs), size=deaths_obs.shape)
    
    y_obs_cd = np.column_stack([cases_noisy, deaths_noisy])
    tau_obs = torch.tensor(t_obs / T_max, dtype=torch.float32).view(-1, 1)
    y_obs_tensor = torch.tensor(y_obs_cd, dtype=torch.float32)
    y0_tensor = torch.tensor(y0, dtype=torch.float32).view(1, -1)

    state_net = SEIRDStateNet(hidden_dim=48)
    pinn = SEIRDPINN(state_net=state_net, T_max=T_max)
    optimizer = optim.Adam(state_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        tau_colloc = torch.rand((n_colloc, 1), dtype=torch.float32)
        loss, _ = pinn.compute_loss(
            tau_obs=tau_obs,
            y_obs=y_obs_tensor,
            tau_colloc=tau_colloc,
            y0=y0_tensor,
            obs_mode='cases_and_deaths',
            constant_beta=beta_const,
            w_data=25.0,
            w_ode=1.0,
            w_ic=25.0,
            w_reg=0.0
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_pred = state_net(tau_obs).cpu().numpy()
        s_pred = u_pred[:, 0]
        e_pred = u_pred[:, 1]

    s_rmse = np.sqrt(np.mean((s_pred - s_true)**2))
    e_rmse = np.sqrt(np.mean((e_pred - e_true)**2))
    s_rel = s_rmse / np.mean(s_true) * 100.0
    e_rel = e_rmse / (np.max(e_true) - np.min(e_true)) * 100.0

    print(f"  Latent Susceptible s(t) | RMSE: {s_rmse:.5f} | Relative Error: {s_rel:.2f}%")
    print(f"  Latent Incubating  e(t) | RMSE: {e_rmse:.5f} | Relative Error: {e_rel:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(t_obs, s_true, 'k-', linewidth=2.2, label='True Susceptible s(t)')
    axes[0].plot(t_obs, s_pred, 'b--', linewidth=2.0, label='PINN Inverted s(t)')
    axes[0].set_title("E4: Latent Susceptible Pool s(t) Inversion", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Time (days)", fontsize=10)
    axes[0].set_ylabel("Fraction s(t)", fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='best', fontsize=9)

    axes[1].plot(t_obs, e_true, 'k-', linewidth=2.2, label='True Incubating e(t)')
    axes[1].plot(t_obs, e_pred, 'g--', linewidth=2.0, label='PINN Inverted e(t)')
    axes[1].set_title("E4: Latent Incubating Pool e(t) Inversion", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Time (days)", fontsize=10)
    axes[1].set_ylabel("Fraction e(t)", fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='best', fontsize=9)

    save_clean_figure(fig, 'fig_e4_hidden_states')
    return {'s_rmse': s_rmse, 'e_rmse': e_rmse}

# -------------------------------------------------------------------------
# Experiment E5: Real-World 2014 Sierra Leone Ebola Outbreak Application
# -------------------------------------------------------------------------
def run_real_outbreak_e5(epochs=2000, lr=2.0e-3, n_colloc=250):
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT E5: Real-World Outbreak Application (Sierra Leone 2014)")
    print("="*70)
    set_seeds(42)

    if not os.path.exists(PROCESSED_DATA_PATH):
        process_sierra_leone_data()
    df = pd.read_csv(PROCESSED_DATA_PATH)

    N = float(POPULATION_SIERRA_LEONE) # 7,092,113
    train_mask = (df['split'] == 'train')
    test_mask = (df['split'] == 'test')

    t_all = df['day'].values.astype(float)
    t_train = df.loc[train_mask, 'day'].values.astype(float)
    t_test = df.loc[test_mask, 'day'].values.astype(float)

    cases_train = df.loc[train_mask, 'cumulative_cases'].values.astype(float)
    deaths_train = df.loc[train_mask, 'cumulative_deaths'].values.astype(float)
    cases_test = df.loc[test_mask, 'cumulative_cases'].values.astype(float)
    deaths_test = df.loc[test_mask, 'cumulative_deaths'].values.astype(float)

    T_train = float(t_train[-1]) # Day 89
    scale_C = float(cases_train[-1])  # 949.0
    scale_D = float(deaths_train[-1]) # 402.0

    i0 = max(11.0 / N, 1e-6)
    e0 = max(16.0 / N, 1e-6)
    d0 = 5.0 / N
    r0 = 0.0
    s0 = 1.0 - (e0 + i0 + r0 + d0)
    y0_real = [s0, e0, i0, r0, d0]

    # 1. Baseline 1: Constant-beta Least Squares Fit
    print("\n--- Fitting Baseline 1: Constant-beta Least Squares ---")
    def ls_constant_res(params):
        sim = solve_seird_dop853(params[0], y0_real, t_train)
        sim_c = (1.0 - sim[:, 0] - sim[:, 1]) * N
        sim_d = sim[:, 4] * N
        return np.concatenate([(sim_c - cases_train) / (np.mean(cases_train) + 1.0),
                               (sim_d - deaths_train) / (np.mean(deaths_train) + 1.0)])

    res_opt = least_squares(ls_constant_res, x0=[0.25], bounds=(0.01, 1.5))
    beta_const_fit = res_opt.x[0]
    print(f"  Fitted Constant beta: {beta_const_fit:.4f} day^-1 (R0 = {beta_const_fit/(GAMMA_DEFAULT+MU_DEFAULT):.2f})")
    sim_const_all = solve_seird_dop853(beta_const_fit, y0_real, t_all)
    cases_pred_const = (1.0 - sim_const_all[:, 0] - sim_const_all[:, 1]) * N
    deaths_pred_const = sim_const_all[:, 4] * N

    # 2. Baseline 2: Piecewise-beta Least Squares
    print("\n--- Fitting Baseline 2: Piecewise-beta Least Squares ---")
    def piecewise_beta_eval(params, t):
        b0, b1, b2 = params
        if np.isscalar(t):
            return b0 if t < 30.0 else (b1 if t < 60.0 else b2)
        b = np.empty_like(t)
        b[t < 30.0] = b0
        b[(t >= 30.0) & (t < 60.0)] = b1
        b[t >= 60.0] = b2
        return b

    def ls_piecewise_res(params):
        sim = solve_seird_dop853(lambda t: piecewise_beta_eval(params, t), y0_real, t_train)
        sim_c = (1.0 - sim[:, 0] - sim[:, 1]) * N
        sim_d = sim[:, 4] * N
        smooth = 0.5 * np.array([params[1] - params[0], params[2] - params[1]])
        return np.concatenate([(sim_c - cases_train) / (np.mean(cases_train) + 1.0),
                               (sim_d - deaths_train) / (np.mean(deaths_train) + 1.0),
                               smooth])

    res_pw = least_squares(ls_piecewise_res, x0=[0.30, 0.25, 0.20], bounds=(0.01, 1.5))
    params_pw = res_pw.x
    print(f"  Fitted Piecewise beta segments: b0={params_pw[0]:.4f}, b1={params_pw[1]:.4f}, b2={params_pw[2]:.4f}")
    sim_pw_all = solve_seird_dop853(lambda t: piecewise_beta_eval(params_pw, t), y0_real, t_all)
    cases_pred_pw = (1.0 - sim_pw_all[:, 0] - sim_pw_all[:, 1]) * N
    deaths_pred_pw = sim_pw_all[:, 4] * N

    # 3. Model 3: Characteristic Scale-Normalized PINN
    print("\n--- Training Model 3: Scale-Normalized PINN with Dynamic beta(t) ---")
    c_norm = torch.tensor(cases_train / scale_C, dtype=torch.float32).view(-1, 1)
    d_norm = torch.tensor(deaths_train / scale_D, dtype=torch.float32).view(-1, 1)
    tau_train = torch.tensor(t_train / T_train, dtype=torch.float32).view(-1, 1)

    state_net = ScaledSEIRDEpidemicNet(hidden_dim=48)
    beta_net = BetaNet(hidden_dim=32)
    optimizer = optim.Adam(list(state_net.parameters()) + list(beta_net.parameters()), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = state_net(tau_train)
        e_t, i_t, r_t, d_t = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        
        pred_c = (i_t + r_t + d_t)
        pred_d = d_t * (scale_C / scale_D)
        loss_data = torch.mean((pred_c - c_norm)**2) + torch.mean((pred_d - d_norm)**2)
        
        tau_f = torch.rand((n_colloc, 1), requires_grad=True)
        out_f = state_net(tau_f)
        e_f, i_f, r_f, d_f = out_f[:, 0:1], out_f[:, 1:2], out_f[:, 2:3], out_f[:, 3:4]
        beta_f = beta_net(tau_f)
        
        de = torch.autograd.grad(e_f, tau_f, torch.ones_like(e_f), create_graph=True)[0]
        di = torch.autograd.grad(i_f, tau_f, torch.ones_like(i_f), create_graph=True)[0]
        dr = torch.autograd.grad(r_f, tau_f, torch.ones_like(r_f), create_graph=True)[0]
        dd = torch.autograd.grad(d_f, tau_f, torch.ones_like(d_f), create_graph=True)[0]
        
        res_e = de - T_train * (beta_f * i_f - SIGMA_DEFAULT * e_f)
        res_i = di - T_train * (SIGMA_DEFAULT * e_f - (GAMMA_DEFAULT + MU_DEFAULT) * i_f)
        res_r = dr - T_train * (GAMMA_DEFAULT * i_f)
        res_d = dd - T_train * (MU_DEFAULT * i_f)
        loss_ode = torch.mean(res_e**2 + res_i**2 + res_r**2 + res_d**2)
        
        out_0 = state_net(torch.tensor([[0.0]]))
        loss_ic = torch.mean((out_0[:, 3:4] - torch.tensor([[deaths_train[0] / scale_C]]))**2)
        
        dbeta_dtau = torch.autograd.grad(beta_f, tau_f, torch.ones_like(beta_f), create_graph=True)[0]
        loss_smooth = torch.mean(dbeta_dtau ** 2)
        
        loss = 25.0 * loss_data + 1.0 * loss_ode + 10.0 * loss_ic + 0.02 * loss_smooth
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0 or (epoch + 1) == epochs:
            print(f"  Epoch [{epoch+1:4d}/{epochs}] | Loss: {loss.item():.5e} (Data: {loss_data.item():.2e}, ODE: {loss_ode.item():.2e})")

    with torch.no_grad():
        out_train = state_net(tau_train).numpy()
        pred_c_train = (out_train[:, 1] + out_train[:, 2] + out_train[:, 3]) * scale_C
        pred_d_train = out_train[:, 3] * scale_C
        learned_beta = beta_net(tau_train).numpy().flatten()
        terminal_beta = float(learned_beta[-1])

    print(f"  PINN Learned Transmission: Initial beta={learned_beta[0]:.4f}, Terminal beta={terminal_beta:.4f} day^-1")

    # Forward extrapolation from terminal calibrated state
    term_e = float(out_train[-1, 0]) * scale_C / N
    term_i = float(out_train[-1, 1]) * scale_C / N
    term_r = float(out_train[-1, 2]) * scale_C / N
    term_d = float(out_train[-1, 3]) * scale_C / N
    term_s = 1.0 - (term_e + term_i + term_r + term_d)
    
    sim_test = solve_seird_dop853(terminal_beta, [term_s, term_e, term_i, term_r, term_d], t_test - t_test[0])
    cases_test_pred = (sim_test[:, 1] + sim_test[:, 2] + sim_test[:, 3] + sim_test[:, 4]) * N
    deaths_test_pred = sim_test[:, 4] * N

    cases_pred_pinn = np.concatenate([pred_c_train, cases_test_pred])
    deaths_pred_pinn = np.concatenate([pred_d_train, deaths_test_pred])

    # Held-out metrics
    def calc_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        rel_err = (mae / np.mean(y_true)) * 100.0
        return mae, rmse, rel_err

    m_const_c = calc_metrics(cases_test, cases_pred_const[test_mask])
    m_const_d = calc_metrics(deaths_test, deaths_pred_const[test_mask])

    m_pw_c = calc_metrics(cases_test, cases_pred_pw[test_mask])
    m_pw_d = calc_metrics(deaths_test, deaths_pred_pw[test_mask])

    m_pinn_c = calc_metrics(cases_test, cases_pred_pinn[test_mask])
    m_pinn_d = calc_metrics(deaths_test, deaths_pred_pinn[test_mask])

    eval_table = [
        {'Method': 'Constant-beta Least Squares', 'Target': 'Cases', 'MAE': m_const_c[0], 'RMSE': m_const_c[1], 'RelErr(%)': m_const_c[2]},
        {'Method': 'Constant-beta Least Squares', 'Target': 'Deaths', 'MAE': m_const_d[0], 'RMSE': m_const_d[1], 'RelErr(%)': m_const_d[2]},
        {'Method': 'Piecewise-beta Least Squares', 'Target': 'Cases', 'MAE': m_pw_c[0], 'RMSE': m_pw_c[1], 'RelErr(%)': m_pw_c[2]},
        {'Method': 'Piecewise-beta Least Squares', 'Target': 'Deaths', 'MAE': m_pw_d[0], 'RMSE': m_pw_d[1], 'RelErr(%)': m_pw_d[2]},
        {'Method': 'PINN with Dynamic beta(t)', 'Target': 'Cases', 'MAE': m_pinn_c[0], 'RMSE': m_pinn_c[1], 'RelErr(%)': m_pinn_c[2]},
        {'Method': 'PINN with Dynamic beta(t)', 'Target': 'Deaths', 'MAE': m_pinn_d[0], 'RMSE': m_pinn_d[1], 'RelErr(%)': m_pinn_d[2]},
    ]
    eval_df = pd.DataFrame(eval_table)
    print("\n" + "="*70)
    print("HELD-OUT FORECAST EVALUATION METRICS (DAYS 90-119):")
    print("="*70)
    print(eval_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(t_train, cases_train, 'ko', markersize=3.5, label='Training Data (Days 0-89)')
    axes[0].plot(t_test, cases_test, 'ro', markersize=3.5, label='Held-Out Test Data (Days 90-119)')
    axes[0].plot(t_all, cases_pred_const, 'b--', linewidth=1.8, label=f"Constant-β (RelErr={m_const_c[2]:.1f}%)")
    axes[0].plot(t_all, cases_pred_pw, 'g-', linewidth=2.0, label=f"Piecewise-β (RelErr={m_pw_c[2]:.1f}%)")
    axes[0].plot(t_all, cases_pred_pinn, 'm-.', linewidth=2.0, label=f"PINN (RelErr={m_pinn_c[2]:.1f}%)")
    axes[0].axvline(x=89.5, color='gray', linestyle=':', label='Train/Test Split (Day 90)')
    axes[0].set_title("E5: Cumulative Diagnosed Cases C(t) Forecast", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Time (days since May 25, 2014)", fontsize=10)
    axes[0].set_ylabel("Cumulative Cases", fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper left', fontsize=8)

    axes[1].plot(t_train, deaths_train, 'ko', markersize=3.5, label='Training Data (Days 0-89)')
    axes[1].plot(t_test, deaths_test, 'ro', markersize=3.5, label='Held-Out Test Data (Days 90-119)')
    axes[1].plot(t_all, deaths_pred_const, 'b--', linewidth=1.8, label=f"Constant-β (RelErr={m_const_d[2]:.1f}%)")
    axes[1].plot(t_all, deaths_pred_pw, 'g-', linewidth=2.0, label=f"Piecewise-β (RelErr={m_pw_d[2]:.1f}%)")
    axes[1].plot(t_all, deaths_pred_pinn, 'm-.', linewidth=2.0, label=f"PINN (RelErr={m_pinn_d[2]:.1f}%)")
    axes[1].axvline(x=89.5, color='gray', linestyle=':', label='Train/Test Split (Day 90)')
    axes[1].set_title("E5: Cumulative Recorded Deaths D(t) Forecast", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Time (days since May 25, 2014)", fontsize=10)
    axes[1].set_ylabel("Cumulative Deaths", fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper left', fontsize=8)

    save_clean_figure(fig, 'fig_e5_real_forecast')
    return eval_df

def main():
    parser = argparse.ArgumentParser(description="PINN Pipeline for SEIRD Model and Ebola Outbreak")
    parser.add_argument('--task', type=str, default='all', choices=['synthetic', 'real', 'all', 'e2', 'e3', 'e4', 'e5'])
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1.5e-3)
    parser.add_argument('--colloc', type=int, default=300)
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  STARTING PINN PIPELINE [Task: {args.task}]")
    print(f"=======================================================")

    if args.task in ['synthetic', 'e2', 'all']:
        run_experiment_e2(epochs=min(args.epochs, 1200), lr=args.lr, n_colloc=args.colloc)
    if args.task in ['synthetic', 'e3', 'all']:
        run_experiment_e3(epochs=args.epochs, lr=args.lr, n_colloc=args.colloc)
    if args.task in ['synthetic', 'e4', 'all']:
        run_experiment_e4(epochs=args.epochs, lr=args.lr, n_colloc=args.colloc)
    if args.task in ['real', 'e5', 'all']:
        run_real_outbreak_e5(epochs=max(args.epochs, 2000), lr=args.lr, n_colloc=args.colloc)

    print("\n[SUCCESS] Pipeline execution finished cleanly.")

if __name__ == '__main__':
    main()
