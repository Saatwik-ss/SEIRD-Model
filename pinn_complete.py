"""
pinn_complete.py
================
Complete, self-contained Physics-Informed Neural Network (PINN) implementation
for the SEIRD compartmental epidemic model with dynamic transmission rate beta(t),
incorporating characteristic scale normalization, synthetic benchmarks (E2-E4),
and real-world calibration on the 2014 Sierra Leone Ebola Outbreak (E5).

Under no condition modifies the improved scale-normalized PINN formulation or its
training loop. Writes out all graphs (.png and .eps) and all evaluation metrics
(MAE, RMSE, MAPE, RelErr%, MaxErr, R2) to terminal and disk.

Dependencies:
    pip install torch numpy scipy pandas matplotlib

Execution:
    python pinn_complete.py               # Runs complete end-to-end suite (E2, E3, E4, E5)
    python pinn_complete.py --task e5     # Runs real 2014 Sierra Leone Ebola calibration
    python pinn_complete.py --task e2     # Runs forward PINN integration test
    python pinn_complete.py --task e3     # Runs inverse beta(t) recovery under noise
    python pinn_complete.py --task e4     # Runs hidden latent state recovery
"""

import os
import sys
import random
import urllib.request
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. EPIDEMIOLOGICAL PARAMETERS & CONVENTIONS
# =============================================================================
POPULATION_SIERRA_LEONE = 7_092_113

# Clinically anchored transition parameters (WHO Ebola Response Team, NEJM 2014)
SIGMA = 1.0 / 9.4      # Incubation transition rate (day^-1, ~9.4 days latency)
GAMMA = 1.0 / 7.0      # Recovery clearance rate (day^-1, ~7.0 days infectiousness)
MU    = 0.1000         # Disease-induced mortality rate (day^-1, CFR ~ 41.2%)
BETA_BASELINE = 0.4858 # Baseline contact rate yielding R0 = beta / (gamma + mu) = 2.00

OUTPUT_DIR = "figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_plots(fig, filename_base):
    eps_path = os.path.join(OUTPUT_DIR, f"{filename_base}.eps")
    png_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot Saved] {eps_path} and {png_path}")

# =============================================================================
# 2. NUMERICAL REFERENCE ODE SOLVER (DOP853)
# =============================================================================
def seird_rhs(t, y, beta_func_or_val, sigma=SIGMA, gamma=GAMMA, mu=MU):
    s, e, i, r, d = y
    beta = beta_func_or_val(t) if callable(beta_func_or_val) else beta_func_or_val
    ds = -beta * s * i
    de =  beta * s * i - sigma * e
    di =  sigma * e - (gamma + mu) * i
    dr =  gamma * i
    dd =  mu * i
    return [ds, de, di, dr, dd]

def solve_seird_reference(beta_val_or_func, y0, t_eval, sigma=SIGMA, gamma=GAMMA, mu=MU):
    sol = solve_ivp(
        fun=lambda t, y: seird_rhs(t, y, beta_val_or_func, sigma, gamma, mu),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12
    )
    return sol.y.T

# =============================================================================
# 3. NEURAL NETWORK ARCHITECTURES
# =============================================================================
class SEIRDStateNet(nn.Module):
    """
    Normalized state network for synthetic benchmarks (E2-E4).
    Maps dimensionless time tau = t / T in [0, 1] -> (s, e, i, r, d) in (0, 1)^5.
    """
    def __init__(self, hidden_dim=48, num_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 5))
        self.net = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tau):
        return self.sigmoid(self.net(tau))


class ScaledSEIRDEpidemicNet(nn.Module):
    """
    Characteristic scale-normalized network for real outbreaks with low attack rates (E5).
    Predicts positive active counts: (e_tilde, i_tilde, r_tilde, d_tilde) in units of C_scale.
    This guarantees gradients are O(1) and eliminates vanishing gradients when I/N ~ 1e-5.
    """
    def __init__(self, hidden_dim=48, num_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, tau):
        return self.softplus(self.net(tau))


class BetaNet(nn.Module):
    """
    Transmission network mapping tau in [0, 1] -> beta(t) > 0.
    Softplus activation guarantees strict biological positivity without artificial bounds.
    """
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, tau):
        return self.softplus(self.net(tau))

# =============================================================================
# 4. DATA LOADER: 2014 SIERRA LEONE EBOLA SURVEILLANCE
# =============================================================================
RAW_URL = "https://raw.githubusercontent.com/cmrivers/ebola/master/country_timeseries.csv"

def load_ebola_sierra_leone_data(cache_csv="data/outbreak_data.csv"):
    if os.path.exists(cache_csv):
        return pd.read_csv(cache_csv)

    print("[Data] Downloading Caitlin Rivers Ebola repository...")
    os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
    raw_path = "data/country_timeseries_raw.csv"
    if not os.path.exists(raw_path):
        urllib.request.urlretrieve(RAW_URL, raw_path)

    df = pd.read_csv(raw_path)
    df['Date'] = pd.to_datetime(df['Date'])
    sl = df[['Date', 'Cases_SierraLeone', 'Deaths_SierraLeone']].dropna().sort_values('Date').reset_index(drop=True)

    start_date = pd.Timestamp('2014-05-25')
    end_date = pd.Timestamp('2014-09-21')
    daily_df = pd.DataFrame({'Date': pd.date_range(start_date, end_date, freq='D')})

    merged = pd.merge(daily_df, sl, on='Date', how='left')
    merged['cumulative_cases'] = merged['Cases_SierraLeone'].interpolate(method='linear').bfill().ffill().round().astype(int)
    merged['cumulative_deaths'] = merged['Deaths_SierraLeone'].interpolate(method='linear').bfill().ffill().round().astype(int)
    merged['cumulative_cases'] = np.maximum.accumulate(merged['cumulative_cases'])
    merged['cumulative_deaths'] = np.maximum.accumulate(merged['cumulative_deaths'])

    merged['day'] = range(len(merged))
    merged['date'] = merged['Date'].dt.strftime('%Y-%m-%d')
    merged['daily_cases'] = merged['cumulative_cases'].diff().fillna(merged['cumulative_cases'].iloc[0]).clip(lower=0).astype(int)
    merged['daily_deaths'] = merged['cumulative_deaths'].diff().fillna(merged['cumulative_deaths'].iloc[0]).clip(lower=0).astype(int)
    merged['daily_cases_7day_avg'] = merged['daily_cases'].rolling(7, center=True, min_periods=1).mean().round(1)

    split_idx = int(len(merged) * 0.75) # 90
    merged['split'] = ['train' if i < split_idx else 'test' for i in range(len(merged))]

    cols = ['day', 'date', 'daily_cases', 'daily_cases_7day_avg', 'cumulative_cases', 'daily_deaths', 'cumulative_deaths', 'split']
    final_df = merged[cols]
    final_df.to_csv(cache_csv, index=False)
    print(f"[Data] Saved preprocessed dataset ({len(final_df)} days) to {cache_csv}")
    return final_df

# =============================================================================
# 5. EXPERIMENT E2: FORWARD PINN INTEGRATION VS DOP853
# =============================================================================
def run_experiment_e2(epochs=1200, lr=1.5e-3, n_colloc=250):
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT E2: Forward PINN Integration vs DOP853")
    print("="*70)
    set_seed(42)

    T_max = 60.0
    beta_const = BETA_BASELINE
    y0 = [0.998, 0.001, 0.001, 0.0, 0.0]
    t_obs = np.linspace(0.0, T_max, 61)

    y_true = solve_seird_reference(beta_const, y0, t_obs)
    tau_obs = torch.tensor(t_obs / T_max, dtype=torch.float32).view(-1, 1)
    y_obs_tensor = torch.tensor(y_true, dtype=torch.float32)
    y0_tensor = torch.tensor(y0, dtype=torch.float32).view(1, -1)

    net = SEIRDStateNet(hidden_dim=48, num_layers=3)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_obs = net(tau_obs)
        loss_data = torch.mean((pred_obs - y_obs_tensor)**2)

        tau_f = torch.rand((n_colloc, 1), requires_grad=True)
        u = net(tau_f)
        s, e, i, r, d = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4], u[:, 4:5]

        ds = torch.autograd.grad(s, tau_f, torch.ones_like(s), create_graph=True)[0]
        de = torch.autograd.grad(e, tau_f, torch.ones_like(e), create_graph=True)[0]
        di = torch.autograd.grad(i, tau_f, torch.ones_like(i), create_graph=True)[0]
        dr = torch.autograd.grad(r, tau_f, torch.ones_like(r), create_graph=True)[0]
        dd = torch.autograd.grad(d, tau_f, torch.ones_like(d), create_graph=True)[0]

        res_s = ds - T_max * (-beta_const * s * i)
        res_e = de - T_max * (beta_const * s * i - SIGMA * e)
        res_i = di - T_max * (SIGMA * e - (GAMMA + MU) * i)
        res_r = dr - T_max * (GAMMA * i)
        res_d = dd - T_max * (MU * i)
        loss_ode = torch.mean(res_s**2 + res_e**2 + res_i**2 + res_r**2 + res_d**2)

        pred_0 = net(torch.tensor([[0.0]], dtype=torch.float32))
        loss_ic = torch.mean((pred_0 - y0_tensor)**2)

        loss = 15.0 * loss_data + 1.0 * loss_ode + 15.0 * loss_ic
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 400 == 0 or (epoch + 1) == epochs:
            print(f"  Epoch [{epoch+1:4d}/{epochs}] | Loss: {loss.item():.5e} (Data: {loss_data.item():.2e}, ODE: {loss_ode.item():.2e})")

    with torch.no_grad():
        y_pred = net(tau_obs).cpu().numpy()

    compartments = ['Susceptible s', 'Exposed e', 'Infectious i', 'Recovered r', 'Deceased d']
    print("\nE2 Quantitative Evaluation Table:")
    e2_rows = []
    for j in range(5):
        l2_err = np.linalg.norm(y_pred[:, j] - y_true[:, j]) / np.linalg.norm(y_true[:, j]) * 100.0
        rmse = np.sqrt(np.mean((y_pred[:, j] - y_true[:, j])**2))
        e2_rows.append({'Compartment': compartments[j], 'Rel_L2_Error(%)': f"{l2_err:.4f}%", 'RMSE': f"{rmse:.4e}"})
        print(f"  {compartments[j]:<15} | Rel L2 Error: {l2_err:.4f}% | RMSE: {rmse:.4e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
    for j in range(5):
        axes[0].plot(t_obs, y_true[:, j], label=f"{compartments[j]} (Truth)", color=colors[j], linewidth=2.0)
        axes[0].plot(t_obs, y_pred[:, j], '--', label=f"{compartments[j]} (PINN)", color=colors[j], linewidth=1.5)
    axes[0].set_title("E2: Forward State Trajectories (DOP853 vs PINN)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Time (days)", fontsize=10)
    axes[0].set_ylabel("Fraction of Population", fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', fontsize=8, ncol=2)

    for j in range(5):
        err = np.abs(y_pred[:, j] - y_true[:, j])
        axes[1].semilogy(t_obs, err, label=f"|Err| {compartments[j]}", color=colors[j], linewidth=1.8)
    axes[1].set_title("E2: Absolute Error vs Time", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Time (days)", fontsize=10)
    axes[1].set_ylabel("Absolute Error (log scale)", fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower right', fontsize=8)

    save_plots(fig, "fig_e2_forward_pinn")
    return net

# =============================================================================
# 6. EXPERIMENT E3: INVERSE DYNAMIC beta(t) RECOVERY UNDER NOISE
# =============================================================================
def run_experiment_e3(epochs=1500, lr=1.5e-3, n_colloc=300):
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT E3: Inverse Dynamic beta(t) Recovery Under Noise")
    print("="*70)
    set_seed(42)

    T_max = 60.0
    y0 = [0.998, 0.001, 0.001, 0.0, 0.0]
    t_obs = np.linspace(0.0, T_max, 61)
    tau_obs = torch.tensor(t_obs / T_max, dtype=torch.float32).view(-1, 1)
    y0_tensor = torch.tensor(y0, dtype=torch.float32).view(1, -1)

    profiles = {
        'Step Intervention': lambda t: np.where(t < 25.0, 0.45, 0.18),
        'Seasonal Sinusoid': lambda t: 0.32 + 0.12 * np.sin(2.0 * np.pi * t / 30.0)
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_idx = 0

    for name, b_func in profiles.items():
        y_clean = solve_seird_reference(b_func, y0, t_obs)
        b_true = b_func(t_obs)

        for noise_lvl in [0.01, 0.05]:
            set_seed(42)
            noise = np.random.normal(0.0, noise_lvl * np.std(y_clean, axis=0, keepdims=True), size=y_clean.shape)
            y_noisy = np.clip(y_clean + noise, 0.0, 1.0)
            y_obs_tensor = torch.tensor(y_noisy, dtype=torch.float32)

            state_net = SEIRDStateNet(hidden_dim=48)
            beta_net = BetaNet(hidden_dim=32)
            optimizer = optim.Adam(list(state_net.parameters()) + list(beta_net.parameters()), lr=lr)

            for epoch in range(epochs):
                optimizer.zero_grad()
                pred_obs = state_net(tau_obs)
                loss_data = torch.mean((pred_obs - y_obs_tensor)**2)

                tau_f = torch.rand((n_colloc, 1), requires_grad=True)
                u = state_net(tau_f)
                s, e, i, r, d = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4], u[:, 4:5]
                b_val = beta_net(tau_f)

                ds = torch.autograd.grad(s, tau_f, torch.ones_like(s), create_graph=True)[0]
                de = torch.autograd.grad(e, tau_f, torch.ones_like(e), create_graph=True)[0]
                di = torch.autograd.grad(i, tau_f, torch.ones_like(i), create_graph=True)[0]
                dr = torch.autograd.grad(r, tau_f, torch.ones_like(r), create_graph=True)[0]
                dd = torch.autograd.grad(d, tau_f, torch.ones_like(d), create_graph=True)[0]

                res_s = ds - T_max * (-b_val * s * i)
                res_e = de - T_max * (b_val * s * i - SIGMA * e)
                res_i = di - T_max * (SIGMA * e - (GAMMA + MU) * i)
                res_r = dr - T_max * (GAMMA * i)
                res_d = dd - T_max * (MU * i)
                loss_ode = torch.mean(res_s**2 + res_e**2 + res_i**2 + res_r**2 + res_d**2)

                pred_0 = state_net(torch.tensor([[0.0]], dtype=torch.float32))
                loss_ic = torch.mean((pred_0 - y0_tensor)**2)

                db_dtau = torch.autograd.grad(b_val, tau_f, torch.ones_like(b_val), create_graph=True)[0]
                loss_smooth = torch.mean(db_dtau**2)

                loss = 20.0 * loss_data + 1.0 * loss_ode + 20.0 * loss_ic + 0.02 * loss_smooth
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                b_pred = beta_net(tau_obs).cpu().numpy().flatten()

            rmse = np.sqrt(np.mean((b_pred - b_true)**2))
            rel_l2 = np.linalg.norm(b_pred - b_true) / np.linalg.norm(b_true) * 100.0
            print(f"  Profile: {name:<18} | Noise: {int(noise_lvl*100)}% | beta RMSE: {rmse:.4f} | Rel L2: {rel_l2:.2f}%")

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

    save_plots(fig, "fig_e3_inverse_beta")

# =============================================================================
# 7. EXPERIMENT E4: HIDDEN LATENT STATE RECOVERY (s & e FROM C & D ONLY)
# =============================================================================
def run_experiment_e4(epochs=1800, lr=1.5e-3, n_colloc=300):
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT E4: Latent Hidden-State Recovery (Cases & Deaths Only)")
    print("="*70)
    set_seed(42)

    T_max = 60.0
    beta_const = BETA_BASELINE
    y0 = [0.998, 0.001, 0.001, 0.0, 0.0]
    t_obs = np.linspace(0.0, T_max, 61)

    y_full = solve_seird_reference(beta_const, y0, t_obs)
    s_true = y_full[:, 0]
    e_true = y_full[:, 1]

    # Clinical surveillance: Only C(t) = 1 - s - e and D(t) = d are observed
    cases_obs = 1.0 - s_true - e_true
    deaths_obs = y_full[:, 4]
    cases_noisy = cases_obs + np.random.normal(0, 0.02 * np.std(cases_obs), size=cases_obs.shape)
    deaths_noisy = deaths_obs + np.random.normal(0, 0.02 * np.std(deaths_obs), size=deaths_obs.shape)

    y_obs_cd = np.column_stack([cases_noisy, deaths_noisy])
    tau_obs = torch.tensor(t_obs / T_max, dtype=torch.float32).view(-1, 1)
    y_obs_tensor = torch.tensor(y_obs_cd, dtype=torch.float32)
    y0_tensor = torch.tensor(y0, dtype=torch.float32).view(1, -1)

    state_net = SEIRDStateNet(hidden_dim=48)
    optimizer = optim.Adam(state_net.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_obs = state_net(tau_obs)
        pred_c = 1.0 - pred_obs[:, 0:1] - pred_obs[:, 1:2]
        pred_d = pred_obs[:, 4:5]
        loss_data = torch.mean((pred_c - y_obs_tensor[:, 0:1])**2) + torch.mean((pred_d - y_obs_tensor[:, 1:2])**2)

        tau_f = torch.rand((n_colloc, 1), requires_grad=True)
        u = state_net(tau_f)
        s, e, i, r, d = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4], u[:, 4:5]

        ds = torch.autograd.grad(s, tau_f, torch.ones_like(s), create_graph=True)[0]
        de = torch.autograd.grad(e, tau_f, torch.ones_like(e), create_graph=True)[0]
        di = torch.autograd.grad(i, tau_f, torch.ones_like(i), create_graph=True)[0]
        dr = torch.autograd.grad(r, tau_f, torch.ones_like(r), create_graph=True)[0]
        dd = torch.autograd.grad(d, tau_f, torch.ones_like(d), create_graph=True)[0]

        res_s = ds - T_max * (-beta_const * s * i)
        res_e = de - T_max * (beta_const * s * i - SIGMA * e)
        res_i = di - T_max * (SIGMA * e - (GAMMA + MU) * i)
        res_r = dr - T_max * (GAMMA * i)
        res_d = dd - T_max * (MU * i)
        loss_ode = torch.mean(res_s**2 + res_e**2 + res_i**2 + res_r**2 + res_d**2)

        pred_0 = state_net(torch.tensor([[0.0]], dtype=torch.float32))
        loss_ic = torch.mean((pred_0 - y0_tensor)**2)

        loss = 25.0 * loss_data + 1.0 * loss_ode + 25.0 * loss_ic
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
    print(f"  Latent Susceptible s(t) | Inversion RMSE: {s_rmse:.5f} | Relative Error: {s_rel:.2f}%")
    print(f"  Latent Incubating  e(t) | Inversion RMSE: {e_rmse:.5f} | Relative Error: {e_rel:.2f}%")

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

    save_plots(fig, "fig_e4_hidden_states")

# =============================================================================
# 8. EXPERIMENT E5: 2014 SIERRA LEONE EBOLA CALIBRATION & HELD-OUT FORECAST
# =============================================================================
def run_experiment_e5(epochs=2000, lr=2.0e-3, n_colloc=250):
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT E5: 2014 Sierra Leone Ebola Outbreak Calibration")
    print("="*70)
    set_seed(42)

    df = load_ebola_sierra_leone_data()
    N = float(POPULATION_SIERRA_LEONE)

    train_mask = (df['split'] == 'train')
    test_mask = (df['split'] == 'test')

    t_all = df['day'].values.astype(float)
    t_train = df.loc[train_mask, 'day'].values.astype(float)
    t_test = df.loc[test_mask, 'day'].values.astype(float)

    cases_train = df.loc[train_mask, 'cumulative_cases'].values.astype(float)
    deaths_train = df.loc[train_mask, 'cumulative_deaths'].values.astype(float)
    cases_test = df.loc[test_mask, 'cumulative_cases'].values.astype(float)
    deaths_test = df.loc[test_mask, 'cumulative_deaths'].values.astype(float)

    T_train = float(t_train[-1])
    scale_C = float(cases_train[-1])  # 949.0
    scale_D = float(deaths_train[-1]) # 402.0

    i0 = max(11.0 / N, 1e-6)
    e0 = max(16.0 / N, 1e-6)
    d0 = 5.0 / N
    r0 = 0.0
    s0 = 1.0 - (e0 + i0 + r0 + d0)
    y0_real = [s0, e0, i0, r0, d0]

    # --- Baseline 1: Constant-beta Least Squares ---
    print("\n--- Baseline 1: Constant-beta Least Squares ---")
    def ls_const(params):
        sim = solve_seird_reference(params[0], y0_real, t_train)
        sim_c = (1.0 - sim[:, 0] - sim[:, 1]) * N
        sim_d = sim[:, 4] * N
        return np.concatenate([(sim_c - cases_train) / (np.mean(cases_train) + 1.0),
                               (sim_d - deaths_train) / (np.mean(deaths_train) + 1.0)])

    res_const = least_squares(ls_const, x0=[0.25], bounds=(0.01, 1.5))
    beta_const_val = res_const.x[0]
    print(f"  Fitted Constant beta: {beta_const_val:.4f} day^-1 (R0 = {beta_const_val/(GAMMA+MU):.2f})")
    sim_const = solve_seird_reference(beta_const_val, y0_real, t_all)
    cases_const_test = (1.0 - sim_const[test_mask, 0] - sim_const[test_mask, 1]) * N
    deaths_const_test = sim_const[test_mask, 4] * N

    # --- Baseline 2: Piecewise-beta Least Squares ---
    print("\n--- Baseline 2: Piecewise-beta Least Squares ---")
    def pw_eval(params, t):
        b0, b1, b2 = params
        if np.isscalar(t):
            return b0 if t < 30.0 else (b1 if t < 60.0 else b2)
        b = np.empty_like(t)
        b[t < 30.0] = b0
        b[(t >= 30.0) & (t < 60.0)] = b1
        b[t >= 60.0] = b2
        return b

    def ls_pw(params):
        sim = solve_seird_reference(lambda t: pw_eval(params, t), y0_real, t_train)
        sim_c = (1.0 - sim[:, 0] - sim[:, 1]) * N
        sim_d = sim[:, 4] * N
        smooth = 0.5 * np.array([params[1] - params[0], params[2] - params[1]])
        return np.concatenate([(sim_c - cases_train) / (np.mean(cases_train) + 1.0),
                               (sim_d - deaths_train) / (np.mean(deaths_train) + 1.0),
                               smooth])

    res_pw = least_squares(ls_pw, x0=[0.30, 0.25, 0.20], bounds=(0.01, 1.5))
    params_pw = res_pw.x
    print(f"  Fitted Piecewise beta segments: b0={params_pw[0]:.4f}, b1={params_pw[1]:.4f}, b2={params_pw[2]:.4f}")
    sim_pw = solve_seird_reference(lambda t: pw_eval(params_pw, t), y0_real, t_all)
    cases_pw_test = (1.0 - sim_pw[test_mask, 0] - sim_pw[test_mask, 1]) * N
    deaths_pw_test = sim_pw[test_mask, 4] * N

    # --- Model 3: Characteristic Scale-Normalized PINN ---
    print("\n--- Model 3: Scale-Normalized PINN with Dynamic beta(t) ---")
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

        res_e = de - T_train * (beta_f * i_f - SIGMA * e_f)
        res_i = di - T_train * (SIGMA * e_f - (GAMMA + MU) * i_f)
        res_r = dr - T_train * (GAMMA * i_f)
        res_d = dd - T_train * (MU * i_f)
        loss_ode = torch.mean(res_e**2 + res_i**2 + res_r**2 + res_d**2)

        out_0 = state_net(torch.tensor([[0.0]]))
        loss_ic = torch.mean((out_0[:, 3:4] - torch.tensor([[deaths_train[0] / scale_C]]))**2)

        dbeta_dtau = torch.autograd.grad(beta_f, tau_f, torch.ones_like(beta_f), create_graph=True)[0]
        loss_smooth = torch.mean(dbeta_dtau**2)

        loss = 25.0 * loss_data + 1.0 * loss_ode + 10.0 * loss_ic + 0.02 * loss_smooth
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0 or (epoch + 1) == epochs:
            print(f"  PINN Epoch [{epoch+1:4d}/{epochs}] | Loss: {loss.item():.5e} (Data: {loss_data.item():.2e}, ODE: {loss_ode.item():.2e})")

    with torch.no_grad():
        out_train = state_net(tau_train).numpy()
        learned_beta = beta_net(tau_train).numpy().flatten()
        terminal_beta = float(learned_beta[-1])

    print(f"  Learned Dynamic beta(t): Day 0 = {learned_beta[0]:.4f} day^-1, Day 89 = {terminal_beta:.4f} day^-1")

    # Forward forecast from calibrated terminal state
    term_e = float(out_train[-1, 0]) * scale_C / N
    term_i = float(out_train[-1, 1]) * scale_C / N
    term_r = float(out_train[-1, 2]) * scale_C / N
    term_d = float(out_train[-1, 3]) * scale_C / N
    term_s = 1.0 - (term_e + term_i + term_r + term_d)

    sim_pinn = solve_seird_reference(terminal_beta, [term_s, term_e, term_i, term_r, term_d], t_test - t_test[0])
    cases_pinn_test = (sim_pinn[:, 1] + sim_pinn[:, 2] + sim_pinn[:, 3] + sim_pinn[:, 4]) * N
    deaths_pinn_test = sim_pinn[:, 4] * N

    # Quantitative evaluation metrics computation
    def compute_stats(y_true, y_pred):
        residuals = y_pred - y_true
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mape = float(np.mean(np.abs(residuals / y_true)) * 100.0)
        rel_err = float((mae / np.mean(y_true)) * 100.0)
        max_err = float(np.max(np.abs(residuals)))
        r2 = float(1.0 - np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2))
        return {'MAE': mae, 'RMSE': rmse, 'MAPE(%)': mape, 'RelErr(%)': rel_err, 'MaxErr': max_err, 'R2': r2}

    m_const_c = compute_stats(cases_test, cases_const_test)
    m_const_d = compute_stats(deaths_test, deaths_const_test)
    m_pw_c = compute_stats(cases_test, cases_pw_test)
    m_pw_d = compute_stats(deaths_test, deaths_pw_test)
    m_pinn_c = compute_stats(cases_test, cases_pinn_test)
    m_pinn_d = compute_stats(deaths_test, deaths_pinn_test)

    eval_results = [
        {'Model': 'Constant-β Least Squares', 'Target': 'Cases', **m_const_c},
        {'Model': 'Constant-β Least Squares', 'Target': 'Deaths', **m_const_d},
        {'Model': 'Piecewise-β Least Squares', 'Target': 'Cases', **m_pw_c},
        {'Model': 'Piecewise-β Least Squares', 'Target': 'Deaths', **m_pw_d},
        {'Model': 'PINN with Dynamic β(t)', 'Target': 'Cases', **m_pinn_c},
        {'Model': 'PINN with Dynamic β(t)', 'Target': 'Deaths', **m_pinn_d},
    ]
    df_eval = pd.DataFrame(eval_results)

    print("\n" + "="*85)
    print("HELD-OUT FORECAST EVALUATION METRICS (DAYS 90-119):")
    print("="*85)
    format_cols = {
        'MAE': '{:.2f}'.format,
        'RMSE': '{:.2f}'.format,
        'MAPE(%)': '{:.2f}%'.format,
        'RelErr(%)': '{:.2f}%'.format,
        'MaxErr': '{:.2f}'.format,
        'R2': '{:.4f}'.format
    }
    print(df_eval.to_string(index=False, formatters=format_cols))

    metrics_csv = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
    df_eval.to_csv(metrics_csv, index=False)
    print(f"\n[Metrics Saved] {metrics_csv}")

    # Plot Figure E5
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(t_train, cases_train, 'ko', markersize=3.5, label='Train Data (Days 0-89)')
    axes[0].plot(t_test, cases_test, 'ro', markersize=3.5, label='Held-Out Test Data (Days 90-119)')
    axes[0].plot(t_all, (1.0 - sim_const[:, 0] - sim_const[:, 1]) * N, 'b--', linewidth=1.8, label=f"Constant-β (RelErr={m_const_c['RelErr(%)']:.1f}%)")
    axes[0].plot(t_all, (1.0 - sim_pw[:, 0] - sim_pw[:, 1]) * N, 'g-', linewidth=2.0, label=f"Piecewise-β (RelErr={m_pw_c['RelErr(%)']:.1f}%)")
    
    cases_pinn_full = np.concatenate([(out_train[:, 1] + out_train[:, 2] + out_train[:, 3]) * scale_C, cases_pinn_test])
    axes[0].plot(t_all, cases_pinn_full, 'm-.', linewidth=2.0, label=f"PINN (RelErr={m_pinn_c['RelErr(%)']:.1f}%)")
    axes[0].axvline(x=89.5, color='gray', linestyle=':', label='Train/Test Split (Day 90)')
    axes[0].set_title("E5: Cumulative Diagnosed Cases C(t) Forecast", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Time (days since May 25, 2014)", fontsize=10)
    axes[0].set_ylabel("Cumulative Cases", fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper left', fontsize=8)

    axes[1].plot(t_train, deaths_train, 'ko', markersize=3.5, label='Train Data (Days 0-89)')
    axes[1].plot(t_test, deaths_test, 'ro', markersize=3.5, label='Held-Out Test Data (Days 90-119)')
    axes[1].plot(t_all, sim_const[:, 4] * N, 'b--', linewidth=1.8, label=f"Constant-β (RelErr={m_const_d['RelErr(%)']:.1f}%)")
    axes[1].plot(t_all, sim_pw[:, 4] * N, 'g-', linewidth=2.0, label=f"Piecewise-β (RelErr={m_pw_d['RelErr(%)']:.1f}%)")
    
    deaths_pinn_full = np.concatenate([out_train[:, 3] * scale_C, deaths_pinn_test])
    axes[1].plot(t_all, deaths_pinn_full, 'm-.', linewidth=2.0, label=f"PINN (RelErr={m_pinn_d['RelErr(%)']:.1f}%)")
    axes[1].axvline(x=89.5, color='gray', linestyle=':', label='Train/Test Split (Day 90)')
    axes[1].set_title("E5: Cumulative Recorded Deaths D(t) Forecast", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Time (days since May 25, 2014)", fontsize=10)
    axes[1].set_ylabel("Cumulative Deaths", fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper left', fontsize=8)

    save_plots(fig, "fig_e5_real_forecast")

    # Plot Comparative Bar Chart
    fig_bar, ax_bar = plt.subplots(1, 2, figsize=(13, 5))
    model_names = ['Constant-β', 'Piecewise-β', 'PINN (Dynamic)']
    w = 0.35
    x = np.arange(len(model_names))
    c_errs = [m_const_c['RelErr(%)'], m_pw_c['RelErr(%)'], m_pinn_c['RelErr(%)']]
    d_errs = [m_const_d['RelErr(%)'], m_pw_d['RelErr(%)'], m_pinn_d['RelErr(%)']]

    ax_bar[0].bar(x - w/2, [m_const_c['MAE'], m_pw_c['MAE'], m_pinn_c['MAE']], w, label='Cases MAE', color='#2b5c8f')
    ax_bar[0].bar(x + w/2, [m_const_d['MAE'], m_pw_d['MAE'], m_pinn_d['MAE']], w, label='Deaths MAE', color='#d95f02')
    ax_bar[0].set_xticks(x)
    ax_bar[0].set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax_bar[0].set_ylabel("Mean Absolute Error (Counts)", fontsize=10, fontweight='bold')
    ax_bar[0].set_title("(a) Out-of-Sample MAE Comparison", fontsize=11, fontweight='bold')
    ax_bar[0].grid(True, linestyle=':', alpha=0.6, axis='y')
    ax_bar[0].legend(loc='upper right', fontsize=9)

    ax_bar[1].bar(x - w/2, c_errs, w, label='Cases Rel Error (%)', color='#2b5c8f')
    ax_bar[1].bar(x + w/2, d_errs, w, label='Deaths Rel Error (%)', color='#d95f02')
    ax_bar[1].set_xticks(x)
    ax_bar[1].set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax_bar[1].set_ylabel("Relative Error (%)", fontsize=10, fontweight='bold')
    ax_bar[1].set_title("(b) Out-of-Sample Relative Error (%)", fontsize=11, fontweight='bold')
    ax_bar[1].grid(True, linestyle=':', alpha=0.6, axis='y')
    ax_bar[1].legend(loc='upper right', fontsize=9)
    for i in range(len(model_names)):
        ax_bar[1].text(x[i] - w/2, c_errs[i] + 1.2, f"{c_errs[i]:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax_bar[1].text(x[i] + w/2, d_errs[i] + 1.2, f"{d_errs[i]:.1f}%", ha='center', va='bottom', fontsize=9)

    save_plots(fig_bar, "fig_metrics_forecast_benchmark")
    return df_eval

# =============================================================================
# 9. CLI ENTRYPOINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Complete PINN Pipeline for SEIRD Model and Ebola Outbreak")
    parser.add_argument('--task', type=str, default='all', choices=['all', 'e2', 'e3', 'e4', 'e5', 'synthetic'],
                        help="Task: e2, e3, e4, e5, synthetic, or all")
    parser.add_argument('--epochs', type=int, default=1500)
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  RUNNING COMPLETE PINN PIPELINE [Task: {args.task}]")
    print(f"=======================================================")

    if args.task in ['all', 'synthetic', 'e2']:
        run_experiment_e2(epochs=min(args.epochs, 1200))
    if args.task in ['all', 'synthetic', 'e3']:
        run_experiment_e3(epochs=args.epochs)
    if args.task in ['all', 'synthetic', 'e4']:
        run_experiment_e4(epochs=args.epochs)
    if args.task in ['all', 'e5']:
        run_experiment_e5(epochs=max(args.epochs, 2000))

    print("\n[SUCCESS] Pipeline execution finished cleanly.")

if __name__ == '__main__':
    main()
