import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp

from data_pipeline import PROCESSED_DATA_PATH, POPULATION_SIERRA_LEONE
from pinn_model import SIGMA_DEFAULT, GAMMA_DEFAULT, MU_DEFAULT

df = pd.read_csv(PROCESSED_DATA_PATH)
N = float(POPULATION_SIERRA_LEONE)
train_mask = (df['split'] == 'train')
test_mask = (df['split'] == 'test')

t_train = df.loc[train_mask, 'day'].values.astype(float)
t_test = df.loc[test_mask, 'day'].values.astype(float)
cases_train = df.loc[train_mask, 'cumulative_cases'].values.astype(float)
deaths_train = df.loc[train_mask, 'cumulative_deaths'].values.astype(float)
cases_test = df.loc[test_mask, 'cumulative_cases'].values.astype(float)
deaths_test = df.loc[test_mask, 'cumulative_deaths'].values.astype(float)

T_train = float(t_train[-1])
scale_C = float(cases_train[-1])  # 949.0
scale_D = float(deaths_train[-1]) # 402.0

# Normalized targets: [0, 1] relative to max in training
c_norm = torch.tensor(cases_train / scale_C, dtype=torch.float32).view(-1, 1)
d_norm = torch.tensor(deaths_train / scale_D, dtype=torch.float32).view(-1, 1)
tau_train = torch.tensor(t_train / T_train, dtype=torch.float32).view(-1, 1)

class ScaledSEIRDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 4) # e_tilde, i_tilde, r_tilde, d_tilde (all positive, scale_C)
        )
        self.softplus = nn.Softplus()
    def forward(self, tau):
        return self.softplus(self.net(tau))

class BetaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.softplus = nn.Softplus()
    def forward(self, tau):
        return self.softplus(self.net(tau))

net = ScaledSEIRDNet()
bnet = BetaNet()
opt = optim.Adam(list(net.parameters()) + list(bnet.parameters()), lr=2e-3)

for epoch in range(2000):
    opt.zero_grad()
    
    # 1. Data loss
    out = net(tau_train)
    e_t, i_t, r_t, d_t = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
    
    pred_c = (i_t + r_t + d_t) # in units of scale_C
    pred_d = d_t * (scale_C / scale_D) # in units of scale_D
    
    loss_data = torch.mean((pred_c - c_norm)**2) + torch.mean((pred_d - d_norm)**2)
    
    # 2. Physics ODE residual on collocation points
    tau_f = torch.rand((200, 1), requires_grad=True)
    out_f = net(tau_f)
    e_f, i_f, r_f, d_f = out_f[:, 0:1], out_f[:, 1:2], out_f[:, 2:3], out_f[:, 3:4]
    beta_f = bnet(tau_f)
    
    de = torch.autograd.grad(e_f, tau_f, torch.ones_like(e_f), create_graph=True)[0]
    di = torch.autograd.grad(i_f, tau_f, torch.ones_like(i_f), create_graph=True)[0]
    dr = torch.autograd.grad(r_f, tau_f, torch.ones_like(r_f), create_graph=True)[0]
    dd = torch.autograd.grad(d_f, tau_f, torch.ones_like(d_f), create_graph=True)[0]
    
    res_e = de - T_train * (beta_f * i_f - SIGMA_DEFAULT * e_f)
    res_i = di - T_train * (SIGMA_DEFAULT * e_f - (GAMMA_DEFAULT + MU_DEFAULT) * i_f)
    res_r = dr - T_train * (GAMMA_DEFAULT * i_f)
    res_d = dd - T_train * (MU_DEFAULT * i_f)
    
    loss_ode = torch.mean(res_e**2 + res_i**2 + res_r**2 + res_d**2)
    
    out_0 = net(torch.tensor([[0.0]]))
    loss_ic = torch.mean((out_0[:, 3:4] - torch.tensor([[deaths_train[0] / scale_C]]))**2)
    
    loss = 20.0 * loss_data + 1.0 * loss_ode + 10.0 * loss_ic
    loss.backward()
    opt.step()

with torch.no_grad():
    out_train = net(tau_train).numpy()
    pred_cases_train = (out_train[:, 1] + out_train[:, 2] + out_train[:, 3]) * scale_C
    pred_deaths_train = out_train[:, 3] * scale_C
    learned_beta = bnet(tau_train).numpy().flatten()
    print("Train Cases Error (MAE):", np.mean(np.abs(pred_cases_train - cases_train)))
    print("Train Deaths Error (MAE):", np.mean(np.abs(pred_deaths_train - deaths_train)))
    print(f"Learned beta: initial {learned_beta[0]:.3f}, terminal {learned_beta[-1]:.3f}")

term_e = float(out_train[-1, 0]) * scale_C / N
term_i = float(out_train[-1, 1]) * scale_C / N
term_r = float(out_train[-1, 2]) * scale_C / N
term_d = float(out_train[-1, 3]) * scale_C / N
term_s = 1.0 - (term_e + term_i + term_r + term_d)
term_beta = float(learned_beta[-1])

print(f"Terminal state counts: E={term_e*N:.1f}, I={term_i*N:.1f}, D={term_d*N:.1f}, beta={term_beta:.3f}")

def seird_rhs(t, y):
    s, e, i, r, d = y
    return [-term_beta*s*i, term_beta*s*i - SIGMA_DEFAULT*e, SIGMA_DEFAULT*e - (GAMMA_DEFAULT+MU_DEFAULT)*i, GAMMA_DEFAULT*i, MU_DEFAULT*i]

sol = solve_ivp(seird_rhs, (0, t_test[-1]-t_test[0]), [term_s, term_e, term_i, term_r, term_d], t_eval=t_test-t_test[0])
test_sim = sol.y.T
pred_cases_test = (test_sim[:, 1] + test_sim[:, 2] + test_sim[:, 3] + test_sim[:, 4]) * N
pred_deaths_test = test_sim[:, 4] * N

print("FORECAST TEST METRICS:")
print("Test Cases MAE:", np.mean(np.abs(pred_cases_test - cases_test)))
print("Test Cases RelErr (%):", np.mean(np.abs(pred_cases_test - cases_test)) / np.mean(cases_test) * 100)
print("Test Deaths MAE:", np.mean(np.abs(pred_deaths_test - deaths_test)))
print("Test Deaths RelErr (%):", np.mean(np.abs(pred_deaths_test - deaths_test)) / np.mean(deaths_test) * 100)
