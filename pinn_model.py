"""
pinn_model.py
-------------
Physics-Informed Neural Network (PINN) architectures and loss formulations
for the SEIRD compartmental epidemic model with dynamic transmission rate beta(t).

Mathematical formulation:
    ds/dt = -beta(t) * s * i
    de/dt =  beta(t) * s * i - sigma * e
    di/dt =  sigma * e - (gamma + mu) * i
    dr/dt =  gamma * i
    dd/dt =  mu * i

Supports:
1. Dimensionless fractional representation (for synthetic benchmarks E2-E4)
2. Characteristic scale-normalized representation (for real-world outbreaks like E5
   where attack rate is << 1% of total population, preventing gradient vanishing).
"""

import torch
import torch.nn as nn
import numpy as np

SIGMA_DEFAULT = 1.0 / 9.4      # Incubation transition rate (day^-1)
GAMMA_DEFAULT = 1.0 / 7.0      # Recovery clearance rate (day^-1)
MU_DEFAULT = 0.1000            # Disease-induced mortality rate (day^-1)

class SEIRDStateNet(nn.Module):
    """
    Standard fractional state network for synthetic experiments (E2-E4).
    Maps dimensionless time tau in [0, 1] to (s, e, i, r, d) in (0, 1)^5.
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
    Characteristic scale-normalized network for real-world outbreaks with low attack rates (E5).
    Predicts positive scaled active counts: (e_tilde, i_tilde, r_tilde, d_tilde) in units of C_scale.
    This guarantees gradients are O(1) and prevents catastrophic vanishing gradients when I/N ~ 1e-5.
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
    Maps dimensionless time tau in [0, 1] to strictly positive transmission rate
    beta(t) > 0 using Softplus activation.
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


class SEIRDPINN:
    """
    Composite Physics-Informed Neural Network controller.
    """
    def __init__(self, 
                 state_net=None, 
                 beta_net=None, 
                 T_max=120.0, 
                 sigma=SIGMA_DEFAULT, 
                 gamma=GAMMA_DEFAULT, 
                 mu=MU_DEFAULT,
                 device='cpu'):
        self.device = torch.device(device)
        self.T_max = float(T_max)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.mu = float(mu)
        
        self.state_net = state_net.to(self.device) if state_net else SEIRDStateNet().to(self.device)
        self.beta_net = beta_net.to(self.device) if beta_net else None

    def compute_states(self, tau):
        return self.state_net(tau)

    def compute_beta(self, tau):
        if self.beta_net is not None:
            return self.beta_net(tau)
        raise ValueError("BetaNet is not initialized for dynamic beta computation.")

    def compute_ode_residuals(self, tau_colloc, constant_beta=None):
        tau = tau_colloc.clone().detach().requires_grad_(True)
        u = self.state_net(tau)
        s, e, i, r, d = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4], u[:, 4:5]

        ds = torch.autograd.grad(s, tau, torch.ones_like(s), create_graph=True)[0]
        de = torch.autograd.grad(e, tau, torch.ones_like(e), create_graph=True)[0]
        di = torch.autograd.grad(i, tau, torch.ones_like(i), create_graph=True)[0]
        dr = torch.autograd.grad(r, tau, torch.ones_like(r), create_graph=True)[0]
        dd = torch.autograd.grad(d, tau, torch.ones_like(d), create_graph=True)[0]

        if constant_beta is not None:
            beta = torch.tensor(constant_beta, dtype=torch.float32, device=self.device)
        else:
            beta = self.beta_net(tau)

        res_s = ds - self.T_max * (-beta * s * i)
        res_e = de - self.T_max * (beta * s * i - self.sigma * e)
        res_i = di - self.T_max * (self.sigma * e - (self.gamma + self.mu) * i)
        res_r = dr - self.T_max * (self.gamma * i)
        res_d = dd - self.T_max * (self.mu * i)

        return res_s, res_e, res_i, res_r, res_d, beta

    def compute_loss(self,
                     tau_obs,
                     y_obs,
                     tau_colloc,
                     y0,
                     obs_mode='full',
                     constant_beta=None,
                     w_data=15.0,
                     w_ode=1.0,
                     w_ic=15.0,
                     w_reg=0.01):
        pred_obs = self.state_net(tau_obs)
        if obs_mode == 'full':
            loss_data = torch.mean((pred_obs - y_obs) ** 2)
        elif obs_mode == 'cases_and_deaths':
            pred_cases = 1.0 - pred_obs[:, 0:1] - pred_obs[:, 1:2]
            pred_deaths = pred_obs[:, 4:5]
            loss_data = torch.mean((pred_cases - y_obs[:, 0:1]) ** 2) + torch.mean((pred_deaths - y_obs[:, 1:2]) ** 2)
        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}")

        res_s, res_e, res_i, res_r, res_d, beta = self.compute_ode_residuals(tau_colloc, constant_beta)
        loss_ode = torch.mean(res_s**2 + res_e**2 + res_i**2 + res_r**2 + res_d**2)

        tau_zero = torch.tensor([[0.0]], dtype=torch.float32, device=self.device)
        pred_ic = self.state_net(tau_zero)
        loss_ic = torch.mean((pred_ic - y0) ** 2)

        if self.beta_net is not None and w_reg > 0:
            tau_reg = tau_colloc.clone().detach().requires_grad_(True)
            beta_eval = self.beta_net(tau_reg)
            dbeta_dtau = torch.autograd.grad(beta_eval, tau_reg, torch.ones_like(beta_eval), create_graph=True)[0]
            loss_smooth = torch.mean(dbeta_dtau ** 2)
        else:
            loss_smooth = torch.tensor(0.0, device=self.device)

        total_loss = w_data * loss_data + w_ode * loss_ode + w_ic * loss_ic + w_reg * loss_smooth
        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_data': loss_data.item(),
            'loss_ode': loss_ode.item(),
            'loss_ic': loss_ic.item(),
            'loss_smooth': loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else loss_smooth
        }
