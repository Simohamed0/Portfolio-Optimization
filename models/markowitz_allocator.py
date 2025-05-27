import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


# ---- Minimal Mu Predictor ----
class MinimalMuPredictionNet(nn.Module):
    def __init__(self, input_dim: int, n_assets: int, hidden_dims=[32, 16]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dims[-1], n_assets))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---- Differentiable Markowitz Solver ----
class DiffMarkowitzLayer(nn.Module):
    def __init__(self, n_assets, sigma_np):
        super().__init__()
        w = cp.Variable(n_assets)
        p = cp.Parameter(n_assets)
        Sigma = cp.Constant(sigma_np.astype(np.float32))
        objective = cp.Minimize(0.5 * cp.quad_form(w, Sigma) + p @ w)
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        if not problem.is_dpp():
            raise ValueError("Problem is not DPP.")

        self.layer = CvxpyLayer(problem, parameters=[p], variables=[w])

    def forward(self, p_batch):
        return self.layer(p_batch)[0]


# ---- Portfolio Optimizer ----
class MarkowitzPortfolioOptimizer(nn.Module):
    def __init__(self, input_dim, n_assets, sigma_np, gamma=1.0):
        super().__init__()
        self.mu_net = MinimalMuPredictionNet(input_dim, n_assets)
        self.markowitz = DiffMarkowitzLayer(n_assets, sigma_np)
        self.gamma = torch.tensor(gamma, dtype=torch.float32)

    def forward(self, x):
        mu = self.mu_net(x)
        p = -self.gamma * mu
        weights = self.markowitz(p)
        return weights, mu


# ---- Quick Test ----
if __name__ == "__main__":
    torch.manual_seed(0)
    n_assets = 5
    sigma = np.diag([0.1, 0.2, 0.15, 0.25, 0.3])
    print("Sigma matrix:\n", sigma)

    optimizer = MarkowitzPortfolioOptimizer(
        input_dim=10, n_assets=n_assets, sigma_np=sigma
    )
    optimizer.eval()

    x = torch.randn(1, 10, requires_grad=True)
    weights, mu = optimizer(x)
    print("Weights:", weights)
    loss = -weights[0, 0]
    loss.backward()
    print("Grad x:", x.grad)
