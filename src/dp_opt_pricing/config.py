
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch

@dataclass
class Config:
    cross_validation: bool = False
    plot_panels: bool = True
    d: int = 2
    use_cuda: Optional[bool] = None
    repeats: int = 1

    T: int = 9
    r: float = 0.05
    K: float = 100.0
    sigma_scalar: float = 0.2
    S0_scalar: float = 100.0
    delta_t: float = 1.0
    n: int = 500
    M: int = 150
    M_nystrom: int = 250

    alphas: Tuple[float, ...] = (1e-6, )
    lengthscales: Tuple[float, ...] = (100., 110., 120., 130.)

    rho: float = 0.2
    payoff_type: str = 'max_call'

    def derived(self):
        dt = self.delta_t / self.T
        sigma = np.ones(self.d) * self.sigma_scalar
        S0 = np.ones(self.d) * self.S0_scalar
        rho_matrix = (1 - self.rho) * np.eye(self.d) + self.rho * np.ones((self.d, self.d))
        cov = np.outer(sigma, sigma) * rho_matrix
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        gammas = tuple(1.0 / (2.0 * (l ** 2)) for l in self.lengthscales)
        return dt, sigma, S0, rho_matrix, cov, gammas
