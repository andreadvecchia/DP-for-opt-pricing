
from typing import Callable
import math
import numpy as np

def gbm_step_correlated(x: np.ndarray, dt: float, r: float, sigma: np.ndarray, L: np.ndarray) -> np.ndarray:
    z = np.random.randn(*x.shape)
    correlated_z = z @ L.T
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = correlated_z * math.sqrt(dt)
    return x * np.exp(drift + diffusion)

def simulate_continuation(x: np.ndarray,
                          V_next: Callable[[np.ndarray], np.ndarray],
                          r: float, dt: float,
                          sigma: np.ndarray, L: np.ndarray, M: int) -> np.ndarray:
    N = x.shape[0]
    vals = np.zeros(N)
    for i in range(N):
        x_next = np.tile(x[i], (M, 1))
        x_next = gbm_step_correlated(x_next, dt, r, sigma, L)
        vals[i] = np.mean(V_next(x_next))
    return np.exp(-r * dt) * vals

def sample_state_at_time(t: int, S0: np.ndarray, sigma: np.ndarray, r: float, dt: float, n: int, d: int) -> np.ndarray:
    jitter = 1.0 if t == 0 else 1e-6
    drift = (r - 0.5 * sigma**2) * t * dt
    log_mean = np.log(S0) + drift
    return np.random.lognormal(mean=log_mean, sigma=sigma * np.sqrt(t * dt + jitter), size=(n, d))
