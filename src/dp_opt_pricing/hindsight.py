
from typing import Callable, Tuple
import math
import numpy as np
from .gbm import gbm_step_correlated

def hindsight_max_exercise(S0: np.ndarray, T: int, dt: float, r: float,
                           sigma: np.ndarray, L: np.ndarray, K: float, n_paths: int = 1000,
                           payoff_fn: Callable[[np.ndarray], np.ndarray] = None) -> Tuple[float, float, float, float]:
    if payoff_fn is None:
        raise ValueError("payoff_fn is required")
    disc = np.exp(-r * dt * np.arange(0, T + 1))

    S_paths = np.tile(S0.reshape(1, -1), (n_paths, 1))
    payoffs = np.zeros(n_paths)
    t_star = np.zeros(n_paths)
    for i in range(n_paths):
        S_t = np.zeros((T + 1, S0.shape[0]))
        S_t[0] = S_paths[i]
        for t in range(T):
            S_t[t + 1] = gbm_step_correlated(S_t[t:t+1], dt, r, sigma, L)[0]
        P_t = payoff_fn(S_t)
        DP_t = disc * P_t
        idx = int(np.argmax(DP_t))
        payoffs[i] = DP_t[idx]
        t_star[i] = idx

    h_mean = float(np.mean(payoffs))
    h_se = float(np.std(payoffs, ddof=1) / math.sqrt(n_paths))
    t_mean = float(np.mean(t_star))
    t_se = float(np.std(t_star, ddof=1) / math.sqrt(n_paths))
    return h_mean, h_se, t_mean, t_se
