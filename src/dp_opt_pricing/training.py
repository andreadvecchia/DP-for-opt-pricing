
from typing import Callable, Dict, Tuple
import math, time
import numpy as np
from scipy.stats import norm
import torch

from .config import Config
from .payoffs import make_payoff
from .gbm import sample_state_at_time, simulate_continuation
from .model import fit_best_falkon, make_predictor

def run_training(cfg: Config, seed: int = 7, collect_times=(0,2,4,6,8)):
    import falkon.utils.devices
    try:
        falkon.utils.devices.__COMP_DATA = {}
    except Exception:
        pass

    dt, sigma, S0, rho_matrix, cov, gammas = cfg.derived()
    L = np.linalg.cholesky(cov)
    payoff = make_payoff(cfg.payoff_type, cfg.K)

    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    final_estimates = []
    panels: Dict[int, Callable[[np.ndarray], np.ndarray]] = {}
    start_all = time.time()

    for run in range(cfg.repeats):
        print(f"\n--- Repetition {run + 1} / {cfg.repeats} ---")
        models: Dict[int, Callable[[np.ndarray], np.ndarray]] = {}

        V_hat: Callable[[np.ndarray], np.ndarray] = lambda x: payoff(np.atleast_2d(x))

        for t in reversed(range(cfg.T)):
            X = sample_state_at_time(t, S0, sigma, cfg.r, dt, cfg.n, cfg.d)
            Y_cont = simulate_continuation(X, V_hat, cfg.r, dt, sigma, L, cfg.M)
            Y = np.maximum(Y_cont, payoff(X))

            model, (alpha, gamma) = fit_best_falkon(
                X, Y, cfg.alphas, gammas, cfg.M_nystrom, cfg.use_cuda
            )
            lengthscale = math.sqrt(1.0 / (2.0 * gamma))
            print(f"t={t:2d}  best alpha={alpha:.2e}  lengthscale={lengthscale:.2f}")

            V_hat = make_predictor(model)
            models[t] = V_hat
            if cfg.d == 2 and t in collect_times:
                panels[t] = V_hat

        price_0 = float(models[0](S0.reshape(1, -1))[0])
        final_estimates.append(price_0)
        print(f"Estimated price at t=0: {price_0:.6f}")

    duration = time.time() - start_all

    final_estimates = np.array(final_estimates, dtype=float)
    mean_price = float(np.mean(final_estimates))
    std_error = float(np.std(final_estimates, ddof=1) / math.sqrt(max(1, len(final_estimates)))) if len(final_estimates) > 1 else 0.0
    z = float(norm.ppf(1 - (1 - 0.95) / 2))
    ci_low = mean_price - z * std_error
    ci_high = mean_price + z * std_error

    summary = {
        "mean_price": mean_price,
        "std_error": std_error,
        "ci_95": (ci_low, ci_high),
        "duration_sec": duration
    }
    return summary, panels, payoff, (dt, sigma, S0, L)
