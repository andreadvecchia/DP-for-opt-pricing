
from typing import Callable
import numpy as np

def call_payoff(x: np.ndarray, K: float) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.maximum(x[:, 0] - K, 0.0)

def max_call_payoff(x: np.ndarray, K: float) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.maximum(np.max(x, axis=1) - K, 0.0)

def geometric_put_payoff(x: np.ndarray, K: float) -> np.ndarray:
    x = np.atleast_2d(x)
    geo_mean = np.exp(np.mean(np.log(np.maximum(x, 1e-12)), axis=1))
    return np.maximum(K - geo_mean, 0.0)

def make_payoff(kind: str, K: float) -> Callable[[np.ndarray], np.ndarray]:
    if kind == 'max_call':
        def _pay(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            if x.shape[1] > 1:
                return max_call_payoff(x, K)
            return call_payoff(x, K)
        return _pay
    elif kind == 'geo_put':
        return lambda x: geometric_put_payoff(x, K)
    else:
        raise ValueError(f"Unknown payoff_type: {kind}")
