
from typing import Tuple
import math
import numpy as np
import torch
from falkon import Falkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions

def fit_best_falkon(X: np.ndarray, Y: np.ndarray,
                    alphas, gammas,
                    M_nystrom: int,
                    use_cuda: bool) -> Tuple[Falkon, Tuple[float, float]]:
    n = X.shape[0]
    idx = np.random.permutation(n)
    n_val = int(0.3 * n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    best_score = np.inf
    best_model = None
    best_params = (None, None)

    for alpha in alphas:
        for gamma in gammas:
            lengthscale = math.sqrt(1.0 / (2.0 * gamma))
            kernel = GaussianKernel(sigma=lengthscale)
            options = FalkonOptions(use_cpu=not use_cuda)
            model = Falkon(kernel=kernel, penalty=alpha, M=M_nystrom, options=options)
            model.fit(
                torch.from_numpy(np.atleast_2d(X_tr).astype(np.float64)),
                torch.from_numpy(Y_tr.astype(np.float64).reshape(-1, 1))
            )
            preds = model.predict(torch.from_numpy(np.atleast_2d(X_val).astype(np.float64)))
            preds_np = preds.detach().cpu().numpy().reshape(-1)
            score = np.mean((preds_np - Y_val)**2)
            if score < best_score:
                best_score = score
                best_model = model
                best_params = (alpha, gamma)

    assert best_model is not None, "Model selection failed."
    return best_model, best_params

def make_predictor(m: Falkon):
    def _call(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float64)
        out = m.predict(torch.from_numpy(x))
        return out.detach().cpu().numpy().reshape(-1)
    return _call
