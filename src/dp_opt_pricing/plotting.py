
from typing import Callable, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

def plot_value_surface(f_hat, X_grid: np.ndarray, t_label: str, ax: plt.Axes) -> None:
    X1_u = np.unique(X_grid[:, 0])
    X2_u = np.unique(X_grid[:, 1])
    X1, X2 = np.meshgrid(X1_u, X2_u)
    grid = np.c_[X1.ravel(), X2.ravel()]
    Z = f_hat(grid).reshape(X1.shape)
    ax.plot_surface(X1, X2, Z, edgecolor='k', alpha=0.9, linewidth=0.3)
    ax.set_title(f"t = {t_label}", fontsize=12, fontweight='bold')
    ax.set_xlabel("$X_t^1$")
    ax.set_ylabel("$X_t^2$")
    ax.set_zlabel("$V_t(X)$")
    ax.view_init(elev=25, azim=135)

def save_value_panel(collected: Dict[int, Callable],
                     payoff_fn: Callable,
                     S_range: Tuple[float, float] = (50, 150),
                     points: int = 30,
                     filename: str = "value_function_panel.pdf") -> None:
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        pass
    grid = np.linspace(S_range[0], S_range[1], points)
    X_plot = np.array(np.meshgrid(grid, grid)).T.reshape(-1, 2)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 10))
    all_times = sorted(collected.keys()) + ['payoff']
    for i, t_plot in enumerate(all_times):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        if t_plot == 'payoff':
            plot_value_surface(lambda x: payoff_fn(np.atleast_2d(x)), X_plot, "T", ax)
        else:
            plot_value_surface(collected[t_plot], X_plot, str(t_plot), ax)
    plt.tight_layout()
    plt.savefig(filename, format="pdf", dpi=300)
    plt.close(fig)
    print(f"Saved panel to {filename}")
