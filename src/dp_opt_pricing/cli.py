
import argparse
from .config import Config
from .training import run_training
from .plotting import save_value_panel
from .hindsight import hindsight_max_exercise

def main():
    p = argparse.ArgumentParser(description="American option pricing with FALKON (KRR-DP)")
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--payoff", type=str, default="max_call", choices=["max_call", "geo_put"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--M", type=int, default=150)
    p.add_argument("--m-nystrom", type=int, default=250)
    p.add_argument("--lengthscales", type=float, nargs="+", default=[100,110,120,130])
    p.add_argument("--alphas", type=float, nargs="+", default=[1e-6])
    p.add_argument("--plot-panels", action="store_true")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    cfg = Config(
        d=args.d,
        payoff_type=args.payoff,
        repeats=args.repeats,
        n=args.n,
        M=args.M,
        M_nystrom=args.m_nystrom,
        lengthscales=tuple(args.lengthscales),
        alphas=tuple(args.alphas),
        plot_panels=args.plot_panels,
        use_cuda=False if args.no_cuda else None,
    )

    summary, panels, payoff_fn, aux = run_training(cfg, seed=args.seed)
    dt, sigma, S0, L = aux

    print("\n--- Final Estimate Summary ---")
    print(f"Mean estimated price: {summary['mean_price']:.6f}")
    print(f"Standard error: {summary['std_error']:.6f}")
    lo, hi = summary['ci_95']
    print(f"95% CI: ({lo:.6f}, {hi:.6f})")
    print(f"Total time: {summary['duration_sec']:.2f} s")

    h_mean, h_se, t_mean, t_se = hindsight_max_exercise(S0, cfg.T, dt, cfg.r, sigma, L, cfg.K, n_paths=1000, payoff_fn=payoff_fn)
    print("\n--- Hindsight Evaluation ---")
    print(f"Mean payoff (hindsight upper bound): {h_mean:.6f} ± {1.96 * h_se:.6f}")
    print(f"Mean exercise time: {t_mean:.3f} ± {1.96 * t_se:.3f}")

    if cfg.plot_panels and cfg.d == 2 and panels:
        save_value_panel(collected=panels, payoff_fn=payoff_fn, filename="value_function_panel.pdf")

if __name__ == "__main__":
    main()
