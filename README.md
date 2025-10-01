
# Dynamic Programming for Option Pricing (DP-for-opt-pricing)

Code accompanying the work:

**Error Propagation in Stochastic Optimal Control, with Applications to American Options Pricing**  
A. Della Vecchia, D. Filipović (under submission at ICLR, 2025)

📄 [Paper (PDF)](docs/DellaVecchia_error.pdf)
🔗 [Google Scholar](https://scholar.google.it/citations?view_op=view_citation&hl=it&user=aaeUheEAAAAJ&citation_for_view=aaeUheEAAAAJ:zYLM7Y9cAGgC)


---

## Overview
The project implements **dynamic programming with regression-based Monte Carlo** for solving optimal stopping problems such as American option pricing.

- At each time step in the backward recursion, the continuation value is estimated by **Monte Carlo simulation** of next-step asset prices.
- These continuation values are then approximated by **kernel ridge regression (KRR)** in a reproducing kernel Hilbert space (RKHS).
- The regression step provides a functional approximation of the value function, which can be queried at arbitrary states.
- By iterating this **Bellman recursion with regression** from maturity back to time 0, we obtain an approximation of the optimal stopping value.
- The implementation uses the **FALKON solver** for scalable KRR, enabling efficient training with Nyström subsampling.

This algorithm is called **KRR-DP** in the paper: Kernel Ridge Regression + Dynamic Programming.

The project includes both a clean **Jupyter notebook** and an **installable Python package** with a CLI.


![Value function surface](docs/put.png)

## Structure
```
DP-for-opt-pricing/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ notebooks/
│  └─ american_option_falkon.ipynb
├─ src/
│  └─ dp_opt_pricing/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ payoffs.py
│     ├─ gbm.py
│     ├─ model.py
│     ├─ plotting.py
│     ├─ hindsight.py
│     └─ cli.py
├─ scripts/
│  └─ train.py
├─ tests/
│  └─ test_import.py
└─ examples/
   └─ quickstart.sh
```

## Install
```bash
pip install -e .
# optional: pip install -r requirements.txt
```

## CLI usage
```bash
dp_opt_pricing-train --d 2 --payoff max_call --repeats 1 --plot-panels
dp_opt_pricing-train --no-cuda  # force CPU
dp_opt_pricing-train -h         # help
```

## Run from script
```bash
python scripts/train.py --d 2 --payoff geo_put --repeats 1 --plot-panels
```

## Notebook
Open `notebooks/american_option_falkon.ipynb` for a recruiter-friendly, annotated demo mapping cells to the paper (equations & Algorithm 1).

## License
MIT
