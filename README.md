
# AOF — American-style Option Pricing with FALKON (KRR-DP)

This repository implements the algorithm from the paper
**"Error Propagation in Stochastic Optimal Control, with Applications to American Options Pricing."**
It includes both a clean **Jupyter notebook** and an **installable Python package** with a CLI.

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
