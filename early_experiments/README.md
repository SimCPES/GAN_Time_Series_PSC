# Early experiments (GenHack phase)

First phase of the project, before the `time_series_gan` library existed. Everything here works on the [GenHack 3](https://www.polytechnique.edu/en/genhack-3-hackathon-generative-modelling) maize-yield data with a simple GAN. The code is kept as written, for reference and to back the corresponding parts of the report.

## Running

Run scripts **from this folder** (`early_experiments/`), since they import each other with paths like `grid_search.functions...` and read data from `data/`.

The GenHack data are **not included**. To run these scripts, put the GenHack 3 station files in `early_experiments/data/`. Two naming conventions are used:

- `1-station_49.csv`, `2-station_80.csv`, `3-station_40.csv`, `4-station_63.csv`: used by `extract_data_part1.py` and `stationarity_tests.ipynb`. Running `extract_data_part1.py` creates `data/round1.npy`, used by the grid search and Bayesian optimisation scripts.
- `station_40.csv`, `station_49.csv`, `station_63.csv`, `station_80.csv`: used by `metropolis_hastings/`.

## Contents

| Path | What it is |
|---|---|
| `GANs/GAN_soline.ipynb` | First GAN implementation on GenHack data |
| `GANs/fElliot.py` | Simple configurable GAN used by the Bayesian optimisation scripts |
| `extract_data_part1.py` | Builds the training set (yields in normal weather conditions) from the station files |
| `stationarity_tests.ipynb` | ADF and KPSS stationarity tests on the yield series |
| `grid_search/` | Grid search over GAN hyperparameters (`gridSearch1-3.py`), plotting scripts (`mefFigure*.py`) and results (`results1-3/`: `.npy` scores, `.png` heatmaps, `.txt` best values) |
| `bayesian_optimization/` | Bayesian optimisation helper (`bayesianOpt.py`), logs of runs (`records*.txt`) and the Optuna study database (`db.sqlite3`) |
| `trainFunction.py` | Bayesian optimisation with the `bayesian-optimization` package |
| `test_optuna.py` | Same search with Optuna |
| `metropolis_hastings/` | First Metropolis-Hastings search over GAN architectures (`Metropolis.py`), with model, data loading and plotting helpers. `figures/` shows the Wasserstein distance and number of layers over 3000 iterations |
| `metrics/` | First standalone versions of the metrics, plus Dynamic Time Warping (`dynamic_time_warping.py`) |
| `metric_calibration.py` | Studies how the Wasserstein distance converges with the number of projections, and how the metrics scale with sample size |
| `metric_normalisation_plot.py` | Plots the measured values used to choose the normalisation factors in the score |
