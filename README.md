# GANs for Time Series Generation

Generating realistic time series with Generative Adversarial Networks (GANs), and measuring how realistic they are.

This repository holds the code for a research project (PSC MAP13, "Deep learning et qualité de simulation de données") carried out at École polytechnique in 2024–2025. It contains:

- **`time_series_gan`**: a small PyTorch library implementing six GAN variants for time series (GAN, WGAN, TimeGAN, TimeWGAN, WTSGAN, XTSGAN), all behind the same interface.
- **Evaluation metrics** for generated data: sliced Wasserstein distance, discrete Fréchet distance, nearest-neighbour distances (INND / ONND) and a combined score.
- **Hyperparameter and architecture optimisation**: a Metropolis-Hastings search over network architectures and Bayesian optimisation (Optuna) of hyperparameters.
- **Experiment notebooks** comparing the models on three datasets.

The project started from the [GenHack 3](https://www.polytechnique.edu/en/genhack-3-hackathon-generative-modelling) challenge (generating maize yields at four French weather stations), then extended to data with strong time dependence.

## Repository layout

```
time_series_gan/        The model library (see "Models" below)
  WrapperGAN.py         Base class: data handling, metrics, plotting, Optuna hyperparameter search
  ModelGAN.py           Base class for point-by-point models (GAN, WGAN)
  ModelTimeGAN.py       Base class for sequence models with an autoencoder (TimeGAN, TimeWGAN)
  GAN.py, WGAN.py, TimeGAN.py, TimeWGAN.py, WTSGAN.py, XTSGAN.py
  architectures.py      Neural network building blocks (generator, discriminator, critic, embedder…)
  WrapperArchi.py       Configurable MLP used by the building blocks
  metrics.py            Evaluation metrics and the combined score
Optimisation_MH.py      Metropolis-Hastings search over architectures + hyperparameters
notebooks/              One notebook per (dataset, model) pair, with saved outputs
results_pkl/            Saved Metropolis-Hastings runs and the best architectures found
early_experiments/      First phase of the project on GenHack data (kept for reference)
```

## Installation

Requires **Python 3.12 or newer**.

```bash
git clone https://github.com/SimCPES/GAN_Time_Series_PSC.git
```

```bash
cd GAN_Time_Series_PSC
```

```bash
pip install -r requirements.txt
```

Run scripts and notebooks **from the repository root**: they import `time_series_gan` and read data files such as `data/data_gdp.csv` using paths relative to the root. In VS Code, set `"jupyter.notebookFileRoot": "${workspaceFolder}"`; with Jupyter, start it from the root and add the root to `sys.path` if needed.

## Quick start

Every model follows the same steps: load data, choose metrics, fit, generate.

```python
import pandas as pd
from time_series_gan import TimeWGAN
from time_series_gan.metrics import dWasserstein, dFrechet, ONND, score

data = pd.read_csv("your_data.csv")   # one column per variable, one row per time step

model = TimeWGAN()
model.set_data(data)
model.set_metrics({
    "dWasserstein": {"function": dWasserstein, "metric_args": {}},
    "dFrechet":     {"function": dFrechet,     "metric_args": {}},
    "score":        {"function": score,        "metric_args": {}},
})

# Hyperparameters and architecture found by optimisation on this dataset
params = {"lr_g": 0.0005526, "lr_c": 0.001336, "lr_e": 0.002986, "lr_r": 0.001287,
          "epochs": 150, "batch_size": 14, "latent_dim": 10, "hidden_dim": 30,
          "seq_length": 10, "n_critic": 4, "lambda_gp": 0.188}
archi = {"generator": {"architecture": "MLP", "layer_sizes": [64, 21, 30], "activation": "ReLU"},
         "critic":    {"architecture": "MLP", "layer_sizes": [80, 57, 1],  "activation": "ReLU"}}

results = model.fit(params=params, architectures=archi, verbose=True)  # verbose=True shows plots
print(results)            # {metric name: (train value, validation value)}

samples = model.generate_samples(100)
```

Main methods shared by all models:

| Method | What it does |
|---|---|
| `set_data(df)` | Loads a pandas DataFrame and splits it 80/20 into training and validation sets |
| `set_metrics(dict)` | Chooses the metrics computed after training |
| `fit(params, architectures, verbose)` | Builds the networks and trains them; returns the metrics on train and validation data |
| `fit_hyperparameters(param_ranges, n_trials)` | Searches hyperparameters with Optuna, optimising the score on validation data |
| `generate_samples(n)` | Generates `n` new time steps |
| `plot_series()`, `plot_compare_series()`, `plot_histograms()` | Compare real and generated data visually |

`model.parameters` lists the hyperparameters of each model and their default values.

## Models

| Model | Idea |
|---|---|
| **GAN** | Original GAN (Goodfellow et al., 2014): a generator and a discriminator trained against each other. Treats each time step independently. |
| **WGAN** | Wasserstein GAN with gradient penalty (Arjovsky et al., 2017; Gulrajani et al., 2017). The discriminator becomes a "critic" estimating the Wasserstein distance, which makes training more stable and reduces mode collapse. |
| **TimeGAN** | TimeGAN (Yoon et al., 2019): works on sequences, with an embedder/recovery autoencoder so the generator works in a latent space. Captures time dependence. |
| **TimeWGAN** | Our combination of TimeGAN and WGAN: TimeGAN's sequence and autoencoder approach, trained with a Wasserstein critic and gradient penalty. |
| **WTSGAN** | Wasserstein Time-Series GAN: a simpler TimeWGAN without the autoencoder. The generator outputs whole sequences directly. |
| **XTSGAN** | WTSGAN plus a "sequencer" network that learns to spot jumps where two generated sequences are joined, pushing the generator to produce sequences that connect smoothly. Experimental. |

## Evaluation metrics

Defined in `time_series_gan/metrics.py`. For all of them, lower is better.

- **`dWasserstein`**: sliced Wasserstein distance. Projects both datasets onto random directions and averages the 1D Wasserstein distances (1000 projections). Measures how well the overall distribution is reproduced.
- **`dFrechet`**: discrete Fréchet distance between the real and generated curves. Measures how well the shape of the series over time is reproduced.
- **`ONND`**: average distance from each real point to its nearest generated point. Measures how well generated data cover the real data.
- **`INND`**, **`ICD`**: nearest-neighbour distance from generated to real points, and a mode-collapse check (not used in the score).
- **`score`**: weighted combination used for optimisation, with each term rescaled to a similar order of magnitude:
  `score = 0.4 × 30 × dWasserstein + 0.4 × dFrechet + 0.2 × ONND × |G|^0.27` (|G| = number of generated points).

A standalone Dynamic Time Warping implementation is in `early_experiments/metrics/dynamic_time_warping.py`.

## Optimisation

- **Metropolis-Hastings (`Optimisation_MH.py`)**: explores architectures (number and size of layers, activation function) together with hyperparameters. Each step proposes a nearby configuration (add/remove a layer, change the activation, or rescale the hyperparameters), trains it, and accepts it with a probability that depends on the score change. `Metropolis_Hasting(beta, data, ModelClass, ite)` runs the search; `generer_resultats` saves all accepted steps to a pickle file; `reestimer` retrains a configuration several times to get a mean score with a 95% confidence interval.
- **Bayesian optimisation**: `model.fit_hyperparameters(...)` uses [Optuna](https://optuna.org/) to tune numeric hyperparameters for a fixed architecture.

## Datasets

The datasets are not included in this repository. The notebooks and `Optimisation_MH.py` expect them in a `data/` folder at the repository root:

| File | Description |
|---|---|
| `data/data_gdp.csv` | French annual GDP (64 values, normalised), from the World Bank. Short series with a clear trend. |
| `data/synthetic_data.csv` | 500 points of `f(x) = (x − 3.5)³ + 0.9·cos(2x) + ε` on [0, 8], with ε ~ N(−2, 3), normalised. |
| `data/data_genhack.csv` | GenHack 3 maize yields at 4 stations (columns `YIELD1`…`YIELD4`), keeping only years with normal weather. |

## Notebooks

`notebooks/` contains one notebook per dataset and model, named `<dataset>_<model>.ipynb`:

- dataset: `GH` = GenHack, `PIB` = French GDP, `Synth` = synthetic
- model: `GAN`, `WGAN`, `TGAN` = TimeGAN, `TWGAN` = TimeWGAN

Each notebook runs the Metropolis-Hastings search and/or Optuna, then trains the best configuration found and evaluates it (usually 20 evaluations to get a mean and standard deviation). Outputs are saved, so the results can be read without re-running.

## Saved results

- `results_pkl/results_<model>_<dataset>.pkl`: Metropolis-Hastings runs. Each file is a list of `(iteration, score, architecture, hyperparameters)` for every accepted step. Load with `charger_resultats` from `Optimisation_MH.py` (importing the model classes first).
- `results_pkl/optimised_architectures.txt`: best architectures and hyperparameters found for each model and dataset (notes in French).
- `results_pkl/old_results/`: earlier runs (WGAN, XTSGAN and first GAN searches).

## Results

Validation metrics after optimising each model (mean ± standard deviation over 20 evaluations). Lower is better.

**GenHack** (TimeWGAN not evaluated: training was too slow for a proper optimisation)

| Metric | GAN | WGAN | TimeGAN |
|---|---|---|---|
| Wasserstein | 0.246 ± 0.024 | 0.249 ± 0.036 | 0.420 ± 0.077 |
| Fréchet | 6.50 ± 0.75 | 6.99 ± 2.03 | 11.26 ± 2.14 |
| ONND | 1.070 ± 0.020 | 1.087 ± 0.025 | 0.818 ± 0.054 |
| Score | **5.57 ± 0.24** | 6.47 ± 0.81 | 8.61 ± 1.68 |

**French GDP**

| Metric | GAN | WGAN | TimeGAN | TimeWGAN |
|---|---|---|---|---|
| Wasserstein | 0.447 ± 0.062 | 0.369 ± 0.116 | 0.519 ± 0.119 | 0.373 ± 0.090 |
| Fréchet | 1.40 ± 0.15 | 1.41 ± 0.58 | 1.45 ± 0.20 | 1.40 ± 0.24 |
| ONND | 0.067 ± 0.036 | 0.154 ± 0.064 | 0.092 ± 0.080 | 0.042 ± 0.030 |
| Score | 5.64 ± 0.99 | 5.28 ± 1.62 | 5.57 ± 1.30 | **4.72 ± 1.09** |

**Synthetic data**

| Metric | GAN | WGAN | TimeGAN | TimeWGAN |
|---|---|---|---|---|
| Wasserstein | 0.188 ± 0.026 | 0.203 ± 0.024 | 0.246 ± 0.031 | 0.168 ± 0.049 |
| Fréchet | 1.87 ± 0.19 | 1.66 ± 0.36 | 2.05 ± 0.47 | 1.62 ± 0.30 |
| ONND | 0.032 ± 0.006 | 0.041 ± 0.010 | 0.011 ± 0.006 | 0.004 ± 0.003 |
| Score | 3.00 ± 0.29 | 3.56 ± 0.73 | 3.91 ± 0.49 | **2.66 ± 0.62** |

Main takeaways:

- GAN and WGAN reproduce the overall distribution well but ignore time dependence. They do best on GenHack, where years are independent.
- TimeWGAN gives the best compromise on data with a trend (GDP, synthetic): it reproduces the distribution better than TimeGAN while also capturing local time structure.
- Sequence models (TimeGAN, TimeWGAN) show jumps where generated sequences are joined. XTSGAN is a first attempt at fixing this.

## Authors

Sylvain Dehayem Kenfouo, Soline du Crest, Amine Fourati, Siméon Gheorghin, Elliot Thorel (École polytechnique).

Supervised by Jean Pachebat, Mouad Yachouti and Yushan Liu (CMAP, École polytechnique).

## References

- Goodfellow et al., *Generative Adversarial Networks*, 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Arjovsky, Chintala, Bottou, *Wasserstein GAN*, 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
- Gulrajani et al., *Improved Training of Wasserstein GANs*, 2017. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
- Yoon, Jarrett, van der Schaar, *Time-series Generative Adversarial Networks*, NeurIPS 2019.
- Bonneel et al., *Sliced and Radon Wasserstein Barycenters of Measures*, 2015.
- Eiter, Mannila, *Computing Discrete Fréchet Distance*, 1994.
- Leznik et al., *[SoK] The Great GAN Bake Off*, 2022.
- Stenger et al., *Evaluation is key: a survey on evaluation measures for synthetic time series*, 2024.
- Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, 2019.
