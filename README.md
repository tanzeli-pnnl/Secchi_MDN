# Secchi MDN Reproduction

This repository now contains a clean Python implementation of the workflow used by Maciel et al. (2023) to train global Mixture Density Network models for Secchi depth from the provided Landsat-simulated reflectance spreadsheets.

## What It Does

- Reads the supplied `rrs_tm_v3.xlsx`, `rrs_etm_v3.xlsx`, and `rrs_oli_v3.xlsx` files.
- Standardizes the sensor tables into a common schema.
- Builds a Maciel-style feature set using raw reflectance bands plus ratio features.
- Trains grouped Monte Carlo MDN models with:
  - log-transformed Secchi targets
  - robust input scaling
  - grouped splits by `local` and acquisition month
  - ensemble prediction via the median across multiple MDN members
- Saves trained weights, preprocessing objects, per-run metrics, and prediction tables.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Train all three sensor-specific global models with the paper-style fixed architecture:

```bash
python main.py train \
  --dataset-dir "/Users/tanz151/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Others/EBSD_LDRD_FY26/Data/Maciel_etal_2023/Simulated Dataset" \
  --output-dir outputs
```

Train only the OLI model:

```bash
python main.py train \
  --dataset-dir "/Users/tanz151/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Others/EBSD_LDRD_FY26/Data/Maciel_etal_2023/Simulated Dataset" \
  --sensor oli \
  --output-dir outputs
```

Useful knobs:

- `--fit-mode final` to train the recommended architecture directly
- `--fit-mode monte_carlo` to keep repeated split evaluation
- `--monte-carlo-runs 50`
- `--ensemble-size 10`
- `--n-mix 5`
- `--hidden-dims 100 100 100 100 100`
- `--learning-rate 0.001`
- `--weight-decay 0.001`
- `--epsilon 0.001`
- `--prediction-mode top`
- `--include-coastal` if you want to experiment with the OLI coastal band

## Notes On Reproduction

- The defaults now match the authors' recommended MDN architecture: `n_mix=5`, `hidden=[100,100,100,100,100]`, `l2=0.001`, `lr=0.001`, `epsilon=0.001`.
- The original public scripts mostly use three visible bands for TM, ETM+, and OLI, with ratio features enabled.
- The original repository wraps a larger TensorFlow MDN package. This repo reimplements the training path in PyTorch so it is easier to understand and maintain.
- The grouped split key here is reconstructed as `local + year-month(date)`, which is the closest match to the `local_year_month` grouping used in the published scripts.
