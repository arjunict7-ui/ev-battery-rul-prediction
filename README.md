# EV Battery Life Prediction Using Machine Learning

Predicting the Remaining Useful Life (RUL) of EV lithium-ion battery cells using classical machine learning regression models, based on the NASA Prognostics Center of Excellence battery dataset.

> Mini Project (INT300), School of Computing, SASTRA Deemed to be University.
> By Arjun Kumar S, Sanjai R, Yuvaraj K — under the guidance of Prof. Arun Prasad P B.

## Overview

Electric vehicle batteries degrade over charge-discharge cycles due to temperature variation, internal resistance changes, and repeated usage. This project builds a supervised ML pipeline to predict how much usable life (RUL) remains in a battery cell before it hits End-of-Life (80% of rated capacity).

Base paper: Swain S, Pattnaik S, Mohanty P and Nanda S (2024), *Machine learning-based remaining useful life prediction for electric vehicle batteries*, IEEE Access, 12, 45231–45247.

## Dataset

- **Source**: NASA Prognostics Center of Excellence battery dataset
- **Size**: 10,000 rows × 18 columns
- **Battery cells**: B0005 (Fast Aging), B0006 (Slow Aging), B0007 (Unstable), B0018 (Temperature Sensitive)
- **Target variable**: `capacity_perc` — percentage decrease in usable capacity

## Pipeline

1. **Missing value handling** — Mean imputation across 9 sensor columns (0.88% overall missingness)
2. **Feature selection** — One-Way ANOVA, reducing 15 candidate features to the top 7 (capacity, battery_impedance, rct, rectified_impedance, re, temperature_charge, temperature_discharge)
3. **Model training** — 70/30 train-test split, three regression models:
   - Random Forest (120 trees, max depth 8)
   - Support Vector Regressor (RBF kernel, C=4, epsilon=0.18)
   - Gradient Boosting (160 estimators, learning rate 0.09) — *novelty of this project*
4. **Evaluation** — R², MSE, RMSE on held-out test set
5. **RUL estimation** — Cycle at which predicted capacity crosses the 80% EOL threshold

## Results

| Model | R² | MSE | RMSE |
|---|---|---|---|
| Random Forest | 0.7268 | 45.6714 | 6.7581 |
| Gradient Boosting | 0.7150 | 47.6388 | 6.9021 |
| SVR | 0.7097 | 48.5325 | 6.9665 |

Random Forest performed best among the three implemented models.

### Estimated RUL per battery

| Battery | EOL Threshold (Ahr) | Predicted EOL (Cycle) | Current Cycle | RUL (Cycles) |
|---|---|---|---|---|
| B0005 | 1.488 | 168 | 118 | 50 |
| B0006 | 1.600 | >300 | 65 | >235 |
| B0007 | 1.512 | 195 | 107 | 88 |
| B0018 | 1.480 | 172 | 112 | 60 |

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (Random Forest, SVR, Gradient Boosting)
- matplotlib / seaborn (visualization)
- Jupyter Notebook

## Repository Structure

```
├── EV_Battery_RUL.ipynb                            # Full pipeline: preprocessing → modeling → evaluation
├── metadata.csv                                     # Battery sensor dataset (10,000 rows × 18 columns)
├── EV_Battery_RUL_Documentation (1)_compressed.pdf  # Full project report
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

```bash
git clone https://github.com/<your-username>/ev-battery-rul-prediction.git
cd ev-battery-rul-prediction
pip install -r requirements.txt
jupyter notebook EV_Battery_RUL.ipynb
```

## Future Work

- Online incremental learning for real-time RUL updates
- Uncertainty quantification (conformal prediction / Monte Carlo dropout)
- Cross-chemistry generalization (LFP, NMC cells)
- Embedded deployment for on-vehicle battery management systems

## References

Full reference list available in the [project report](report/EV_Battery_RUL_Documentation.pdf).
