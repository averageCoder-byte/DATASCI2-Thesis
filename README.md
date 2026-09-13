# DATASCI2-Thesis

**LGMMA-X: Historical Sequence-Based Anomaly Detection and Probabilistic Scoring of XAU/USD Intraday Market Data Using LSTM Autoencoder and Gaussian Mixture Model**

This repository contains the implementation, evaluation, and experimental artifacts for the DATASCI2 thesis project. The proposed LGMMA-X framework performs sequence-based anomaly detection on historical 5-minute XAU/USD market data using an LSTM Autoencoder and a Gaussian Mixture Model (GMM) for probabilistic anomaly scoring.

## Project Overview

LGMMA-X processes historical XAU/USD candlestick data through the following stages:

1. Data ingestion and validation
2. Preprocessing and feature engineering
3. Gap-aware sequence construction
4. LSTM Autoencoder training and reconstruction
5. Reconstruction-error generation
6. GMM model selection and fitting
7. Probabilistic anomaly scoring
8. Validation-based threshold calibration
9. Held-out test evaluation
10. Baseline model comparison

The repository also contains the baseline implementations, statistical comparison notebooks, and Apache Airflow workflows used to support reproducible experimentation.

## Technology Stack

- Python
- TensorFlow / Keras
- scikit-learn
- pandas
- NumPy
- statsmodels
- arch
- Apache Airflow
- Docker
- Jupyter Notebook
- Git / GitHub

## Data

The study uses historical **5-minute XAU/USD** market data exported in GMT/UTC. The dataset contains OHLC values and engineered market features. Temporal gaps are preserved during preprocessing and sequence construction to prevent sequences from crossing discontinuities in the source data.

## Repository Structure

```text
src/
├── airflow/       # Airflow DAGs and orchestration
├── baseline/      # Baseline anomaly-detection models
├── modeling/      # LGMMA-X modeling and evaluation
├── notebooks/     # Analysis and visualization notebooks
└── pipeline/      # Data ingestion and preprocessing
```
