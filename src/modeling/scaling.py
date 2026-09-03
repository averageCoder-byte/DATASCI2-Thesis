from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler


# Paths
MODEL_DIR = Path(__file__).resolve().parent

INPUT_DIR = MODEL_DIR / "data" / "split"
OUTPUT_DIR = MODEL_DIR / "data" / "scaled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load split datasets
train_df = pd.read_parquet(INPUT_DIR / "train.parquet")
val_df = pd.read_parquet(INPUT_DIR / "validation.parquet")
test_df = pd.read_parquet(INPUT_DIR / "test.parquet")


# LSTM input features
FEATURE_COLUMNS = [
    "log_return",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "rolling_volatility",
    "rolling_mean_return",
]


# Initialize scaler
scaler = RobustScaler()


# Fit ONLY on training data
scaler.fit(train_df[FEATURE_COLUMNS])


# Transform all datasets using training parameters
train_df[FEATURE_COLUMNS] = scaler.transform(
    train_df[FEATURE_COLUMNS]
)

val_df[FEATURE_COLUMNS] = scaler.transform(
    val_df[FEATURE_COLUMNS]
)

test_df[FEATURE_COLUMNS] = scaler.transform(
    test_df[FEATURE_COLUMNS]
)


# Save scaled datasets
train_df.to_parquet(
    OUTPUT_DIR / "train_scaled.parquet",
    index=False,
)

val_df.to_parquet(
    OUTPUT_DIR / "validation_scaled.parquet",
    index=False,
)

test_df.to_parquet(
    OUTPUT_DIR / "test_scaled.parquet",
    index=False,
)


# Summary
print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(val_df):,}")
print(f"Test:       {len(test_df):,}")

print(f"\nSaved to: {OUTPUT_DIR}")