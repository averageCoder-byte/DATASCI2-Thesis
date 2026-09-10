from pathlib import Path

import numpy as np
import pandas as pd


# Paths
MODEL_DIR = Path(__file__).resolve().parent

INPUT_DIR = MODEL_DIR / "data" / "scaled"
OUTPUT_DIR = MODEL_DIR / "data" / "sequences"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Configuration
SEQUENCE_LENGTH = 60

FEATURE_COLUMNS = [
    "log_return",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "rolling_volatility",
    "rolling_mean_return",
]


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: list[str],
) -> np.ndarray:

    sequences = []

    # Process each continuous timestamp segment separately
    for _, segment in df.groupby("segment_id", sort=False):

        values = segment[feature_columns].to_numpy(dtype=np.float32)

        if len(values) < sequence_length:
            continue

        for i in range(len(values) - sequence_length + 1):
            sequence = values[i:i + sequence_length]
            sequences.append(sequence)

    return np.asarray(sequences, dtype=np.float32)


# Load scaled datasets
train_df = pd.read_parquet(
    INPUT_DIR / "train_scaled.parquet"
)

val_df = pd.read_parquet(
    INPUT_DIR / "validation_scaled.parquet"
)

test_df = pd.read_parquet(
    INPUT_DIR / "test_scaled.parquet"
)


# Create sequences
X_train = create_sequences(
    train_df,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
)

X_val = create_sequences(
    val_df,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
)

X_test = create_sequences(
    test_df,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
)


# Save sequences
np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train,
)

np.save(
    OUTPUT_DIR / "X_validation.npy",
    X_val,
)

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test,
)


# Summary
print("Sequence generation complete.")

print(f"\nTrain:")
print(f"  Sequences: {len(X_train):,}")
print(f"  Shape:     {X_train.shape}")

print(f"\nValidation:")
print(f"  Sequences: {len(X_val):,}")
print(f"  Shape:     {X_val.shape}")

print(f"\nTest:")
print(f"  Sequences: {len(X_test):,}")
print(f"  Shape:     {X_test.shape}")

print(f"\nSequence length: {SEQUENCE_LENGTH}")
print(f"Features:        {len(FEATURE_COLUMNS)}")
print(f"Saved to:        {OUTPUT_DIR}")