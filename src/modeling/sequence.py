from pathlib import Path

import numpy as np
import pandas as pd


# Paths
MODEL_DIR = Path(__file__).resolve().parent

INPUT_DIR = MODEL_DIR / "data" / "scaled"
SEQUENCE_OUTPUT_DIR = MODEL_DIR / "data" / "sequences"
METADATA_OUTPUT_DIR = MODEL_DIR / "data" / "sequence_metadata"

SEQUENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
) -> tuple[np.ndarray, pd.DataFrame]:

    sequences = []
    metadata = []

    # Process each continuous timestamp segment separately
    for segment_id, segment in df.groupby("segment_id", sort=False):

        # Ensure chronological order within each segment
        segment = segment.sort_values("Date").reset_index(drop=True)

        values = segment[feature_columns].to_numpy(dtype=np.float32)

        if len(values) < sequence_length:
            continue

        for i in range(len(values) - sequence_length + 1):
            sequence = values[i:i + sequence_length]

            sequences.append(sequence)

            metadata.append(
                {
                    "sequence_id": len(metadata),
                    "segment_id": segment_id,
                    "start_timestamp": segment.iloc[i]["Date"],
                    "end_timestamp": segment.iloc[
                        i + sequence_length - 1
                    ]["Date"],
                    "num_candles": sequence_length,
                }
            )

    return (
        np.asarray(sequences, dtype=np.float32),
        pd.DataFrame(metadata),
    )


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


# Create sequences and metadata
X_train, train_metadata = create_sequences(
    train_df,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
)

X_val, val_metadata = create_sequences(
    val_df,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
)

X_test, test_metadata = create_sequences(
    test_df,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
)


# Save sequences
np.save(
    SEQUENCE_OUTPUT_DIR / "X_train.npy",
    X_train,
)

np.save(
    SEQUENCE_OUTPUT_DIR / "X_validation.npy",
    X_val,
)

np.save(
    SEQUENCE_OUTPUT_DIR / "X_test.npy",
    X_test,
)


# Save sequence metadata
train_metadata.to_parquet(
    METADATA_OUTPUT_DIR / "train_sequence_metadata.parquet",
    index=False,
)

val_metadata.to_parquet(
    METADATA_OUTPUT_DIR / "validation_sequence_metadata.parquet",
    index=False,
)

test_metadata.to_parquet(
    METADATA_OUTPUT_DIR / "test_sequence_metadata.parquet",
    index=False,
)


# Validation checks
assert len(X_train) == len(train_metadata)
assert len(X_val) == len(val_metadata)
assert len(X_test) == len(test_metadata)

assert (train_metadata["num_candles"] == SEQUENCE_LENGTH).all()
assert (val_metadata["num_candles"] == SEQUENCE_LENGTH).all()
assert (test_metadata["num_candles"] == SEQUENCE_LENGTH).all()


# Summary
print("Sequence generation complete.")

print("\nTrain:")
print(f"  Sequences: {len(X_train):,}")
print(f"  Shape:     {X_train.shape}")
print(f"  Metadata:  {len(train_metadata):,}")

print("\nValidation:")
print(f"  Sequences: {len(X_val):,}")
print(f"  Shape:     {X_val.shape}")
print(f"  Metadata:  {len(val_metadata):,}")

print("\nTest:")
print(f"  Sequences: {len(X_test):,}")
print(f"  Shape:     {X_test.shape}")
print(f"  Metadata:  {len(test_metadata):,}")

print(f"\nSequence length: {SEQUENCE_LENGTH}")
print(f"Features:        {len(FEATURE_COLUMNS)}")

print(f"\nSequences saved to:")
print(f"  {SEQUENCE_OUTPUT_DIR}")

print("\nMetadata saved to:")
print(f"  {METADATA_OUTPUT_DIR}")