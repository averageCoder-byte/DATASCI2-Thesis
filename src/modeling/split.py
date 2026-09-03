from pathlib import Path
import pandas as pd


# Paths
MODEL_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    MODEL_DIR
    / ".."
    / "pipeline"
    / "data"
    / "processed"
    / "xauusd_5m_processed.parquet"
).resolve()

OUTPUT_DIR = MODEL_DIR / "data" / "split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load processed dataset
df = pd.read_parquet(INPUT_PATH)

# Ensure chronological order
df = df.sort_values("Date").reset_index(drop=True)


# Chronological split: 70 / 15 / 15
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()


# Save split datasets
train_df.to_parquet(
    OUTPUT_DIR / "train.parquet",
    index=False,
)

val_df.to_parquet(
    OUTPUT_DIR / "validation.parquet",
    index=False,
)

test_df.to_parquet(
    OUTPUT_DIR / "test.parquet",
    index=False,
)


# Summary
print(f"Total:      {len(df):,}")
print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(val_df):,}")
print(f"Test:       {len(test_df):,}")

print("\nDate ranges:")
print(f"Train:      {train_df['Date'].iloc[0]} → {train_df['Date'].iloc[-1]}")
print(f"Validation: {val_df['Date'].iloc[0]} → {val_df['Date'].iloc[-1]}")
print(f"Test:       {test_df['Date'].iloc[0]} → {test_df['Date'].iloc[-1]}")

print(f"\nSaved to: {OUTPUT_DIR}")