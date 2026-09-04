from pathlib import Path

import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
SPLIT_DIR = MODEL_DIR / "data" / "split"
OUTPUT_DIR = MODEL_DIR / "data" / "sequence_metadata"

TEST_PATH = SPLIT_DIR / "test.parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 60


def create_sequence_metadata(
    df: pd.DataFrame,
    sequence_length: int,
) -> pd.DataFrame:
    """Create timestamp metadata for each generated sequence."""

    metadata = []
    sequence_id = 0

    for segment_id, segment in df.groupby("segment_id", sort=False):
        segment = segment.reset_index(drop=True)

        if len(segment) < sequence_length:
            continue

        for i in range(len(segment) - sequence_length + 1):
            sequence = segment.iloc[i : i + sequence_length]

            metadata.append(
                {
                    "sequence_id": sequence_id,
                    "segment_id": segment_id,
                    "start_timestamp": sequence.iloc[0]["Date"],
                    "end_timestamp": sequence.iloc[-1]["Date"],
                }
            )

            sequence_id += 1

    return pd.DataFrame(metadata)


print("Loading test split...")
test_df = pd.read_parquet(TEST_PATH)

print(f"Test rows: {len(test_df):,}")
print(f"Test segments: {test_df['segment_id'].nunique():,}")

print("\nGenerating sequence metadata...")

metadata_df = create_sequence_metadata(
    test_df,
    sequence_length=SEQUENCE_LENGTH,
)

print(f"Generated sequences: {len(metadata_df):,}")

print("\nSequence metadata preview:")
print(metadata_df.head())

print("\nValidating sequence metadata...")

expected_sequences = 21_858

if len(metadata_df) != expected_sequences:
    raise ValueError(
        f"Expected {expected_sequences:,} sequences, "
        f"but generated {len(metadata_df):,}."
    )

if metadata_df["sequence_id"].duplicated().any():
    raise ValueError("Duplicate sequence IDs detected.")

if not metadata_df["start_timestamp"].is_monotonic_increasing:
    raise ValueError("Sequence start timestamps are not chronological.")

if (metadata_df["end_timestamp"] < metadata_df["start_timestamp"]).any():
    raise ValueError("Invalid sequence timestamp range detected.")

print("Sequence metadata validation passed.")

OUTPUT_PATH = OUTPUT_DIR / "test_sequence_metadata.parquet"

metadata_df.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("\nSequence metadata saved to:")
print(OUTPUT_PATH)