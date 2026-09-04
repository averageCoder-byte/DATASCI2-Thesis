"""
Evaluation-only proxy labeling for LGMMA-X.

Proxy labels are NOT used during LSTM Autoencoder training or GMM fitting.

Thresholds are derived exclusively from the training split and then applied
to the target split.

Proxy criteria:
    1. Extreme return
    2. Range expansion
    3. Volatility burst
    4. Flash movement

A sequence is proxy-anomalous if at least one timestep within the 60-candle
sequence satisfies at least one proxy criterion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEQUENCE_LENGTH = 60

FLASH_FORWARD_CANDLES = 2       # 10 minutes
FLASH_RETRACE_CANDLES = 4       # additional 20 minutes
RETRACEMENT_THRESHOLD = 0.50

QUANTILE = 0.99


def load_processed_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the processed XAU/USD dataset."""

    df = pd.read_parquet(path)

    required = {
        "Date",
        "Close",
        "log_return",
        "range_pct",
        "rolling_volatility",
        "segment_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
    )

    df = (
        df.sort_values(["segment_id", "Date"])
        .reset_index(drop=True)
    )

    return df


def load_sequence_metadata(
    path: str | Path,
) -> pd.DataFrame:
    """Load sequence metadata."""

    metadata = pd.read_parquet(path)

    required = {
        "sequence_id",
        "segment_id",
        "start_timestamp",
        "end_timestamp",
    }

    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    metadata = metadata.copy()

    metadata["start_timestamp"] = pd.to_datetime(
        metadata["start_timestamp"],
        utc=True,
    )

    metadata["end_timestamp"] = pd.to_datetime(
        metadata["end_timestamp"],
        utc=True,
    )

    return (
        metadata
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )


def calculate_flash_movement(
    df: pd.DataFrame,
    h: int = FLASH_FORWARD_CANDLES,
    r: int = FLASH_RETRACE_CANDLES,
) -> pd.DataFrame:
    """
    Calculate flash-movement metrics.

    Shock:
        |P[t+h] - P[t]|

    Reversal:
        |P[t+h+r] - P[t+h]|

    Retracement ratio:
        Reversal / Shock

    Calculations are performed independently within each continuous
    segment, preventing movement calculations from crossing timestamp gaps.
    """

    result = df.copy()

    result["shock"] = np.nan
    result["reversal"] = np.nan
    result["retracement_ratio"] = np.nan

    for _, group in result.groupby(
        "segment_id",
        sort=False,
    ):
        indices = group.index
        prices = group["Close"].to_numpy()

        n = len(prices)

        shock = np.full(n, np.nan)
        reversal = np.full(n, np.nan)

        # Need t+h+r to exist.
        for i in range(n - h - r):
            shock[i] = abs(
                prices[i + h] - prices[i]
            )

            reversal[i] = abs(
                prices[i + h + r] - prices[i + h]
            )

        retracement_ratio = np.divide(
            reversal,
            shock,
            out=np.full(n, np.nan),
            where=shock != 0,
        )

        result.loc[indices, "shock"] = shock
        result.loc[
            indices,
            "reversal",
        ] = reversal
        result.loc[
            indices,
            "retracement_ratio",
        ] = retracement_ratio

    return result


def derive_thresholds(
    train_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Derive proxy thresholds exclusively from training data.
    """

    abs_returns = (
        train_df["log_return"]
        .abs()
        .dropna()
    )

    thresholds = {
        "extreme_return": float(
            abs_returns.mean()
            + 3.0 * abs_returns.std()
        ),

        "range_expansion": float(
            train_df["range_pct"]
            .dropna()
            .quantile(QUANTILE)
        ),

        "volatility_burst": float(
            train_df["rolling_volatility"]
            .dropna()
            .quantile(QUANTILE)
        ),

        "flash_shock": float(
            train_df["shock"]
            .dropna()
            .quantile(QUANTILE)
        ),
    }

    return thresholds


def apply_timestep_labels(
    df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Apply fixed training-derived thresholds."""

    result = df.copy()

    result["extreme_return"] = (
        result["log_return"].abs()
        > thresholds["extreme_return"]
    )

    result["range_expansion"] = (
        result["range_pct"]
        > thresholds["range_expansion"]
    )

    result["volatility_burst"] = (
        result["rolling_volatility"]
        > thresholds["volatility_burst"]
    )

    result["flash_movement"] = (
        (
            result["shock"]
            > thresholds["flash_shock"]
        )
        &
        (
            result["retracement_ratio"]
            >= RETRACEMENT_THRESHOLD
        )
    )

    result["proxy_anomaly_timestep"] = (
        result["extreme_return"]
        |
        result["range_expansion"]
        |
        result["volatility_burst"]
        |
        result["flash_movement"]
    )

    return result


def aggregate_sequence_labels(
    labeled_df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert timestep-level proxy flags into sequence-level labels.

    A sequence is anomalous when at least one timestep satisfies
    any proxy criterion.
    """

    rows = []

    criteria = [
        "extreme_return",
        "range_expansion",
        "volatility_burst",
        "flash_movement",
    ]

    for _, sequence in metadata.iterrows():

        mask = (
            (labeled_df["segment_id"] == sequence["segment_id"])
            &
            (
                labeled_df["Date"]
                >= sequence["start_timestamp"]
            )
            &
            (
                labeled_df["Date"]
                <= sequence["end_timestamp"]
            )
        )

        window = labeled_df.loc[mask]

        if len(window) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence {sequence['sequence_id']} "
                f"expected {SEQUENCE_LENGTH} candles, "
                f"found {len(window)}."
            )

        row = {
            "sequence_id": sequence["sequence_id"],
            "segment_id": sequence["segment_id"],
            "start_timestamp": sequence["start_timestamp"],
            "end_timestamp": sequence["end_timestamp"],
            "anomalous_timestep_count": int(
                window["proxy_anomaly_timestep"].sum()
            ),
            "anomalous_timestep_pct": (
                window["proxy_anomaly_timestep"].mean() * 100
            ),
        }

        for criterion in criteria:
            row[criterion] = int(
                window[criterion].any()
            )

        row["proxy_anomaly"] = int(
            window["proxy_anomaly_timestep"].any()
        )

        rows.append(row)

    return pd.DataFrame(rows)

def analyze_criterion_overlap(labeled_df):
    criteria = [
        "extreme_return",
        "range_expansion",
        "volatility_burst",
        "flash_movement",
    ]

    # Number of criteria triggered at each timestep
    trigger_count = labeled_df[criteria].sum(axis=1)

    print()
    print("Timestep-level criterion overlap:")
    print()

    total = len(labeled_df)

    for n in range(1, len(criteria) + 1):
        count = (trigger_count == n).sum()
        rate = count / total * 100

        print(
            f"  Exactly {n} criterion(s): "
            f"{count:,} ({rate:.2f}%)"
        )

    print()
    print("Pairwise criterion overlap:")
    print()

    for i in range(len(criteria)):
        for j in range(i + 1, len(criteria)):
            a = criteria[i]
            b = criteria[j]

            overlap = (
                labeled_df[a] &
                labeled_df[b]
            ).sum()

            print(
                f"  {a} + {b}: "
                f"{overlap:,}"
            )
def analyze_sequence_severity(labeled_df, sequence_metadata):
    criteria = [
        "extreme_return",
        "range_expansion",
        "volatility_burst",
        "flash_movement",
    ]

    labeled_df = labeled_df.sort_values(
        ["segment_id", "Date"]
    ).reset_index(drop=True)

    # Create sequence IDs using the exact metadata windows.
    sequence_results = []

    for row in sequence_metadata.itertuples(index=False):
        sequence_df = labeled_df[
            (labeled_df["segment_id"] == row.segment_id)
            & (labeled_df["Date"] >= row.start_timestamp)
            & (labeled_df["Date"] <= row.end_timestamp)
        ]

        if len(sequence_df) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence {row.sequence_id} contains "
                f"{len(sequence_df)} candles instead of "
                f"{SEQUENCE_LENGTH}."
            )

        anomalous_timesteps = sequence_df[
            "proxy_anomaly_timestep"
        ].sum()

        criteria_triggered = [
            criterion
            for criterion in criteria
            if sequence_df[criterion].any()
        ]

        sequence_results.append({
            "sequence_id": row.sequence_id,
            "anomalous_timestep_count": int(
                anomalous_timesteps
            ),
            "anomalous_timestep_pct": (
                anomalous_timesteps / SEQUENCE_LENGTH * 100
            ),
            "criteria_count": len(criteria_triggered),
            "criteria_triggered": ", ".join(
                criteria_triggered
            ),
        })

    severity_df = pd.DataFrame(sequence_results)

    print()
    print("Sequence-level proxy severity:")
    print()

    print(
        f"  Normal sequences: "
        f"{(severity_df['anomalous_timestep_count'] == 0).sum():,}"
    )

    print(
        f"  Anomalous sequences: "
        f"{(severity_df['anomalous_timestep_count'] > 0).sum():,}"
    )

    print()
    print("Anomalous timestep count per sequence:")
    print()

    anomalous = severity_df[
        severity_df["anomalous_timestep_count"] > 0
    ]

    severity_bins = [
        ("1", 1, 1),
        ("2–3", 2, 3),
        ("4–5", 4, 5),
        ("6–10", 6, 10),
        ("11–20", 11, 20),
        ("21–30", 21, 30),
        ("31–40", 31, 40),
        ("41+", 41, SEQUENCE_LENGTH),
    ]

    for label, lower, upper in severity_bins:
        sequence_count = (
            anomalous["anomalous_timestep_count"]
            .between(lower, upper)
            .sum()
        )

        rate = sequence_count / len(anomalous) * 100

        print(
            f"  {label} anomalous timesteps: "
            f"{sequence_count:,} "
            f"({rate:.2f}%)"
        )

    print()
    print("Anomalous timestep percentage:")
    print()

    print(
        f"  Mean:   "
        f"{anomalous['anomalous_timestep_pct'].mean():.2f}%"
    )

    print(
        f"  Median: "
        f"{anomalous['anomalous_timestep_pct'].median():.2f}%"
    )

    print(
        f"  Max:    "
        f"{anomalous['anomalous_timestep_pct'].max():.2f}%"
    )

    print()
    print("Number of distinct criteria per anomalous sequence:")
    print()

    for count in range(1, 5):
        sequence_count = (
            anomalous["criteria_count"] == count
        ).sum()

        print(
            f"  {count} criterion(s): "
            f"{sequence_count:,}"
        )

    return severity_df


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate evaluation-only proxy labels "
            "for XAU/USD sequences."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Processed XAU/USD parquet.",
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Sequence metadata parquet.",
    )

    parser.add_argument(
        "--train-start",
        required=True,
        help="Training start timestamp in UTC.",
    )

    parser.add_argument(
        "--train-end",
        required=True,
        help="Training end timestamp in UTC.",
    )

    parser.add_argument(
        "--target-start",
        required=True,
        help="Target split start timestamp in UTC.",
    )

    parser.add_argument(
        "--target-end",
        required=True,
        help="Target split end timestamp in UTC.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = load_processed_data(args.data)

    metadata = load_sequence_metadata(
        args.metadata
    )

    # ---------------------------------------------------------
    # Calculate flash movement
    # ---------------------------------------------------------

    df = calculate_flash_movement(df)

    # ---------------------------------------------------------
    # Parse timestamps
    # ---------------------------------------------------------

    train_start = pd.Timestamp(
        args.train_start,
        tz="UTC",
    )

    train_end = pd.Timestamp(
        args.train_end,
        tz="UTC",
    )

    target_start = pd.Timestamp(
        args.target_start,
        tz="UTC",
    )

    target_end = pd.Timestamp(
        args.target_end,
        tz="UTC",
    )

    # ---------------------------------------------------------
    # Training data
    # ---------------------------------------------------------

    train_df = df[
        (df["Date"] >= train_start)
        &
        (df["Date"] <= train_end)
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training interval contains no rows."
        )

    # ---------------------------------------------------------
    # Target data
    # ---------------------------------------------------------

    target_df = df[
        (df["Date"] >= target_start)
        &
        (df["Date"] <= target_end)
    ].copy()

    if target_df.empty:
        raise ValueError(
            "Target interval contains no rows."
        )

    # ---------------------------------------------------------
    # Derive thresholds ONLY from training data
    # ---------------------------------------------------------

    thresholds = derive_thresholds(
        train_df
    )

        # ---------------------------------------------------------
    # Timestep-level calibration diagnostics
    # ---------------------------------------------------------

    labeled_train = apply_timestep_labels(
        train_df,
        thresholds,
    )

    labeled_target = apply_timestep_labels(
        target_df,
        thresholds,
    )

    analyze_criterion_overlap(labeled_target)

    severity_df = analyze_sequence_severity(
        labeled_target,
        metadata,
    )

    print()
    print("Timestep-level calibration:")
    print()

    for criterion in [
        "extreme_return",
        "range_expansion",
        "volatility_burst",
        "flash_movement",
        "proxy_anomaly_timestep",
    ]:
        train_count = labeled_train[criterion].sum()
        target_count = labeled_target[criterion].sum()

        train_rate = (
            train_count / len(labeled_train) * 100
        )

        target_rate = (
            target_count / len(labeled_target) * 100
        )

        print(
            f"  {criterion}:"
        )
        print(
            f"    train: {train_count:,} "
            f"({train_rate:.2f}%)"
        )
        print(
            f"    test:  {target_count:,} "
            f"({target_rate:.2f}%)"
        )

    # ---------------------------------------------------------
    # Apply thresholds to target data
    # ---------------------------------------------------------

    # labeled_target = apply_timestep_labels(
    #     target_df,
    #     thresholds,
    # )

    # ---------------------------------------------------------
    # Select target sequences
    # ---------------------------------------------------------

    target_metadata = metadata[
        (metadata["start_timestamp"] >= target_start)
        &
        (metadata["end_timestamp"] <= target_end)
    ].copy()

    # ---------------------------------------------------------
    # Aggregate timestep labels
    # into sequence-level labels
    # ---------------------------------------------------------

    sequence_labels = aggregate_sequence_labels(
        labeled_target,
        target_metadata,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequence_labels.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("Proxy labeling complete.")
    print()
    print(f"Training rows: {len(train_df):,}")
    print(f"Target rows:   {len(target_df):,}")
    print(
        f"Sequences:     {len(sequence_labels):,}"
    )

    print()
    print("Training-derived thresholds:")

    for name, value in thresholds.items():
        print(
            f"  {name}: {value:.10f}"
        )

    print()
    print("Sequence-level proxy counts:")

    for criterion in [
        "extreme_return",
        "range_expansion",
        "volatility_burst",
        "flash_movement",
        "proxy_anomaly",
    ]:
        count = sequence_labels[criterion].sum()

        print(
            f"  {criterion}: {count:,}"
        )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()