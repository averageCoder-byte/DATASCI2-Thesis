from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "modeling" / "data"
BASELINE_DIR = BASE_DIR / "baseline" / "data" / "classical_ml"
STATISTICAL_COMPARISON_DIR = (
    BASE_DIR / "baseline" / "statistical_comparison"
)

STATISTICAL_COMPARISON_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

TEST_SEQUENCES = (
    MODEL_DIR
    / "sequences"
    / "X_test.npy"
)

TEST_LABELS = (
    MODEL_DIR
    / "evaluation"
    / "test_proxy_labels.csv"
)

TEST_METADATA = (
    MODEL_DIR
    / "sequence_metadata"
    / "test_sequence_metadata.parquet"
)

MODEL_PATH = (
    BASELINE_DIR
    / "one_class_svm.joblib"
)

THRESHOLD_PATH = (
    BASELINE_DIR
    / "one_class_svm_threshold.txt"
)


# ------------------------------------------------------------
# Output file
# ------------------------------------------------------------

OUTPUT_PATH = (
    STATISTICAL_COMPARISON_DIR
    / "one_class_svm_test_scores.csv"
)


# ============================================================
# Load data
# ============================================================

def load_data():
    """
    Load the frozen One-Class SVM model, test sequences,
    frozen validation threshold, proxy labels, and sequence
    metadata.

    No model fitting, threshold selection, or tuning is
    performed in this script.
    """

    if not TEST_SEQUENCES.exists():
        raise FileNotFoundError(
            f"Missing test sequences:\n{TEST_SEQUENCES}"
        )

    if not TEST_LABELS.exists():
        raise FileNotFoundError(
            f"Missing test proxy labels:\n{TEST_LABELS}"
        )

    if not TEST_METADATA.exists():
        raise FileNotFoundError(
            f"Missing test sequence metadata:\n{TEST_METADATA}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing One-Class SVM model:\n{MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing One-Class SVM threshold:\n{THRESHOLD_PATH}"
        )

    X_test = np.load(
        TEST_SEQUENCES
    )

    test_labels = pd.read_csv(
        TEST_LABELS
    )

    test_metadata = pd.read_parquet(
        TEST_METADATA
    )

    model = joblib.load(
        MODEL_PATH
    )

    threshold = float(
        THRESHOLD_PATH.read_text(
            encoding="utf-8"
        ).strip()
    )

    return (
        X_test,
        test_labels,
        test_metadata,
        model,
        threshold,
    )


# ============================================================
# Flatten sequences
# ============================================================

def flatten_sequences(X):
    """
    Convert:

        (samples, 60, 7)

    into:

        (samples, 420)

    One-Class SVM operates on a two-dimensional feature matrix.
    """

    if X.ndim != 3:
        raise ValueError(
            "Expected test sequences with shape "
            "(samples, timesteps, features). "
            f"Received shape: {X.shape}"
        )

    return X.reshape(
        X.shape[0],
        -1,
    )


# ============================================================
# Anomaly scores
# ============================================================

def anomaly_scores(model, X):
    """
    One-Class SVM decision_function():

        larger = more normal

    Negating produces the standardized convention:

        larger = more anomalous
    """

    return -model.decision_function(
        X
    )


# ============================================================
# Validate inputs
# ============================================================

def validate_inputs(
    X_test,
    test_labels,
    test_metadata,
):
    """
    Verify that all sequence-level artifacts contain the same
    number of test sequences and the required columns.
    """

    required_label_columns = {
        "sequence_id",
        "proxy_anomaly",
    }

    required_metadata_columns = {
        "sequence_id",
        "segment_id",
        "start_timestamp",
        "end_timestamp",
    }

    missing_label_columns = (
        required_label_columns
        - set(test_labels.columns)
    )

    if missing_label_columns:
        raise ValueError(
            "Test proxy labels are missing required "
            f"columns: {sorted(missing_label_columns)}"
        )

    missing_metadata_columns = (
        required_metadata_columns
        - set(test_metadata.columns)
    )

    if missing_metadata_columns:
        raise ValueError(
            "Test sequence metadata is missing required "
            f"columns: {sorted(missing_metadata_columns)}"
        )

    n_sequences = len(X_test)

    if len(test_labels) != n_sequences:
        raise ValueError(
            "Test sequence count does not match proxy labels. "
            f"Sequences: {n_sequences:,}; "
            f"Labels: {len(test_labels):,}"
        )

    if len(test_metadata) != n_sequences:
        raise ValueError(
            "Test sequence count does not match sequence metadata. "
            f"Sequences: {n_sequences:,}; "
            f"Metadata: {len(test_metadata):,}"
        )

    if test_labels["sequence_id"].duplicated().any():
        raise ValueError(
            "Duplicate sequence_id values found in "
            "test proxy labels."
        )

    if test_metadata["sequence_id"].duplicated().any():
        raise ValueError(
            "Duplicate sequence_id values found in "
            "test sequence metadata."
        )


# ============================================================
# Build standardized output
# ============================================================

def build_output(
    scores,
    threshold,
    test_labels,
    test_metadata,
):
    """
    Build the standardized sequence-level evaluation table.

    Output columns:

        sequence_id
        segment_id
        start_timestamp
        end_timestamp
        anomaly_score
        proxy_anomaly
        prediction
    """

    scores_df = pd.DataFrame({
        "sequence_id": test_metadata[
            "sequence_id"
        ].to_numpy(),

        "segment_id": test_metadata[
            "segment_id"
        ].to_numpy(),

        "start_timestamp": test_metadata[
            "start_timestamp"
        ].to_numpy(),

        "end_timestamp": test_metadata[
            "end_timestamp"
        ].to_numpy(),

        "anomaly_score": scores,
    })

    labels_df = test_labels[
        [
            "sequence_id",
            "proxy_anomaly",
        ]
    ].copy()

    # --------------------------------------------------------
    # Align labels by sequence_id
    # --------------------------------------------------------

    output = scores_df.merge(
        labels_df,
        on="sequence_id",
        how="left",
        validate="one_to_one",
    )

    if output["proxy_anomaly"].isna().any():
        raise ValueError(
            "Some test sequences have no matching proxy label."
        )

    output["proxy_anomaly"] = (
        output["proxy_anomaly"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Apply frozen validation threshold
    # --------------------------------------------------------

    output["prediction"] = (
        output["anomaly_score"]
        >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    output = output.sort_values(
        by=[
            "start_timestamp",
            "end_timestamp",
            "sequence_id",
        ]
    ).reset_index(
        drop=True
    )

    return output


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "ONE-CLASS SVM TEST SCORE GENERATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load frozen artifacts
    # --------------------------------------------------------

    (
        X_test,
        test_labels,
        test_metadata,
        model,
        threshold,
    ) = load_data()

    print("\nLoaded:")
    print(
        f"Test sequences: "
        f"{X_test.shape}"
    )

    print(
        f"Proxy labels:   "
        f"{test_labels.shape}"
    )

    print(
        f"Metadata:       "
        f"{test_metadata.shape}"
    )

    print(
        f"Model:          "
        f"{MODEL_PATH}"
    )

    print(
        f"Threshold:      "
        f"{threshold:.12f}"
    )

    # --------------------------------------------------------
    # Validate artifacts
    # --------------------------------------------------------

    validate_inputs(
        X_test,
        test_labels,
        test_metadata,
    )

    # --------------------------------------------------------
    # Flatten sequences
    # --------------------------------------------------------

    X_test_flat = flatten_sequences(
        X_test
    )

    print("\nFlattened test shape:")
    print(
        X_test_flat.shape
    )

    # --------------------------------------------------------
    # Generate frozen test scores
    # --------------------------------------------------------

    print(
        "\nGenerating One-Class SVM "
        "test anomaly scores..."
    )

    scores = anomaly_scores(
        model,
        X_test_flat,
    )

    if len(scores) != len(X_test):
        raise ValueError(
            "Number of generated anomaly scores "
            "does not match number of test sequences."
        )

    # --------------------------------------------------------
    # Build standardized output
    # --------------------------------------------------------

    output = build_output(
        scores,
        threshold,
        test_labels,
        test_metadata,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if len(output) != len(X_test):
        raise ValueError(
            "Final output row count does not match "
            "test sequence count."
        )

    required_output_columns = [
        "sequence_id",
        "segment_id",
        "start_timestamp",
        "end_timestamp",
        "anomaly_score",
        "proxy_anomaly",
        "prediction",
    ]

    if list(output.columns) != required_output_columns:
        raise ValueError(
            "Unexpected output columns.\n"
            f"Expected: {required_output_columns}\n"
            f"Received: {list(output.columns)}"
        )

    if output["sequence_id"].duplicated().any():
        raise ValueError(
            "Duplicate sequence_id values found "
            "in final output."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST SCORE SUMMARY")
    print("=" * 70)

    print(
        f"Sequences:          "
        f"{len(output):,}"
    )

    print(
        f"Anomaly score mean: "
        f"{output['anomaly_score'].mean():.6f}"
    )

    print(
        f"Anomaly score std:   "
        f"{output['anomaly_score'].std():.6f}"
    )

    print(
        f"Anomaly score min:   "
        f"{output['anomaly_score'].min():.6f}"
    )

    print(
        f"Anomaly score max:   "
        f"{output['anomaly_score'].max():.6f}"
    )

    print(
        f"Proxy normal:        "
        f"{(output['proxy_anomaly'] == 0).sum():,}"
    )

    print(
        f"Proxy anomalous:     "
        f"{(output['proxy_anomaly'] == 1).sum():,}"
    )

    print(
        f"Predicted normal:    "
        f"{(output['prediction'] == 0).sum():,}"
    )

    print(
        f"Predicted anomalous: "
        f"{(output['prediction'] == 1).sum():,}"
    )

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\nFirst five rows:")
    print(
        output.head().to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()