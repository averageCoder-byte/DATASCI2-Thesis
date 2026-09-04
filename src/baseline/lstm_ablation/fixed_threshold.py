from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# Paths
# ============================================================

BASELINE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASELINE_DIR.parent / "modeling"
DATA_DIR = MODEL_DIR / "data"

VALIDATION_ERRORS_PATH = (
    DATA_DIR
    / "validation_results"
    / "validation_reconstruction_errors.npy"
)

TEST_ERRORS_PATH = (
    DATA_DIR
    / "test_results"
    / "test_reconstruction_errors.npy"
)

VALIDATION_LABELS_PATH = (
    DATA_DIR
    / "evaluation"
    / "validation_proxy_labels.csv"
)

TEST_LABELS_PATH = (
    DATA_DIR
    / "evaluation"
    / "test_proxy_labels.csv"
)

OUTPUT_DIR = BASELINE_DIR / "data" / "lstm_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_PATH = OUTPUT_DIR / "fixed_threshold.txt"
TEST_RESULTS_PATH = OUTPUT_DIR / "test_fixed_threshold_results.csv"


# ============================================================
# Constants
# ============================================================

EXPECTED_VALIDATION_SEQUENCES = 21_812
EXPECTED_TEST_SEQUENCES = 21_858


# ============================================================
# Data loading
# ============================================================

def load_reconstruction_errors(path: Path, expected_count: int):
    """Load and validate reconstruction errors."""

    errors = np.load(path)

    if errors.ndim != 1:
        raise ValueError(
            f"Expected 1D reconstruction errors, got shape {errors.shape}."
        )

    if len(errors) != expected_count:
        raise ValueError(
            f"Expected {expected_count} reconstruction errors, "
            f"found {len(errors)}."
        )

    if not np.isfinite(errors).all():
        raise ValueError(
            "Reconstruction errors contain non-finite values."
        )

    return errors


def load_proxy_labels(path: Path, expected_count: int):
    """Load and validate sequence-level proxy labels."""

    df = pd.read_csv(path)

    required_columns = {
        "sequence_id",
        "proxy_anomaly",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(df) != expected_count:
        raise ValueError(
            f"Expected {expected_count} proxy labels, "
            f"found {len(df)}."
        )

    if df["sequence_id"].duplicated().any():
        raise ValueError(
            "Duplicate sequence_id values found."
        )

    if not df["proxy_anomaly"].isin([0, 1]).all():
        raise ValueError(
            "proxy_anomaly must contain only 0 and 1."
        )

    return df


# ============================================================
# Threshold selection
# ============================================================

def select_threshold(errors, labels):
    """
    Select reconstruction-error threshold using maximum
    balanced accuracy on the validation set.

    Higher reconstruction error indicates greater anomaly.
    """

    unique_thresholds = np.unique(errors)

    best_threshold = None
    best_balanced_accuracy = -np.inf

    for threshold in unique_thresholds:

        predictions = (
            errors >= threshold
        ).astype(int)

        score = balanced_accuracy_score(
            labels,
            predictions,
        )

        if score > best_balanced_accuracy:
            best_balanced_accuracy = score
            best_threshold = threshold

    return best_threshold, best_balanced_accuracy


# ============================================================
# Evaluation
# ============================================================

def evaluate_threshold(errors, labels, threshold):
    """Evaluate a frozen reconstruction-error threshold."""

    predictions = (
        errors >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions,
    )

    roc_auc = roc_auc_score(
        labels,
        errors,
    )

    pr_auc = average_precision_score(
        labels,
        errors,
    )

    predicted_anomalous = int(predictions.sum())
    total = len(predictions)

    return {
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "predicted_anomalous": predicted_anomalous,
        "predicted_anomaly_pct": (
            predicted_anomalous / total * 100
        ),
        "proxy_anomalous": int(labels.sum()),
        "proxy_anomaly_pct": (
            labels.sum() / total * 100
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("LSTM AUTOENCODER + FIXED THRESHOLD BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_errors = load_reconstruction_errors(
        VALIDATION_ERRORS_PATH,
        EXPECTED_VALIDATION_SEQUENCES,
    )

    validation_labels_df = load_proxy_labels(
        VALIDATION_LABELS_PATH,
        EXPECTED_VALIDATION_SEQUENCES,
    )

    validation_labels = (
        validation_labels_df["proxy_anomaly"]
        .to_numpy()
    )

    print("\nValidation dataset:")
    print(f"  Sequences: {len(validation_errors)}")

    print(
        f"  Proxy-normal: "
        f"{(validation_labels == 0).sum()}"
    )

    print(
        f"  Proxy-anomalous: "
        f"{(validation_labels == 1).sum()}"
    )

    threshold, validation_balanced_accuracy = (
        select_threshold(
            validation_errors,
            validation_labels,
        )
    )

    print("\n" + "=" * 60)
    print("SELECTED VALIDATION THRESHOLD")
    print("(Criterion: maximum balanced accuracy)")
    print("=" * 60)

    print(f"Threshold:          {threshold:.6f}")
    print(
        f"Validation balanced accuracy: "
        f"{validation_balanced_accuracy:.6f}"
    )

    # --------------------------------------------------------
    # Freeze threshold
    # --------------------------------------------------------

    THRESHOLD_PATH.write_text(
        f"{threshold:.12f}\n",
        encoding="utf-8",
    )

    print(f"\nThreshold saved to:")
    print(THRESHOLD_PATH)

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_errors = load_reconstruction_errors(
        TEST_ERRORS_PATH,
        EXPECTED_TEST_SEQUENCES,
    )

    test_labels_df = load_proxy_labels(
        TEST_LABELS_PATH,
        EXPECTED_TEST_SEQUENCES,
    )

    test_labels = (
        test_labels_df["proxy_anomaly"]
        .to_numpy()
    )

    print("\nTest dataset:")
    print(f"  Sequences: {len(test_errors)}")

    results = evaluate_threshold(
        test_errors,
        test_labels,
        threshold,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"\nFrozen validation threshold:"
        f" {results['threshold']:.6f}"
    )

    print("\nConfusion matrix:")
    print(f"  TN: {results['tn']:,}")
    print(f"  FP: {results['fp']:,}")
    print(f"  FN: {results['fn']:,}")
    print(f"  TP: {results['tp']:,}")

    print("\nThreshold-dependent metrics:")
    print(f"  Precision:          {results['precision']:.6f}")
    print(f"  Recall:             {results['recall']:.6f}")
    print(f"  F1:                 {results['f1']:.6f}")
    print(
        f"  Balanced accuracy:  "
        f"{results['balanced_accuracy']:.6f}"
    )

    print("\nThreshold-independent metrics:")
    print(f"  ROC-AUC:             {results['roc_auc']:.6f}")
    print(f"  PR-AUC:              {results['pr_auc']:.6f}")

    print(
        f"\nPredicted anomalous: "
        f"{results['predicted_anomalous']:,} "
        f"({results['predicted_anomaly_pct']:.2f}%)"
    )

    print(
        f"Proxy anomalous:     "
        f"{results['proxy_anomalous']:,} "
        f"({results['proxy_anomaly_pct']:.2f}%)"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame([results])

    results_df.to_csv(
        TEST_RESULTS_PATH,
        index=False,
    )

    print("\nResults saved to:")
    print(TEST_RESULTS_PATH)


if __name__ == "__main__":
    main()