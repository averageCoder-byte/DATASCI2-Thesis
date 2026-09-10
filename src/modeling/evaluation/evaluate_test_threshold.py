"""
Final held-out test evaluation for the LGMMA-X anomaly detection pipeline.

This script evaluates the frozen anomaly-score threshold selected on the
validation set and reports both threshold-dependent and threshold-independent
metrics against the rule-based proxy labels.

Threshold-dependent:
    - Confusion matrix
    - Precision
    - Recall
    - F1
    - Balanced accuracy
    - Predicted anomaly prevalence

Threshold-independent:
    - ROC-AUC
    - PR-AUC

IMPORTANT:
    The test set must NOT be used to optimize or modify the threshold.
    The threshold is loaded from selected_threshold.txt, which was selected
    using the validation set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

# This file is located in:
# src/modeling/evaluation/evaluate_test_threshold.py

BASE_DIR = Path(__file__).resolve().parent.parent

ANOMALY_SCORES_PATH = (
    BASE_DIR
    / "data"
    / "gmm_results"
    / "test_anomaly_scores.csv"
)

PROXY_LABELS_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "test_proxy_labels.csv"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "selected_threshold.txt"
)


# ============================================================
# EXPECTED DATASET SIZE
# ============================================================

EXPECTED_TEST_SEQUENCES = 21_858


# ============================================================
# LOAD DATA
# ============================================================


def load_anomaly_scores() -> pd.DataFrame:
    """Load GMM anomaly scores for the test sequences."""

    if not ANOMALY_SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Test anomaly scores not found:\n"
            f"{ANOMALY_SCORES_PATH}"
        )

    df = pd.read_csv(ANOMALY_SCORES_PATH)

    required_columns = {
        "sequence_id",
        "anomaly_score",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Test anomaly score file is missing required columns: "
            f"{sorted(missing)}"
        )

    return df[["sequence_id", "anomaly_score"]].copy()


def load_proxy_labels() -> pd.DataFrame:
    """Load rule-based proxy labels for the test sequences."""

    if not PROXY_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Test proxy labels not found:\n"
            f"{PROXY_LABELS_PATH}"
        )

    df = pd.read_csv(PROXY_LABELS_PATH)

    required_columns = {
        "sequence_id",
        "proxy_anomaly",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Test proxy label file is missing required columns: "
            f"{sorted(missing)}"
        )

    return df[["sequence_id", "proxy_anomaly"]].copy()


def load_threshold() -> float:
    """Load the frozen validation-selected anomaly threshold."""

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Selected threshold not found:\n"
            f"{THRESHOLD_PATH}"
        )

    raw_value = THRESHOLD_PATH.read_text().strip()

    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid threshold value in:\n"
            f"{THRESHOLD_PATH}\n"
            f"Value found: {raw_value!r}"
        ) from exc

    if not np.isfinite(threshold):
        raise ValueError(
            f"Threshold must be finite. Found: {threshold}"
        )

    return threshold


# ============================================================
# VALIDATION
# ============================================================


def validate_inputs(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> None:
    """Validate the test evaluation inputs."""

    if scores_df["sequence_id"].duplicated().any():
        duplicates = (
            scores_df.loc[
                scores_df["sequence_id"].duplicated(),
                "sequence_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate sequence_id values found in test anomaly "
            f"scores. Examples: {duplicates[:10]}"
        )

    if labels_df["sequence_id"].duplicated().any():
        duplicates = (
            labels_df.loc[
                labels_df["sequence_id"].duplicated(),
                "sequence_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate sequence_id values found in test proxy "
            f"labels. Examples: {duplicates[:10]}"
        )

    if len(scores_df) != EXPECTED_TEST_SEQUENCES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SEQUENCES:,} test anomaly scores, "
            f"found {len(scores_df):,}."
        )

    if len(labels_df) != EXPECTED_TEST_SEQUENCES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SEQUENCES:,} test proxy labels, "
            f"found {len(labels_df):,}."
        )

    if not np.isfinite(scores_df["anomaly_score"]).all():
        raise ValueError(
            "Test anomaly scores contain NaN or infinite values."
        )

    unique_labels = set(
        labels_df["proxy_anomaly"].dropna().unique()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            "proxy_anomaly must contain only binary values 0 and 1. "
            f"Found: {sorted(unique_labels)}"
        )


# ============================================================
# MERGE
# ============================================================


def merge_evaluation_data(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge model scores and proxy labels using sequence_id.

    The merge must remain one-to-one.
    """

    merged = scores_df.merge(
        labels_df,
        on="sequence_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_TEST_SEQUENCES:
        raise ValueError(
            "One-to-one merge did not produce the expected number "
            f"of test sequences.\n"
            f"Expected: {EXPECTED_TEST_SEQUENCES:,}\n"
            f"Found:    {len(merged):,}"
        )

    return merged


# ============================================================
# METRICS
# ============================================================


def evaluate_threshold(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Evaluate binary predictions using the frozen threshold.
    """

    y_pred = (
        anomaly_scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    roc_auc = roc_auc_score(
        y_true,
        anomaly_scores,
    )

    pr_auc = average_precision_score(
        y_true,
        anomaly_scores,
    )

    predicted_anomaly_count = int(
        y_pred.sum()
    )

    actual_anomaly_count = int(
        y_true.sum()
    )

    total = len(y_true)

    predicted_anomaly_prevalence = (
        predicted_anomaly_count / total * 100
    )

    proxy_anomaly_prevalence = (
        actual_anomaly_count / total * 100
    )

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
        "predicted_anomaly_count": predicted_anomaly_count,
        "predicted_anomaly_prevalence": (
            predicted_anomaly_prevalence
        ),
        "proxy_anomaly_count": actual_anomaly_count,
        "proxy_anomaly_prevalence": (
            proxy_anomaly_prevalence
        ),
    }


# ============================================================
# OUTPUT
# ============================================================


def print_results(
    results: dict,
    total_sequences: int,
) -> None:
    """Print final test evaluation results."""

    print()
    print("=" * 60)
    print("LGMMA-X FINAL TEST EVALUATION")
    print("=" * 60)

    print()
    print("Evaluation dataset:")
    print(f"  Test sequences: {total_sequences:,}")

    print()
    print("Frozen validation threshold:")
    print(f"  Threshold: {results['threshold']:.6f}")

    print()
    print("-" * 60)
    print("THRESHOLD-DEPENDENT METRICS")
    print("-" * 60)

    print()
    print("Confusion matrix:")
    print(f"  TN: {results['tn']:,}")
    print(f"  FP: {results['fp']:,}")
    print(f"  FN: {results['fn']:,}")
    print(f"  TP: {results['tp']:,}")

    print()
    print(f"Precision:          {results['precision']:.6f}")
    print(f"Recall:             {results['recall']:.6f}")
    print(f"F1:                 {results['f1']:.6f}")
    print(
        f"Balanced accuracy:  "
        f"{results['balanced_accuracy']:.6f}"
    )

    print()
    print(
        f"Predicted anomalous: "
        f"{results['predicted_anomaly_count']:,} "
        f"({results['predicted_anomaly_prevalence']:.2f}%)"
    )

    print(
        f"Proxy anomalous:     "
        f"{results['proxy_anomaly_count']:,} "
        f"({results['proxy_anomaly_prevalence']:.2f}%)"
    )

    print()
    print("-" * 60)
    print("THRESHOLD-INDEPENDENT METRICS")
    print("-" * 60)

    print()
    print(
        f"ROC-AUC: {results['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC:  {results['pr_auc']:.6f}"
    )

    print()
    print("=" * 60)
    print("TEST EVALUATION COMPLETE")
    print("=" * 60)
    print()


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Run the final held-out test evaluation."""

    print()
    print("Loading test anomaly scores...")
    scores_df = load_anomaly_scores()

    print(
        f"  Loaded {len(scores_df):,} "
        "test anomaly scores."
    )

    print("Loading test proxy labels...")
    labels_df = load_proxy_labels()

    print(
        f"  Loaded {len(labels_df):,} "
        "test proxy labels."
    )

    print("Loading frozen validation threshold...")
    threshold = load_threshold()

    print(
        f"  Loaded threshold: {threshold:.6f}"
    )

    print("Validating inputs...")
    validate_inputs(
        scores_df,
        labels_df,
    )

    print("Merging scores and proxy labels...")
    merged = merge_evaluation_data(
        scores_df,
        labels_df,
    )

    y_true = (
        merged["proxy_anomaly"]
        .astype(int)
        .to_numpy()
    )

    anomaly_scores = (
        merged["anomaly_score"]
        .astype(float)
        .to_numpy()
    )

    print("Evaluating frozen threshold...")
    results = evaluate_threshold(
        y_true=y_true,
        anomaly_scores=anomaly_scores,
        threshold=threshold,
    )

    print_results(
        results,
        total_sequences=len(merged),
    )


if __name__ == "__main__":
    main()