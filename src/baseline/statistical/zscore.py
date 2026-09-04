from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "modeling" / "data"
BASELINE_DIR = BASE_DIR / "baseline" / "data" / "statistical"

BASELINE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_SEQUENCES = (
    MODEL_DIR
    / "sequences"
    / "X_train.npy"
)

VALIDATION_SEQUENCES = (
    MODEL_DIR
    / "sequences"
    / "X_validation.npy"
)

TEST_SEQUENCES = (
    MODEL_DIR
    / "sequences"
    / "X_test.npy"
)

VALIDATION_LABELS = (
    MODEL_DIR
    / "evaluation"
    / "validation_proxy_labels.csv"
)

TEST_LABELS = (
    MODEL_DIR
    / "evaluation"
    / "test_proxy_labels.csv"
)


# ============================================================
# Configuration
# ============================================================

Z_THRESHOLDS = [
    2.5,
    3.0,
    3.5,
]


# ============================================================
# Load data
# ============================================================

def load_data():

    X_train = np.load(
        TRAIN_SEQUENCES
    )

    X_validation = np.load(
        VALIDATION_SEQUENCES
    )

    X_test = np.load(
        TEST_SEQUENCES
    )

    validation_labels = pd.read_csv(
        VALIDATION_LABELS
    )

    test_labels = pd.read_csv(
        TEST_LABELS
    )

    required_column = "proxy_anomaly"

    if required_column not in validation_labels.columns:
        raise ValueError(
            f"Missing '{required_column}' "
            "in validation proxy labels."
        )

    if required_column not in test_labels.columns:
        raise ValueError(
            f"Missing '{required_column}' "
            "in test proxy labels."
        )

    return (
        X_train,
        X_validation,
        X_test,
        validation_labels,
        test_labels,
    )


# ============================================================
# Fit training distribution
# ============================================================

def fit_zscore_distribution(X_train):

    # Flatten all training timesteps while
    # preserving the seven feature dimensions.
    #
    # Shape:
    # (102132, 60, 7)
    #       ↓
    # (6127920, 7)

    X_train_flat = X_train.reshape(
        -1,
        X_train.shape[-1],
    )

    means = np.mean(
        X_train_flat,
        axis=0,
    )

    stds = np.std(
        X_train_flat,
        axis=0,
    )

    # Prevent division by zero for any
    # zero-variance feature.

    stds = np.where(
        stds == 0,
        1.0,
        stds,
    )

    return means, stds


# ============================================================
# Generate sequence-level Z-score
# ============================================================

def calculate_z_scores(
    X,
    means,
    stds,
):

    # Standardize each feature.

    z = (
        X - means
    ) / stds

    # Absolute deviation because the baseline
    # detects both unusually positive and unusually
    # negative observations.

    abs_z = np.abs(z)

    # A sequence is represented by its largest
    # standardized deviation across all 60 timesteps
    # and seven features.

    sequence_scores = np.max(
        abs_z,
        axis=(1, 2),
    )

    return sequence_scores


# ============================================================
# Evaluate threshold
# ============================================================

def evaluate_threshold(
    y_true,
    scores,
    threshold,
):

    predictions = (
        scores > threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    pr_auc = average_precision_score(
        y_true,
        scores,
    )

    roc_auc = roc_auc_score(
        y_true,
        scores,
    )

    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "predicted_anomalous": int(
            predictions.sum()
        ),
        "total_sequences": len(y_true),
    }


# ============================================================
# Select validation threshold
# ============================================================

def select_threshold(
    y_true,
    scores,
):

    validation_results = []

    best_threshold = None
    best_f1 = -np.inf
    best_pr_auc = -np.inf

    for threshold in Z_THRESHOLDS:

        metrics = evaluate_threshold(
            y_true,
            scores,
            threshold,
        )

        validation_results.append(
            metrics
        )

        # ----------------------------------------------------
        # Selection rule
        #
        # Primary:
        #   highest F1
        #
        # Secondary:
        #   highest PR-AUC
        # ----------------------------------------------------

        if (
            metrics["f1"] > best_f1
            or (
                np.isclose(
                    metrics["f1"],
                    best_f1,
                )
                and metrics["pr_auc"]
                > best_pr_auc
            )
        ):

            best_threshold = threshold
            best_f1 = metrics["f1"]
            best_pr_auc = metrics["pr_auc"]

    return (
        best_threshold,
        validation_results,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Z-SCORE STATISTICAL BASELINE")
    print("=" * 70)

    (
        X_train,
        X_validation,
        X_test,
        validation_labels,
        test_labels,
    ) = load_data()

    # --------------------------------------------------------
    # Sequence shapes
    # --------------------------------------------------------

    print("\nSequence shapes:")

    print(
        f"Train:      {X_train.shape}"
    )

    print(
        f"Validation: {X_validation.shape}"
    )

    print(
        f"Test:       {X_test.shape}"
    )

    # --------------------------------------------------------
    # Validate counts
    # --------------------------------------------------------

    if len(X_validation) != len(
        validation_labels
    ):

        raise ValueError(
            "Validation sequence count does not "
            "match validation labels."
        )

    if len(X_test) != len(
        test_labels
    ):

        raise ValueError(
            "Test sequence count does not "
            "match test labels."
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_validation = (
        validation_labels[
            "proxy_anomaly"
        ]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test_labels[
            "proxy_anomaly"
        ]
        .astype(int)
        .to_numpy()
    )

    print("\nValidation labels:")

    print(
        f"Normal:    "
        f"{(y_validation == 0).sum():,}"
    )

    print(
        f"Anomalous: "
        f"{(y_validation == 1).sum():,}"
    )

    print("\nTest labels:")

    print(
        f"Normal:    "
        f"{(y_test == 0).sum():,}"
    )

    print(
        f"Anomalous: "
        f"{(y_test == 1).sum():,}"
    )

    # ========================================================
    # FIT TRAINING DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("FITTING TRAINING Z-SCORE DISTRIBUTION")
    print("=" * 70)

    means, stds = fit_zscore_distribution(
        X_train
    )

    print("\nFeature means:")

    print(means)

    print("\nFeature standard deviations:")

    print(stds)

    # ========================================================
    # GENERATE Z-SCORES
    # ========================================================

    print("\nGenerating sequence-level Z-scores...")

    validation_scores = calculate_z_scores(
        X_validation,
        means,
        stds,
    )

    test_scores = calculate_z_scores(
        X_test,
        means,
        stds,
    )

    print("\nValidation score summary:")

    print(
        f"Mean:   {validation_scores.mean():.6f}"
    )

    print(
        f"Median: {np.median(validation_scores):.6f}"
    )

    print(
        f"Max:    {validation_scores.max():.6f}"
    )

    print("\nTest score summary:")

    print(
        f"Mean:   {test_scores.mean():.6f}"
    )

    print(
        f"Median: {np.median(test_scores):.6f}"
    )

    print(
        f"Max:    {test_scores.max():.6f}"
    )

    # ========================================================
    # VALIDATION THRESHOLD SELECTION
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD SELECTION")
    print("=" * 70)

    (
        selected_threshold,
        validation_results,
    ) = select_threshold(
        y_validation,
        validation_scores,
    )

    validation_df = pd.DataFrame(
        validation_results
    )

    print(
        validation_df[
            [
                "threshold",
                "f1",
                "pr_auc",
                "precision",
                "recall",
                "balanced_accuracy",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSelected threshold: "
        f"|Z| > {selected_threshold:.1f}"
    )

    selected_validation = next(
        result
        for result in validation_results
        if result["threshold"]
        == selected_threshold
    )

    print(
        "Validation F1: "
        f"{selected_validation['f1']:.6f}"
    )

    print(
        "Validation PR-AUC: "
        f"{selected_validation['pr_auc']:.6f}"
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_results = evaluate_threshold(
        y_test,
        test_scores,
        selected_threshold,
    )

    print(
        f"Threshold:          "
        f"|Z| > {selected_threshold:.1f}"
    )

    print(
        f"TN:                 "
        f"{test_results['tn']:,}"
    )

    print(
        f"FP:                 "
        f"{test_results['fp']:,}"
    )

    print(
        f"FN:                 "
        f"{test_results['fn']:,}"
    )

    print(
        f"TP:                 "
        f"{test_results['tp']:,}"
    )

    print(
        f"Precision:          "
        f"{test_results['precision']:.6f}"
    )

    print(
        f"Recall:             "
        f"{test_results['recall']:.6f}"
    )

    print(
        f"F1:                 "
        f"{test_results['f1']:.6f}"
    )

    print(
        f"Balanced accuracy:  "
        f"{test_results['balanced_accuracy']:.6f}"
    )

    print(
        f"ROC-AUC:            "
        f"{test_results['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC:             "
        f"{test_results['pr_auc']:.6f}"
    )

    print(
        f"Predicted anomalous: "
        f"{test_results['predicted_anomalous']:,}"
    )

    print(
        f"Total sequences:     "
        f"{test_results['total_sequences']:,}"
    )

    # ========================================================
    # Save validation selection
    # ========================================================

    validation_results_path = (
        BASELINE_DIR
        / "zscore_validation_selection.csv"
    )

    validation_df.to_csv(
        validation_results_path,
        index=False,
    )

    # ========================================================
    # Save training distribution
    # ========================================================

    distribution_df = pd.DataFrame({
        "feature_index": np.arange(
            len(means)
        ),
        "mean": means,
        "std": stds,
    })

    distribution_path = (
        BASELINE_DIR
        / "zscore_training_distribution.csv"
    )

    distribution_df.to_csv(
        distribution_path,
        index=False,
    )

    # ========================================================
    # Save threshold
    # ========================================================

    threshold_path = (
        BASELINE_DIR
        / "zscore_threshold.txt"
    )

    threshold_path.write_text(
        f"{selected_threshold:.1f}",
        encoding="utf-8",
    )

    # ========================================================
    # Save final test results
    # ========================================================

    results = {
        "method": "Z-score",
        "threshold": selected_threshold,
        "tn": test_results["tn"],
        "fp": test_results["fp"],
        "fn": test_results["fn"],
        "tp": test_results["tp"],
        "precision": test_results["precision"],
        "recall": test_results["recall"],
        "f1": test_results["f1"],
        "balanced_accuracy": (
            test_results[
                "balanced_accuracy"
            ]
        ),
        "roc_auc": test_results[
            "roc_auc"
        ],
        "pr_auc": test_results[
            "pr_auc"
        ],
        "predicted_anomalous": (
            test_results[
                "predicted_anomalous"
            ]
        ),
        "total_sequences": (
            test_results[
                "total_sequences"
            ]
        ),
    }

    results_path = (
        BASELINE_DIR
        / "zscore_test_results.csv"
    )

    pd.DataFrame(
        [results]
    ).to_csv(
        results_path,
        index=False,
    )

    print("\nSaved:")

    print(
        validation_results_path
    )

    print(
        distribution_path
    )

    print(
        threshold_path
    )

    print(
        results_path
    )


if __name__ == "__main__":
    main()