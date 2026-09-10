from pathlib import Path

import joblib
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
from sklearn.svm import OneClassSVM


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "modeling" / "data"
BASELINE_DIR = BASE_DIR / "baseline" / "data" / "classical_ml"

BASELINE_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_SEQUENCES = MODEL_DIR / "sequences" / "X_train.npy"
VALIDATION_SEQUENCES = MODEL_DIR / "sequences" / "X_validation.npy"
TEST_SEQUENCES = MODEL_DIR / "sequences" / "X_test.npy"

VALIDATION_LABELS = (
    MODEL_DIR / "evaluation" / "validation_proxy_labels.csv"
)

TEST_LABELS = (
    MODEL_DIR / "evaluation" / "test_proxy_labels.csv"
)


# ============================================================
# Configuration search space
# ============================================================

KERNEL = "rbf"

NU_OPTIONS = [
    0.01,
    0.025,
    0.05,
]

GAMMA_OPTIONS = [
    "scale",
    "auto",
]


# ============================================================
# Load data
# ============================================================

def load_data():

    X_train = np.load(TRAIN_SEQUENCES)
    X_validation = np.load(VALIDATION_SEQUENCES)
    X_test = np.load(TEST_SEQUENCES)

    validation_labels = pd.read_csv(
        VALIDATION_LABELS
    )

    test_labels = pd.read_csv(
        TEST_LABELS
    )

    required_column = "proxy_anomaly"

    if required_column not in validation_labels.columns:
        raise ValueError(
            f"Missing '{required_column}' in validation proxy labels."
        )

    if required_column not in test_labels.columns:
        raise ValueError(
            f"Missing '{required_column}' in test proxy labels."
        )

    return (
        X_train,
        X_validation,
        X_test,
        validation_labels,
        test_labels,
    )


# ============================================================
# Flatten sequences
# ============================================================

def flatten_sequences(X):

    return X.reshape(
        X.shape[0],
        -1,
    )


# ============================================================
# Generate anomaly scores
# ============================================================

def anomaly_scores(model, X):

    # One-Class SVM decision_function():
    #
    # larger = more normal
    #
    # Negate so that:
    #
    # larger = more anomalous

    return -model.decision_function(X)


# ============================================================
# Validation threshold selection
# ============================================================

def select_threshold(
    y_true,
    scores,
):

    candidate_thresholds = np.unique(scores)

    best_threshold = None
    best_balanced_accuracy = -np.inf

    for threshold in candidate_thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        score = balanced_accuracy_score(
            y_true,
            predictions,
        )

        if score > best_balanced_accuracy:

            best_balanced_accuracy = score
            best_threshold = threshold

    return (
        best_threshold,
        best_balanced_accuracy,
    )


# ============================================================
# Threshold evaluation
# ============================================================

def evaluate_threshold(
    y_true,
    scores,
    threshold,
):

    predictions = (
        scores >= threshold
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
        "predicted_anomalous": int(
            predictions.sum()
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ONE-CLASS SVM BASELINE")
    print("=" * 70)

    (
        X_train,
        X_validation,
        X_test,
        validation_labels,
        test_labels,
    ) = load_data()

    # --------------------------------------------------------
    # Sequence information
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

    if len(X_validation) != len(validation_labels):
        raise ValueError(
            "Validation sequence count does not match "
            "validation labels."
        )

    if len(X_test) != len(test_labels):
        raise ValueError(
            "Test sequence count does not match "
            "test labels."
        )

    # --------------------------------------------------------
    # Flatten sequences
    # --------------------------------------------------------

    X_train_flat = flatten_sequences(
        X_train
    )

    X_validation_flat = flatten_sequences(
        X_validation
    )

    X_test_flat = flatten_sequences(
        X_test
    )

    print("\nFlattened shape:")
    print(
        f"Train:      {X_train_flat.shape}"
    )
    print(
        f"Validation: {X_validation_flat.shape}"
    )
    print(
        f"Test:       {X_test_flat.shape}"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_validation = (
        validation_labels["proxy_anomaly"]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test_labels["proxy_anomaly"]
        .astype(int)
        .to_numpy()
    )

    print("\nValidation labels:")
    print(
        f"Normal:    {(y_validation == 0).sum():,}"
    )
    print(
        f"Anomalous: {(y_validation == 1).sum():,}"
    )

    print("\nTest labels:")
    print(
        f"Normal:    {(y_test == 0).sum():,}"
    )
    print(
        f"Anomalous: {(y_test == 1).sum():,}"
    )

    # ========================================================
    # VALIDATION MODEL SELECTION
    # ========================================================

    validation_results = []

    best_model = None
    best_nu = None
    best_gamma = None
    best_threshold = None
    best_validation_score = -np.inf

    print("\n" + "=" * 70)
    print("VALIDATION MODEL SELECTION")
    print("=" * 70)

    configuration_number = 0

    for nu in NU_OPTIONS:

        for gamma in GAMMA_OPTIONS:

            configuration_number += 1

            print("\n" + "-" * 70)
            print(
                f"Configuration {configuration_number}/6"
            )
            print(
                f"Kernel: {KERNEL}"
            )
            print(
                f"Nu:     {nu}"
            )
            print(
                f"Gamma:  {gamma}"
            )
            print("-" * 70)

            model = OneClassSVM(
                kernel=KERNEL,
                nu=nu,
                gamma=gamma,
            )

            print("Fitting One-Class SVM...")

            model.fit(
                X_train_flat
            )

            print("Model fitted.")

            print(
                "Generating validation scores..."
            )

            validation_scores = anomaly_scores(
                model,
                X_validation_flat,
            )

            threshold, validation_balanced_accuracy = (
                select_threshold(
                    y_validation,
                    validation_scores,
                )
            )

            validation_metrics = evaluate_threshold(
                y_validation,
                validation_scores,
                threshold,
            )

            validation_results.append({
                "kernel": KERNEL,
                "nu": nu,
                "gamma": gamma,
                "threshold": threshold,
                "precision": validation_metrics[
                    "precision"
                ],
                "recall": validation_metrics[
                    "recall"
                ],
                "f1": validation_metrics[
                    "f1"
                ],
                "balanced_accuracy": (
                    validation_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "tn": validation_metrics["tn"],
                "fp": validation_metrics["fp"],
                "fn": validation_metrics["fn"],
                "tp": validation_metrics["tp"],
            })

            print(
                f"Threshold: {threshold:.6f}"
            )

            print(
                "Balanced accuracy: "
                f"{validation_balanced_accuracy:.6f}"
            )

            print(
                f"F1: "
                f"{validation_metrics['f1']:.6f}"
            )

            print(
                f"Precision: "
                f"{validation_metrics['precision']:.6f}"
            )

            print(
                f"Recall: "
                f"{validation_metrics['recall']:.6f}"
            )

            # ------------------------------------------------
            # Select best configuration
            # ------------------------------------------------

            if (
                validation_balanced_accuracy
                > best_validation_score
            ):

                best_validation_score = (
                    validation_balanced_accuracy
                )

                best_model = model
                best_nu = nu
                best_gamma = gamma
                best_threshold = threshold

    # ========================================================
    # Save validation model-selection results
    # ========================================================

    validation_results_df = pd.DataFrame(
        validation_results
    )

    validation_results_path = (
        BASELINE_DIR
        / "one_class_svm_validation_selection.csv"
    )

    validation_results_df.to_csv(
        validation_results_path,
        index=False,
    )

    # ========================================================
    # Selected configuration
    # ========================================================

    print("\n" + "=" * 70)
    print("SELECTED CONFIGURATION")
    print("=" * 70)

    print(
        f"Kernel:       {KERNEL}"
    )

    print(
        f"Nu:           {best_nu}"
    )

    print(
        f"Gamma:        {best_gamma}"
    )

    print(
        f"Threshold:    {best_threshold:.6f}"
    )

    print(
        "Validation balanced accuracy: "
        f"{best_validation_score:.6f}"
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        "Generating test anomaly scores..."
    )

    test_scores = anomaly_scores(
        best_model,
        X_test_flat,
    )

    test_metrics = evaluate_threshold(
        y_test,
        test_scores,
        best_threshold,
    )

    # Threshold-independent metrics

    roc_auc = roc_auc_score(
        y_test,
        test_scores,
    )

    pr_auc = average_precision_score(
        y_test,
        test_scores,
    )

    print(
        f"Kernel:             {KERNEL}"
    )

    print(
        f"Nu:                 {best_nu}"
    )

    print(
        f"Gamma:              {best_gamma}"
    )

    print(
        f"Threshold:          {best_threshold:.6f}"
    )

    print(
        f"TN:                 "
        f"{test_metrics['tn']:,}"
    )

    print(
        f"FP:                 "
        f"{test_metrics['fp']:,}"
    )

    print(
        f"FN:                 "
        f"{test_metrics['fn']:,}"
    )

    print(
        f"TP:                 "
        f"{test_metrics['tp']:,}"
    )

    print(
        f"Precision:          "
        f"{test_metrics['precision']:.6f}"
    )

    print(
        f"Recall:             "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"F1:                 "
        f"{test_metrics['f1']:.6f}"
    )

    print(
        "Balanced accuracy:  "
        f"{test_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"ROC-AUC:            "
        f"{roc_auc:.6f}"
    )

    print(
        f"PR-AUC:             "
        f"{pr_auc:.6f}"
    )

    print(
        "Predicted anomalous: "
        f"{test_metrics['predicted_anomalous']:,}"
    )

    print(
        "Total sequences:     "
        f"{len(y_test):,}"
    )

    # ========================================================
    # Save selected model
    # ========================================================

    model_path = (
        BASELINE_DIR
        / "one_class_svm.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    # ========================================================
    # Save threshold
    # ========================================================

    threshold_path = (
        BASELINE_DIR
        / "one_class_svm_threshold.txt"
    )

    threshold_path.write_text(
        f"{best_threshold:.12f}",
        encoding="utf-8",
    )

    # ========================================================
    # Save final test results
    # ========================================================

    final_results = {
        "kernel": KERNEL,
        "nu": best_nu,
        "gamma": best_gamma,
        "threshold": best_threshold,
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "balanced_accuracy": test_metrics[
            "balanced_accuracy"
        ],
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "predicted_anomalous": test_metrics[
            "predicted_anomalous"
        ],
        "total_sequences": len(y_test),
    }

    results_path = (
        BASELINE_DIR
        / "one_class_svm_test_results.csv"
    )

    pd.DataFrame(
        [final_results]
    ).to_csv(
        results_path,
        index=False,
    )

    print("\nSaved:")
    print(model_path)
    print(threshold_path)
    print(validation_results_path)
    print(results_path)


if __name__ == "__main__":
    main()