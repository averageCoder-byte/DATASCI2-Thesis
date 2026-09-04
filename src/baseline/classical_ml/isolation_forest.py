from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
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
# Configuration
# ============================================================

RANDOM_STATE = 42

# Configurations specified by the thesis methodology
N_ESTIMATORS_OPTIONS = [100, 200]

CONTAMINATION = "auto"


# ============================================================
# Load data
# ============================================================

def load_data():

    X_train = np.load(TRAIN_SEQUENCES)
    X_validation = np.load(VALIDATION_SEQUENCES)
    X_test = np.load(TEST_SEQUENCES)

    validation_labels = pd.read_csv(VALIDATION_LABELS)
    test_labels = pd.read_csv(TEST_LABELS)

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
    """
    Convert:

        (samples, 60, 7)

    into:

        (samples, 420)
    """

    return X.reshape(X.shape[0], -1)


# ============================================================
# Anomaly scores
# ============================================================

def anomaly_scores(model, X):
    """
    Isolation Forest score_samples():

        larger = more normal

    Negating produces:

        larger = more anomalous
    """

    return -model.score_samples(X)


# ============================================================
# Threshold selection
# ============================================================

def select_threshold(y_true, scores):

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
# Evaluate threshold
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
    print("ISOLATION FOREST BASELINE")
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
    print(f"Train:      {X_train.shape}")
    print(f"Validation: {X_validation.shape}")
    print(f"Test:       {X_test.shape}")

    if len(X_validation) != len(validation_labels):
        raise ValueError(
            "Validation sequence count does not match "
            "validation labels."
        )

    if len(X_test) != len(test_labels):
        raise ValueError(
            "Test sequence count does not match test labels."
        )

    # --------------------------------------------------------
    # Flatten sequences
    # --------------------------------------------------------

    X_train_flat = flatten_sequences(X_train)
    X_validation_flat = flatten_sequences(X_validation)
    X_test_flat = flatten_sequences(X_test)

    print("\nFlattened shape:")
    print(f"Train:      {X_train_flat.shape}")
    print(f"Validation: {X_validation_flat.shape}")
    print(f"Test:       {X_test_flat.shape}")

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
    # MODEL SELECTION
    # ========================================================

    validation_results = []

    best_model = None
    best_n_estimators = None
    best_threshold = None
    best_validation_score = -np.inf
    best_validation_scores = None

    print("\n" + "=" * 70)
    print("VALIDATION MODEL SELECTION")
    print("=" * 70)

    for n_estimators in N_ESTIMATORS_OPTIONS:

        print(
            f"\nTesting n_estimators={n_estimators}..."
        )

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_train_flat)

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

        threshold_metrics = evaluate_threshold(
            y_validation,
            validation_scores,
            threshold,
        )

        validation_results.append({
            "n_estimators": n_estimators,
            "contamination": CONTAMINATION,
            "threshold": threshold,
            "precision": threshold_metrics[
                "precision"
            ],
            "recall": threshold_metrics[
                "recall"
            ],
            "f1": threshold_metrics[
                "f1"
            ],
            "balanced_accuracy": (
                threshold_metrics[
                    "balanced_accuracy"
                ]
            ),
            "tn": threshold_metrics["tn"],
            "fp": threshold_metrics["fp"],
            "fn": threshold_metrics["fn"],
            "tp": threshold_metrics["tp"],
        })

        print(
            f"Threshold: {threshold:.6f}"
        )

        print(
            "Balanced accuracy: "
            f"{validation_balanced_accuracy:.6f}"
        )

        print(
            f"F1: {threshold_metrics['f1']:.6f}"
        )

        print(
            f"Precision: "
            f"{threshold_metrics['precision']:.6f}"
        )

        print(
            f"Recall: "
            f"{threshold_metrics['recall']:.6f}"
        )

        # ----------------------------------------------------
        # Select best configuration
        # ----------------------------------------------------

        if (
            validation_balanced_accuracy
            > best_validation_score
        ):

            best_validation_score = (
                validation_balanced_accuracy
            )

            best_model = model
            best_n_estimators = n_estimators
            best_threshold = threshold
            best_validation_scores = (
                validation_scores
            )

    # --------------------------------------------------------
    # Save validation model-selection results
    # --------------------------------------------------------

    validation_results_df = pd.DataFrame(
        validation_results
    )

    validation_results_path = (
        BASELINE_DIR
        / "isolation_forest_validation_selection.csv"
    )

    validation_results_df.to_csv(
        validation_results_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SELECTED CONFIGURATION")
    print("=" * 70)

    print(
        f"n_estimators: {best_n_estimators}"
    )

    print(
        f"contamination: {CONTAMINATION}"
    )

    print(
        f"Threshold: {best_threshold:.6f}"
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

    # Generate test scores using the selected model
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
        f"n_estimators:       {best_n_estimators}"
    )

    print(
        f"Threshold:          {best_threshold:.6f}"
    )

    print(
        f"TN:                 {test_metrics['tn']:,}"
    )

    print(
        f"FP:                 {test_metrics['fp']:,}"
    )

    print(
        f"FN:                 {test_metrics['fn']:,}"
    )

    print(
        f"TP:                 {test_metrics['tp']:,}"
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
        f"Balanced accuracy:  "
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
    # Save final model
    # ========================================================

    model_path = (
        BASELINE_DIR
        / "isolation_forest.joblib"
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
        / "isolation_forest_threshold.txt"
    )

    threshold_path.write_text(
        f"{best_threshold:.12f}",
        encoding="utf-8",
    )

    # ========================================================
    # Save final test results
    # ========================================================

    final_results = {
        "n_estimators": best_n_estimators,
        "contamination": CONTAMINATION,
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
        / "isolation_forest_test_results.csv"
    )

    pd.DataFrame([final_results]).to_csv(
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