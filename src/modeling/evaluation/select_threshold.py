from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


MODEL_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = MODEL_DIR / "data"

RECONSTRUCTION_ERROR_PATH = (
    DATA_DIR
    / "validation_results"
    / "validation_reconstruction_errors.npy"
)

GMM_PATH = (
    DATA_DIR
    / "gmm_results"
    / "final_gmm.joblib"
)

PROXY_LABEL_PATH = (
    DATA_DIR
    / "evaluation"
    / "validation_proxy_labels.csv"
)

OUTPUT_PATH = (
    DATA_DIR
    / "evaluation"
    / "selected_threshold.txt"
)


def load_validation_scores():
    """Load validation reconstruction errors and score them with the final GMM."""

    reconstruction_errors = np.load(
        RECONSTRUCTION_ERROR_PATH
    )

    if reconstruction_errors.ndim != 1:
        reconstruction_errors = reconstruction_errors.reshape(-1)

    print(
        f"Validation reconstruction errors: "
        f"{len(reconstruction_errors):,}"
    )

    gmm = joblib.load(GMM_PATH)

    print(f"GMM components: {gmm.n_components}")

    log_likelihood = gmm.score_samples(
        reconstruction_errors.reshape(-1, 1)
    )

    anomaly_scores = -log_likelihood

    print("\nValidation anomaly score summary:")
    print(f"  Mean:   {anomaly_scores.mean():.6f}")
    print(f"  Std:    {anomaly_scores.std():.6f}")
    print(f"  Min:    {anomaly_scores.min():.6f}")
    print(f"  Median: {np.median(anomaly_scores):.6f}")
    print(f"  Max:    {anomaly_scores.max():.6f}")

    return anomaly_scores


def load_proxy_labels():
    """Load validation sequence-level proxy labels."""

    df = pd.read_csv(PROXY_LABEL_PATH)

    required_columns = [
        "sequence_id",
        "proxy_anomaly",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        f"\nValidation proxy labels: "
        f"{len(df):,}"
    )

    return df


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
):
    """
    Select the anomaly-score threshold that maximizes
    validation balanced accuracy.
    """

    candidate_thresholds = np.unique(scores)

    best = None

    for threshold in candidate_thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

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

        balanced_accuracy = (
            balanced_accuracy_score(
                labels,
                predictions,
            )
        )

        result = {
            "threshold": threshold,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "balanced_accuracy": balanced_accuracy,
        }

        if (
            best is None
            or result["balanced_accuracy"]
            > best["balanced_accuracy"]
        ):
            best = result

    return best


def main():

    print("=" * 60)
    print("DIAGNOSIS 2C: VALIDATION THRESHOLD SELECTION")
    print("=" * 60)

    scores = load_validation_scores()

    proxy_df = load_proxy_labels()

    if len(scores) != len(proxy_df):
        raise ValueError(
            "Validation score count does not match "
            "validation proxy-label count."
        )

    labels = (
        proxy_df["proxy_anomaly"]
        .astype(int)
        .to_numpy()
    )

    print(
        f"\nProxy-normal sequences: "
        f"{(labels == 0).sum():,}"
    )

    print(
        f"Proxy-anomalous sequences: "
        f"{(labels == 1).sum():,}"
    )

    best = select_threshold(
        scores,
        labels,
    )

    print("\n" + "=" * 60)
    print("SELECTED VALIDATION THRESHOLD")
    print("(Criterion: maximum balanced accuracy)")
    print("=" * 60)

    print(
        f"Threshold:          "
        f"{best['threshold']:.6f}"
    )

    print(
        f"F1:                 "
        f"{best['f1']:.6f}"
    )

    print(
        f"Precision:          "
        f"{best['precision']:.6f}"
    )

    print(
        f"Recall:             "
        f"{best['recall']:.6f}"
    )

    print(
        f"Balanced accuracy:  "
        f"{best['balanced_accuracy']:.6f}"
    )

    predictions = (
    scores >= best["threshold"]
    ).astype(int)

    tn = int(
        ((labels == 0) & (predictions == 0)).sum()
    )

    fp = int(
        ((labels == 0) & (predictions == 1)).sum()
    )

    fn = int(
        ((labels == 1) & (predictions == 0)).sum()
    )

    tp = int(
        ((labels == 1) & (predictions == 1)).sum()
    )

    predicted_anomalous_pct = (
        predictions.mean() * 100
    )

    print("\nConfusion matrix:")
    print(f"  TN: {tn:,}")
    print(f"  FP: {fp:,}")
    print(f"  FN: {fn:,}")
    print(f"  TP: {tp:,}")

    print(
        f"\nPredicted anomalous: "
        f"{predictions.sum():,} "
        f"({predicted_anomalous_pct:.2f}%)"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        f"{best['threshold']:.10f}\n"
    )

    print(
        f"\nThreshold saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "\nIMPORTANT: This threshold is now "
        "selected using validation only."
    )

    print(
        "Freeze this threshold before evaluating "
        "the test set."
    )


if __name__ == "__main__":
    main()