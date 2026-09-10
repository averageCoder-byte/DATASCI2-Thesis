from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

BASELINE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASELINE_DIR.parent / "modeling"
DATA_DIR = MODEL_DIR / "data"

TEST_ERRORS_PATH = (
    DATA_DIR / "test_results" / "test_reconstruction_errors.npy"
)

TEST_LABELS_PATH = (
    DATA_DIR / "evaluation" / "test_proxy_labels.csv"
)

OUTPUT_DIR = BASELINE_DIR / "data" / "lstm_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = (
    OUTPUT_DIR / "test_raw_error_ranking_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"sequence_id", "proxy_anomaly"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )

    df = df[["sequence_id", "proxy_anomaly"]].copy()

    if df["sequence_id"].duplicated().any():
        raise ValueError(
            f"Duplicate sequence_id values found in {path}"
        )

    df["proxy_anomaly"] = df["proxy_anomaly"].astype(int)

    if not df["proxy_anomaly"].isin([0, 1]).all():
        raise ValueError(
            f"proxy_anomaly must contain only 0/1 in {path}"
        )

    return df


def validate_errors(errors: np.ndarray) -> None:
    if errors.ndim != 1:
        raise ValueError(
            "Test reconstruction errors must be 1-dimensional."
        )

    if not np.isfinite(errors).all():
        raise ValueError(
            "Test reconstruction errors contain non-finite values."
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LSTM AUTOENCODER + RAW ERROR RANKING BASELINE")
print("=" * 60)

test_errors = np.load(TEST_ERRORS_PATH)
test_labels = load_labels(TEST_LABELS_PATH)

validate_errors(test_errors)

if len(test_errors) != len(test_labels):
    raise ValueError(
        "Test reconstruction errors and labels have different lengths."
    )

test_true = test_labels["proxy_anomaly"].to_numpy()

print("\nTest dataset:")
print(f"  Sequences: {len(test_errors)}")
print(
    f"  Proxy-normal: {(test_true == 0).sum()}"
)
print(
    f"  Proxy-anomalous: {(test_true == 1).sum()}"
)


# ============================================================
# RAW ERROR RANKING
# ============================================================

roc_auc = roc_auc_score(
    test_true,
    test_errors,
)

pr_auc = average_precision_score(
    test_true,
    test_errors,
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

normal_errors = test_errors[test_true == 0]
anomalous_errors = test_errors[test_true == 1]


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print("\nRaw reconstruction-error ranking:")
print("  Higher reconstruction error = more anomalous")

print("\nThreshold-independent metrics:")
print(f"  ROC-AUC:             {roc_auc:.6f}")
print(f"  PR-AUC:              {pr_auc:.6f}")

print("\nReconstruction-error distribution:")

print("\n  Proxy-normal:")
print(f"    Mean:              {normal_errors.mean():.6f}")
print(f"    Median:            {np.median(normal_errors):.6f}")
print(f"    Std:               {normal_errors.std():.6f}")
print(f"    Min:               {normal_errors.min():.6f}")
print(f"    Max:               {normal_errors.max():.6f}")

print("\n  Proxy-anomalous:")
print(f"    Mean:              {anomalous_errors.mean():.6f}")
print(f"    Median:            {np.median(anomalous_errors):.6f}")
print(f"    Std:               {anomalous_errors.std():.6f}")
print(f"    Min:               {anomalous_errors.min():.6f}")
print(f"    Max:               {anomalous_errors.max():.6f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "test_sequences": len(test_errors),
            "proxy_anomalous": int(test_true.sum()),
            "proxy_anomalous_pct": (
                test_true.sum() / len(test_true) * 100
            ),
            "normal_error_mean": normal_errors.mean(),
            "normal_error_median": np.median(normal_errors),
            "normal_error_std": normal_errors.std(),
            "normal_error_min": normal_errors.min(),
            "normal_error_max": normal_errors.max(),
            "anomalous_error_mean": anomalous_errors.mean(),
            "anomalous_error_median": np.median(anomalous_errors),
            "anomalous_error_std": anomalous_errors.std(),
            "anomalous_error_min": anomalous_errors.min(),
            "anomalous_error_max": anomalous_errors.max(),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
    ]
)

results.to_csv(
    RESULT_PATH,
    index=False,
)

print("\nResults saved to:")
print(RESULT_PATH)

print("\nDone.")