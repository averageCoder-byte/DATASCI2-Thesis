"""
summarize_all_models.py

Consolidates saved test results from all anomaly-detection models
into one CSV for display and downstream comparative analysis.

This script:
- DOES NOT train models
- DOES NOT fit models
- DOES NOT tune thresholds
- DOES NOT perform statistical comparison
- DOES NOT rank models
- Reads only previously saved evaluation artifacts
- Reconstructs LGMMA-X test metrics from saved scores/labels
- Exports exactly one CSV file

Output:
    src/baseline/data/master_model_comparison.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODELING_DATA = ROOT / "src" / "modeling" / "data"
BASELINE_DATA = ROOT / "src" / "baseline" / "data"

OUTPUT_FILE = BASELINE_DATA / "master_model_comparison.csv"


# ============================================================
# MODEL ORDER
# ============================================================

MODEL_ORDER = [
    "LGMMA-X (LSTM-AE + GMM)",
    "ARIMA-GARCH",
    "Z-score",
    "Isolation Forest",
    "One-Class SVM",
    "LSTM-AE (Fixed Threshold)",
    "LSTM-AE (Raw Error Ranking)",
]


# ============================================================
# COMMON METRIC COLUMNS
# ============================================================

METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "predicted_anomalous",
    "total_sequences",
]


# ============================================================
# HELPERS
# ============================================================

def empty_result(model_name):
    """Create an empty result dictionary."""
    result = {"model": model_name}

    for column in METRIC_COLUMNS:
        result[column] = np.nan

    return result


def normalize_result_columns(df):
    """
    Normalize common metric column names so that different
    result CSV formats can be combined.
    """

    rename_map = {
        "tn": "true_negative",
        "fp": "false_positive",
        "fn": "false_negative",
        "tp": "true_positive",
        "total": "total_sequences",
        "f1_score": "f1",
        "balanced_accuracy_score": "balanced_accuracy",
        "roc_auc_score": "roc_auc",
        "pr_auc_score": "pr_auc",
        "average_precision": "pr_auc",
    }

    df = df.rename(columns=rename_map)

    return df


def read_single_result_csv(path, model_name):
    """
    Read a model's saved test-result CSV.

    The CSV is expected to contain one row of final metrics.
    """

    if not path.exists():
        print(f"[WARNING] Result not found for: {model_name}")
        print(f"          Expected: {path}")
        return None

    try:
        df = pd.read_csv(path)

        if df.empty:
            print(f"[WARNING] Empty result file for: {model_name}")
            return None

        df = normalize_result_columns(df)

        row = df.iloc[-1]

        result = empty_result(model_name)

        for column in METRIC_COLUMNS:
            if column in df.columns:
                result[column] = pd.to_numeric(
                    row[column],
                    errors="coerce"
                )

        return result

    except Exception as exc:
        print(f"[WARNING] Could not read {model_name}: {exc}")
        return None


# ============================================================
# LGMMA-X
# ============================================================

def load_lgmma_x_result():
    """
    Reconstruct LGMMA-X final test metrics from:

        modeling/data/gmm_results/test_anomaly_scores.csv
        modeling/data/evaluation/test_proxy_labels.csv
        modeling/data/evaluation/selected_threshold.txt

    The threshold was selected previously on validation data and
    is NOT optimized here.

    Prediction rule:

        anomaly_score >= selected_threshold
            -> anomalous
        anomaly_score < selected_threshold
            -> normal
    """

    model_name = "LGMMA-X (LSTM-AE + GMM)"

    score_file = (
        MODELING_DATA
        / "gmm_results"
        / "test_anomaly_scores.csv"
    )

    label_file = (
        MODELING_DATA
        / "evaluation"
        / "test_proxy_labels.csv"
    )

    threshold_file = (
        MODELING_DATA
        / "evaluation"
        / "selected_threshold.txt"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    missing = []

    for path in [score_file, label_file, threshold_file]:
        if not path.exists():
            missing.append(path)

    if missing:
        print(f"[WARNING] Could not reconstruct {model_name}.")
        for path in missing:
            print(f"          Missing: {path}")
        return None

    try:
        # ----------------------------------------------------
        # Load anomaly scores
        # ----------------------------------------------------

        scores_df = pd.read_csv(score_file)

        # ----------------------------------------------------
        # Load proxy labels
        # ----------------------------------------------------

        labels_df = pd.read_csv(label_file)

        # ----------------------------------------------------
        # Normalize possible column names
        # ----------------------------------------------------

        scores_df = scores_df.rename(
            columns={
                "score": "anomaly_score",
                "gmm_score": "anomaly_score",
            }
        )

        labels_df = labels_df.rename(
            columns={
                "label": "proxy_anomaly",
                "proxy_label": "proxy_anomaly",
            }
        )

        if "anomaly_score" not in scores_df.columns:
            raise ValueError(
                "Could not find 'anomaly_score' column in "
                f"{score_file.name}"
            )

        if "proxy_anomaly" not in labels_df.columns:
            raise ValueError(
                "Could not find 'proxy_anomaly' column in "
                f"{label_file.name}"
            )

        # ----------------------------------------------------
        # Merge by sequence_id when available
        # ----------------------------------------------------

        if (
            "sequence_id" in scores_df.columns
            and "sequence_id" in labels_df.columns
        ):
            df = pd.merge(
                scores_df,
                labels_df,
                on="sequence_id",
                how="inner",
                validate="one_to_one",
            )

        else:
            if len(scores_df) != len(labels_df):
                raise ValueError(
                    "Score and proxy-label files have different "
                    "row counts and no sequence_id is available."
                )

            df = pd.concat(
                [
                    scores_df.reset_index(drop=True),
                    labels_df.reset_index(drop=True),
                ],
                axis=1,
            )

        # ----------------------------------------------------
        # Load frozen validation threshold
        # ----------------------------------------------------

        threshold_text = threshold_file.read_text().strip()
        threshold = float(threshold_text)

        # ----------------------------------------------------
        # Convert labels
        # ----------------------------------------------------

        y_true = (
            pd.to_numeric(
                df["proxy_anomaly"],
                errors="coerce"
            )
            .astype(int)
            .to_numpy()
        )

        anomaly_scores = (
            pd.to_numeric(
                df["anomaly_score"],
                errors="coerce"
            )
            .to_numpy()
        )

        valid = (
            np.isfinite(anomaly_scores)
            & np.isfinite(y_true)
        )

        y_true = y_true[valid]
        anomaly_scores = anomaly_scores[valid]

        # ----------------------------------------------------
        # Apply frozen threshold
        # ----------------------------------------------------

        y_pred = (
            anomaly_scores >= threshold
        ).astype(int)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        result = empty_result(model_name)

        result.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "predicted_anomalous": int(y_pred.sum()),
                "total_sequences": int(len(y_true)),
            }
        )

        print(
            f"[OK] Reconstructed {model_name} "
            f"using frozen threshold {threshold:.6f}"
        )

        return result

    except Exception as exc:
        print(f"[WARNING] Could not reconstruct {model_name}: {exc}")
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("MODEL RESULT SUMMARIZER")
    print("=" * 72)
    print()

    results = []

    # --------------------------------------------------------
    # 1. LGMMA-X
    # --------------------------------------------------------

    lgmma_result = load_lgmma_x_result()

    if lgmma_result is not None:
        results.append(lgmma_result)

    # --------------------------------------------------------
    # 2. ARIMA-GARCH
    # --------------------------------------------------------

    arima_path = (
        BASELINE_DATA
        / "traditional"
        / "arima_garch_test_results.csv"
    )

    arima_result = read_single_result_csv(
        arima_path,
        "ARIMA-GARCH",
    )

    if arima_result is not None:
        results.append(arima_result)

    # --------------------------------------------------------
    # 3. Z-score
    # --------------------------------------------------------

    zscore_path = (
        BASELINE_DATA
        / "statistical"
        / "zscore_test_results.csv"
    )

    zscore_result = read_single_result_csv(
        zscore_path,
        "Z-score",
    )

    if zscore_result is not None:
        results.append(zscore_result)

    # --------------------------------------------------------
    # 4. Isolation Forest
    # --------------------------------------------------------

    isolation_path = (
        BASELINE_DATA
        / "classical_ml"
        / "isolation_forest_test_results.csv"
    )

    isolation_result = read_single_result_csv(
        isolation_path,
        "Isolation Forest",
    )

    if isolation_result is not None:
        results.append(isolation_result)

    # --------------------------------------------------------
    # 5. One-Class SVM
    # --------------------------------------------------------

    ocsvm_path = (
        BASELINE_DATA
        / "classical_ml"
        / "one_class_svm_test_results.csv"
    )

    ocsvm_result = read_single_result_csv(
        ocsvm_path,
        "One-Class SVM",
    )

    if ocsvm_result is not None:
        results.append(ocsvm_result)

    # --------------------------------------------------------
    # 6. LSTM-AE Fixed Threshold
    # --------------------------------------------------------

    fixed_threshold_path = (
        BASELINE_DATA
        / "lstm_ablation"
        / "test_fixed_threshold_results.csv"
    )

    fixed_result = read_single_result_csv(
        fixed_threshold_path,
        "LSTM-AE (Fixed Threshold)",
    )

    if fixed_result is not None:
        results.append(fixed_result)

    # --------------------------------------------------------
    # 7. LSTM-AE Raw Error Ranking
    # --------------------------------------------------------

    raw_error_path = (
        BASELINE_DATA
        / "lstm_ablation"
        / "test_raw_error_ranking_results.csv"
    )

    raw_result = read_single_result_csv(
        raw_error_path,
        "LSTM-AE (Raw Error Ranking)",
    )

    if raw_result is not None:
        # Raw error ranking is intentionally threshold-free.
        # Therefore binary metrics remain blank.
        for column in [
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
            "predicted_anomalous",
        ]:
            raw_result[column] = np.nan

        results.append(raw_result)

    # ========================================================
    # BUILD MASTER TABLE
    # ========================================================

    if not results:
        print()
        print("[ERROR] No model results were found.")
        return

    master_df = pd.DataFrame(results)

    # Preserve intended model order
    master_df["model"] = pd.Categorical(
        master_df["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    master_df = (
        master_df
        .sort_values("model")
        .reset_index(drop=True)
    )

    # Convert categorical back to string
    master_df["model"] = master_df["model"].astype(str)

    # ========================================================
    # ROUND METRICS FOR DISPLAY
    # ========================================================

    percentage_metrics = [
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
    ]

    for column in percentage_metrics:
        if column in master_df.columns:
            master_df[column] = master_df[column].round(6)

    # ========================================================
    # EXPORT ONLY ONE FILE
    # ========================================================

    BASELINE_DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # TERMINAL DISPLAY
    # ========================================================

    print()
    print("=" * 72)
    print("MODELS FOUND")
    print("=" * 72)

    for model in master_df["model"]:
        print(f"  ✓ {model}")

    print()
    print("=" * 72)
    print("TEST METRICS")
    print("=" * 72)
    print()

    display_columns = [
        "model",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
    ]

    print(
        master_df[display_columns].to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("OUTPUT")
    print("=" * 72)
    print()
    print(OUTPUT_FILE)
    print()
    print(f"Rows: {len(master_df)}")
    print("Exported exactly one CSV file.")
    print()


if __name__ == "__main__":
    main()