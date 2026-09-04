from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from arch import arch_model
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = SRC_DIR / "modeling" / "data"

TRAIN_PATH = DATA_DIR / "split" / "train.parquet"
VALIDATION_PATH = DATA_DIR / "split" / "validation.parquet"
TEST_PATH = DATA_DIR / "split" / "test.parquet"

VALIDATION_LABELS_PATH = (
    DATA_DIR / "evaluation" / "validation_proxy_labels.csv"
)

TEST_LABELS_PATH = (
    DATA_DIR / "evaluation" / "test_proxy_labels.csv"
)

VALIDATION_METADATA_PATH = (
    DATA_DIR
    / "sequence_metadata"
    / "validation_sequence_metadata.parquet"
)

TEST_METADATA_PATH = (
    DATA_DIR
    / "sequence_metadata"
    / "test_sequence_metadata.parquet"
)

OUTPUT_DIR = (
    SRC_DIR / "baseline" / "data" / "traditional"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 60

# ARIMA search space
ARIMA_P = range(0, 6)
ARIMA_D = range(0, 3)
ARIMA_Q = range(0, 6)

# GARCH configurations.
GARCH_CONFIGS = [
    (1, 1),
    (1, 2),
    (2, 1),
]

# Volatility-adjusted residual threshold candidates.
THRESHOLD_CANDIDATES = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
]


# ============================================================
# DATA LOADING
# ============================================================

def load_split(path):
    df = pd.read_parquet(path)

    required = {
        "Date",
        "log_return",
        "segment_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    return df


def load_labels(path):
    df = pd.read_csv(path)

    required = {
        "sequence_id",
        "proxy_anomaly",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    return df


# ============================================================
# ARIMA MODEL SEARCH
# ============================================================

def search_arima(train_returns):
    """
    Exhaustive ARIMA(p,d,q) search.

    p = 0..5
    d = 0..2
    q = 0..5

    Models are fitted exclusively on the training series.

    Selection is based primarily on BIC, with AIC also
    recorded for transparency.
    """

    results = []

    total = (
        len(list(ARIMA_P))
        * len(list(ARIMA_D))
        * len(list(ARIMA_Q))
    )

    completed = 0

    print(
        f"\nSearching {total} ARIMA configurations..."
    )

    for p in ARIMA_P:

        for d in ARIMA_D:

            for q in ARIMA_Q:

                order = (p, d, q)

                completed += 1

                try:
                    trend = "c" if d == 0 else "n"

                    model = ARIMA(
                        train_returns,
                        order=order,
                        trend=trend,
                    )

                    fitted = model.fit()

                    results.append(
                        {
                            "p": p,
                            "d": d,
                            "q": q,
                            "aic": fitted.aic,
                            "bic": fitted.bic,
                            "log_likelihood": fitted.llf,
                            "converged": True,
                        }
                    )

                except Exception as exc:

                    results.append(
                        {
                            "p": p,
                            "d": d,
                            "q": q,
                            "aic": np.nan,
                            "bic": np.nan,
                            "log_likelihood": np.nan,
                            "converged": False,
                            "error": str(exc),
                        }
                    )

                if (
                    completed % 10 == 0
                    or completed == total
                ):
                    print(
                        f"  Completed "
                        f"{completed}/{total}"
                    )

    results_df = pd.DataFrame(results)

    valid = results_df[
        results_df["converged"]
        & results_df["bic"].notna()
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "No ARIMA configurations converged."
        )

    # BIC is the primary model-selection criterion.
    selected_row = (
        valid
        .sort_values(
            ["bic", "aic"]
        )
        .iloc[0]
    )

    selected_order = (
        int(selected_row["p"]),
        int(selected_row["d"]),
        int(selected_row["q"]),
    )

    results_df = results_df.sort_values(
        "bic",
        na_position="last",
    )

    results_df.to_csv(
        OUTPUT_DIR / "arima_model_selection.csv",
        index=False,
    )

    print(
        "\nSelected ARIMA:"
        f" ARIMA{selected_order}"
    )

    print(
        f"AIC: {selected_row['aic']:.6f}"
    )

    print(
        f"BIC: {selected_row['bic']:.6f}"
    )

    return selected_order, results_df


def fit_selected_arima(
    train_returns,
    order,
):
    print(
        f"\nFitting selected ARIMA{order}..."
    )

    model = ARIMA(
        train_returns,
        order=order,
        trend="c",
    )

    fitted = model.fit()

    return fitted


# ============================================================
# ARIMA RESIDUAL GENERATION
# ============================================================

def generate_sequential_residuals(
    fitted_model,
    values,
):
    """
    Generate one-step-ahead residuals using the already-fitted
    ARIMA model without refitting its parameters.

    Statsmodels append(..., refit=False) updates the state using
    observed values while keeping model parameters fixed.

    This prevents validation/test information from changing
    the estimated ARIMA coefficients.
    """

    extended = fitted_model.append(
        values,
        refit=False,
    )

    residuals = np.asarray(
        extended.resid[-len(values):],
        dtype=float,
    )

    return residuals


# ============================================================
# GARCH
# ============================================================

def fit_garch(
    residuals,
    p,
    q,
):
    """
    Fit a zero-mean GARCH(p,q) model to ARIMA residuals.

    Residuals are scaled by 100 for numerical stability.
    """

    scaled_residuals = (
        residuals * 100.0
    )

    model = arch_model(
        scaled_residuals,
        mean="Zero",
        vol="GARCH",
        p=p,
        q=q,
        dist="normal",
        rescale=False,
    )

    fitted = model.fit(
        disp="off"
    )

    return fitted


def extract_garch_parameters(
    fitted,
    p,
    q,
):
    params = fitted.params

    omega = float(
        params["omega"]
    )

    alpha = np.array(
        [
            float(
                params[f"alpha[{i}]"]
            )
            for i in range(1, p + 1)
        ]
    )

    beta = np.array(
        [
            float(
                params[f"beta[{i}]"]
            )
            for i in range(1, q + 1)
        ]
    )

    return omega, alpha, beta


def recursive_garch_volatility(
    residuals,
    fitted,
    p,
    q,
    initial_residuals,
    initial_variances,
):
    """
    Recursively calculate conditional volatility using
    fixed GARCH parameters.

    This supports GARCH(1,1), GARCH(1,2), and GARCH(2,1).
    """

    omega, alpha, beta = (
        extract_garch_parameters(
            fitted,
            p,
            q,
        )
    )

    # Work in the same percentage-scaled space used for fitting.
    scaled_residuals = (
        np.asarray(residuals)
        * 100.0
    )

    history_residuals = list(
        np.asarray(initial_residuals)
        * 100.0
    )

    history_variances = list(
        np.asarray(initial_variances)
        ** 2
        * 10000.0
    )

    variances = []

    for residual in scaled_residuals:

        variance = omega

        # ARCH terms
        for i in range(p):

            lag = i + 1

            if lag <= len(history_residuals):
                variance += (
                    alpha[i]
                    * history_residuals[-lag] ** 2
                )

        # GARCH terms
        for i in range(q):

            lag = i + 1

            if lag <= len(history_variances):
                variance += (
                    beta[i]
                    * history_variances[-lag]
                )

        variance = max(
            variance,
            1e-12,
        )

        variances.append(
            variance
        )

        history_residuals.append(
            residual
        )

        history_variances.append(
            variance
        )

    # Convert percentage variance back to
    # original-return variance.
    return (
        np.sqrt(
            np.asarray(variances)
        )
        / 100.0
    )


# ============================================================
# VOLATILITY-ADJUSTED ANOMALY SCORE
# ============================================================

def calculate_adjusted_score(
    residuals,
    volatility,
):
    volatility = np.maximum(
        volatility,
        1e-12,
    )

    return (
        np.abs(residuals)
        / volatility
    )


# ============================================================
# SEQUENCE AGGREGATION
# ============================================================

def create_sequence_scores(
    df,
    timestep_scores,
    metadata_path,
):
    """
    Aggregate timestep-level standardized residuals
    into the exact existing 60-candle sequence metadata.

    Sequence boundaries are therefore identical to the
    proposed model and all other sequence-level baselines.
    """

    metadata = pd.read_parquet(
        metadata_path
    )

    metadata["start_timestamp"] = pd.to_datetime(
        metadata["start_timestamp"],
        utc=True,
    )

    metadata["end_timestamp"] = pd.to_datetime(
        metadata["end_timestamp"],
        utc=True,
    )

    required = {
        "sequence_id",
        "segment_id",
        "start_timestamp",
        "end_timestamp",
    }

    missing = (
        required
        - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            f"Missing sequence metadata columns: "
            f"{sorted(missing)}"
        )

    working = df[
        [
            "Date",
            "segment_id",
        ]
    ].copy()

    working["score"] = (
        np.asarray(
            timestep_scores
        )
    )

    output = []

    for _, row in metadata.iterrows():

        segment = working[
            working["segment_id"]
            == row["segment_id"]
        ]

        mask = (
            (segment["Date"]
             >= row["start_timestamp"])
            &
            (segment["Date"]
             <= row["end_timestamp"])
        )

        window = segment.loc[mask]

        if len(window) != SEQUENCE_LENGTH:

            raise ValueError(
                f"Sequence "
                f"{row['sequence_id']} contains "
                f"{len(window)} candles instead of "
                f"{SEQUENCE_LENGTH}."
            )

        output.append(
            {
                "sequence_id":
                    row["sequence_id"],

                "segment_id":
                    row["segment_id"],

                "start_timestamp":
                    row["start_timestamp"],

                "end_timestamp":
                    row["end_timestamp"],

                # A sequence is anomalous if at least one
                # timestep violates the volatility-adjusted
                # residual boundary.
                "anomaly_score":
                    window["score"].max(),
            }
        )

    return pd.DataFrame(output)


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def evaluate_threshold(
    y_true,
    scores,
    threshold,
):
    predictions = (
        scores >= threshold
    ).astype(int)

    return {
        "threshold":
            threshold,

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),
    }


def calibrate_threshold(
    y_true,
    scores,
):
    rows = []

    for threshold in THRESHOLD_CANDIDATES:

        rows.append(
            evaluate_threshold(
                y_true,
                scores,
                threshold,
            )
        )

    results = pd.DataFrame(
        rows
    )

    # Balanced accuracy is used for calibration so that
    # proxy-normal and proxy-anomalous sequences receive
    # equal weight despite the high anomaly prevalence.
    selected = (
        results
        .sort_values(
            [
                "balanced_accuracy",
                "f1",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return (
        float(
            selected["threshold"]
        ),
        results,
    )


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(
    y_true,
    scores,
    threshold,
):
    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    return {
        "threshold":
            threshold,

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                scores,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                scores,
            ),

        "predicted_anomalous":
            int(
                predictions.sum()
            ),

        "total":
            int(
                len(y_true)
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ARIMA-GARCH BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    print("\nLoading data...")

    train = load_split(
        TRAIN_PATH
    )

    validation = load_split(
        VALIDATION_PATH
    )

    test = load_split(
        TEST_PATH
    )

    validation_labels = load_labels(
        VALIDATION_LABELS_PATH
    )

    test_labels = load_labels(
        TEST_LABELS_PATH
    )

    print(
        f"Train rows:      {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows:       {len(test):,}"
    )

    # --------------------------------------------------------
    # 2. Training returns
    # --------------------------------------------------------

    train_returns = (
        train["log_return"]
        .astype(float)
        .values
    )

    validation_returns = (
        validation["log_return"]
        .astype(float)
        .values
    )

    test_returns = (
        test["log_return"]
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # 3. Exhaustive ARIMA search
    # --------------------------------------------------------

    selected_arima_order, arima_search = (
        search_arima(
            train_returns
        )
    )

    # --------------------------------------------------------
    # 4. Fit selected ARIMA
    # --------------------------------------------------------

    arima_result = fit_selected_arima(
        train_returns,
        selected_arima_order,
    )

    train_residuals = np.asarray(
        arima_result.resid,
        dtype=float,
    )

    # --------------------------------------------------------
    # 5. Generate validation residuals
    # --------------------------------------------------------

    print(
        "\nGenerating validation "
        "ARIMA residuals..."
    )

    validation_residuals = (
        generate_sequential_residuals(
            arima_result,
            validation_returns,
        )
    )

    # --------------------------------------------------------
    # 6. Generate test residuals
    # --------------------------------------------------------

    print(
        "Generating test ARIMA residuals..."
    )

    # Update the fitted ARIMA state through validation
    # before generating test residuals.
    arima_after_validation = (
        arima_result.append(
            validation_returns,
            refit=False,
        )
    )

    test_residuals = (
        generate_sequential_residuals(
            arima_after_validation,
            test_returns,
        )
    )

    # --------------------------------------------------------
    # 7. Test GARCH configurations
    # --------------------------------------------------------

    garch_selection = []

    best_configuration = None
    best_validation_ba = -np.inf

    print(
        "\nEvaluating GARCH configurations..."
    )

    for garch_p, garch_q in GARCH_CONFIGS:

        print(
            f"\nGARCH({garch_p},{garch_q})"
        )

        try:

            garch_result = fit_garch(
                train_residuals,
                garch_p,
                garch_q,
            )

            # Training conditional volatility.
            train_conditional_volatility = (
                np.asarray(
                    garch_result
                    .conditional_volatility
                )
                / 100.0
            )

            # We need enough historical residuals/variance
            # values to initialize recursive forecasting.
            initial_history_length = max(
                garch_p,
                garch_q,
                2,
            )

            initial_residuals = (
                train_residuals[
                    -initial_history_length:
                ]
            )

            initial_variances = (
                train_conditional_volatility[
                    -initial_history_length:
                ]
            )

            # Validation volatility
            validation_volatility = (
                recursive_garch_volatility(
                    validation_residuals,
                    garch_result,
                    garch_p,
                    garch_q,
                    initial_residuals,
                    initial_variances,
                )
            )

            # Test volatility starts after validation.
            combined_residuals = np.concatenate(
                [
                    validation_residuals,
                ]
            )

            validation_variance_history = (
                recursive_garch_volatility(
                    validation_residuals,
                    garch_result,
                    garch_p,
                    garch_q,
                    initial_residuals,
                    initial_variances,
                )
            )

            test_initial_residuals = (
                np.concatenate(
                    [
                        initial_residuals,
                        validation_residuals,
                    ]
                )[-initial_history_length:]
            )

            test_initial_variances = (
                np.concatenate(
                    [
                        initial_variances,
                        validation_variance_history,
                    ]
                )[-initial_history_length:]
            )

            test_volatility = (
                recursive_garch_volatility(
                    test_residuals,
                    garch_result,
                    garch_p,
                    garch_q,
                    test_initial_residuals,
                    test_initial_variances,
                )
            )

            validation_scores = (
                calculate_adjusted_score(
                    validation_residuals,
                    validation_volatility,
                )
            )

            test_scores = (
                calculate_adjusted_score(
                    test_residuals,
                    test_volatility,
                )
            )

            # ------------------------------------------------
            # Sequence aggregation
            # ------------------------------------------------

            validation_sequences = (
                create_sequence_scores(
                    validation,
                    validation_scores,
                    VALIDATION_METADATA_PATH,
                )
            )

            test_sequences = (
                create_sequence_scores(
                    test,
                    test_scores,
                    TEST_METADATA_PATH,
                )
            )

            validation_merged = (
                validation_sequences
                .merge(
                    validation_labels[
                        [
                            "sequence_id",
                            "proxy_anomaly",
                        ]
                    ],
                    on="sequence_id",
                    how="inner",
                    validate="one_to_one",
                )
            )

            test_merged = (
                test_sequences
                .merge(
                    test_labels[
                        [
                            "sequence_id",
                            "proxy_anomaly",
                        ]
                    ],
                    on="sequence_id",
                    how="inner",
                    validate="one_to_one",
                )
            )

            y_validation = (
                validation_merged[
                    "proxy_anomaly"
                ]
                .astype(int)
                .values
            )

            validation_sequence_scores = (
                validation_merged[
                    "anomaly_score"
                ]
                .astype(float)
                .values
            )

            threshold, threshold_results = (
                calibrate_threshold(
                    y_validation,
                    validation_sequence_scores,
                )
            )

            validation_predictions = (
                validation_sequence_scores
                >= threshold
            ).astype(int)

            validation_ba = (
                balanced_accuracy_score(
                    y_validation,
                    validation_predictions,
                )
            )

            validation_f1 = (
                f1_score(
                    y_validation,
                    validation_predictions,
                    zero_division=0,
                )
            )

            validation_precision = (
                precision_score(
                    y_validation,
                    validation_predictions,
                    zero_division=0,
                )
            )

            validation_recall = (
                recall_score(
                    y_validation,
                    validation_predictions,
                    zero_division=0,
                )
            )

            garch_selection.append(
                {
                    "garch_p": garch_p,
                    "garch_q": garch_q,
                    "aic": garch_result.aic,
                    "bic": garch_result.bic,
                    "validation_threshold": threshold,
                    "validation_precision":
                        validation_precision,
                    "validation_recall":
                        validation_recall,
                    "validation_f1":
                        validation_f1,
                    "validation_balanced_accuracy":
                        validation_ba,
                }
            )

            print(
                f"  AIC: {garch_result.aic:.6f}"
            )

            print(
                f"  BIC: {garch_result.bic:.6f}"
            )

            print(
                f"  Threshold: {threshold:.6f}"
            )

            print(
                f"  Validation BA: "
                f"{validation_ba:.6f}"
            )

            print(
                f"  Validation F1: "
                f"{validation_f1:.6f}"
            )

            # Select GARCH according to validation
            # anomaly-detection performance.
            if validation_ba > best_validation_ba:

                best_validation_ba = (
                    validation_ba
                )

                best_configuration = {
                    "garch_p": garch_p,
                    "garch_q": garch_q,
                    "threshold": threshold,
                    "garch_result": garch_result,
                    "validation_scores":
                        validation_sequence_scores,
                    "test_scores":
                        test_merged[
                            "anomaly_score"
                        ]
                        .astype(float)
                        .values,
                    "validation_merged":
                        validation_merged,
                    "test_merged":
                        test_merged,
                    "threshold_results":
                        threshold_results,
                }

        except Exception as exc:

            print(
                f"  FAILED: {exc}"
            )

            garch_selection.append(
                {
                    "garch_p": garch_p,
                    "garch_q": garch_q,
                    "aic": np.nan,
                    "bic": np.nan,
                    "validation_threshold": np.nan,
                    "validation_precision": np.nan,
                    "validation_recall": np.nan,
                    "validation_f1": np.nan,
                    "validation_balanced_accuracy":
                        np.nan,
                    "error": str(exc),
                }
            )

    garch_selection_df = pd.DataFrame(
        garch_selection
    )

    garch_selection_df.to_csv(
        OUTPUT_DIR
        / "garch_validation_selection.csv",
        index=False,
    )

    if best_configuration is None:
        raise RuntimeError(
            "No GARCH configuration succeeded."
        )

    # --------------------------------------------------------
    # 8. Freeze selected configuration
    # --------------------------------------------------------

    selected_garch_p = (
        best_configuration["garch_p"]
    )

    selected_garch_q = (
        best_configuration["garch_q"]
    )

    selected_threshold = (
        best_configuration["threshold"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED ARIMA-GARCH CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"ARIMA order: "
        f"{selected_arima_order}"
    )

    print(
        f"GARCH: "
        f"({selected_garch_p},"
        f"{selected_garch_q})"
    )

    print(
        f"Threshold: "
        f"{selected_threshold:.6f}"
    )

    print(
        f"Validation balanced accuracy: "
        f"{best_validation_ba:.6f}"
    )

    # --------------------------------------------------------
    # 9. Save threshold selection
    # --------------------------------------------------------

    best_configuration[
        "threshold_results"
    ].to_csv(
        OUTPUT_DIR
        / "arima_garch_threshold_selection.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 10. Final frozen test evaluation
    # --------------------------------------------------------

    test_merged = (
        best_configuration[
            "test_merged"
        ]
    )

    y_test = (
        test_merged[
            "proxy_anomaly"
        ]
        .astype(int)
        .values
    )

    test_scores = (
        test_merged[
            "anomaly_score"
        ]
        .astype(float)
        .values
    )

    test_results = evaluate_test(
        y_test,
        test_scores,
        selected_threshold,
    )

    test_results_df = pd.DataFrame(
        [test_results]
    )

    test_results_df.to_csv(
        OUTPUT_DIR
        / "arima_garch_test_results.csv",
        index=False,
    )

    # Continuous sequence scores
    test_merged.to_csv(
        OUTPUT_DIR
        / "arima_garch_test_scores.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 11. Save model
    # --------------------------------------------------------

    joblib.dump(
        {
            "arima_order":
                selected_arima_order,

            "arima_result":
                arima_result,

            "garch_p":
                selected_garch_p,

            "garch_q":
                selected_garch_q,

            "garch_result":
                best_configuration[
                    "garch_result"
                ],

            "threshold":
                selected_threshold,
        },
        OUTPUT_DIR
        / "arima_garch_model.joblib",
    )

    (
        OUTPUT_DIR
        / "arima_garch_threshold.txt"
    ).write_text(
        f"{selected_threshold:.10f}\n"
    )

    # --------------------------------------------------------
    # 12. Print final result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    for key, value in test_results.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.6f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    print(
        "\nSaved outputs to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()