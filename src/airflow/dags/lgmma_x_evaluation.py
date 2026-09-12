from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path("/opt/airflow/project")
SRC_ROOT = PROJECT_ROOT / "src"

MODELING_ROOT = SRC_ROOT / "modeling"
BASELINE_ROOT = SRC_ROOT / "baseline"

PROCESSED_DATA_PATH = (
    SRC_ROOT
    / "pipeline"
    / "data"
    / "processed"
    / "xauusd_5m_processed.parquet"
)

VALIDATION_METADATA_PATH = (
    MODELING_ROOT
    / "data"
    / "sequence_metadata"
    / "validation_sequence_metadata.parquet"
)

TEST_METADATA_PATH = (
    MODELING_ROOT
    / "data"
    / "sequence_metadata"
    / "test_sequence_metadata.parquet"
)

VALIDATION_PROXY_LABEL_PATH = (
    MODELING_ROOT
    / "data"
    / "evaluation"
    / "validation_proxy_labels.csv"
)

TEST_PROXY_LABEL_PATH = (
    MODELING_ROOT
    / "data"
    / "evaluation"
    / "test_proxy_labels.csv"
)


# =============================================================================
# SPLIT TIMESTAMPS (UTC)
# =============================================================================

TRAIN_START = "2023-11-06 21:35:00+00:00"
TRAIN_END = "2025-10-28 14:50:00+00:00"

VALIDATION_START = "2025-10-28 14:55:00+00:00"
VALIDATION_END = "2026-04-01 11:10:00+00:00"

TEST_START = "2026-04-01 11:15:00+00:00"
TEST_END = "2026-09-02 23:55:00+00:00"


# =============================================================================
# DEFAULT ARGS
# =============================================================================

default_args = {
    "owner": "datasci2",
    "retries": 0,
}


# =============================================================================
# DAG
# =============================================================================

with DAG(
    dag_id="lgmma_x_evaluation",
    description="Evaluate LGMMA-X and baseline anomaly detection models",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=6),
    tags=["lgmma-x", "evaluation", "xauusd"],
) as dag:

    # -------------------------------------------------------------------------
    # LGMMA-X evaluation preparation
    # -------------------------------------------------------------------------

    generate_validation_proxy_labels = BashOperator(
        task_id="generate_validation_proxy_labels",
        bash_command=f"""
            python {MODELING_ROOT}/evaluation/proxy_labels.py \
                --data "{PROCESSED_DATA_PATH}" \
                --metadata "{VALIDATION_METADATA_PATH}" \
                --train-start "{TRAIN_START}" \
                --train-end "{TRAIN_END}" \
                --target-start "{VALIDATION_START}" \
                --target-end "{VALIDATION_END}" \
                --output "{VALIDATION_PROXY_LABEL_PATH}"
        """,
    )

    generate_test_proxy_labels = BashOperator(
        task_id="generate_test_proxy_labels",
        bash_command=f"""
            python {MODELING_ROOT}/evaluation/proxy_labels.py \
                --data "{PROCESSED_DATA_PATH}" \
                --metadata "{TEST_METADATA_PATH}" \
                --train-start "{TRAIN_START}" \
                --train-end "{TRAIN_END}" \
                --target-start "{TEST_START}" \
                --target-end "{TEST_END}" \
                --output "{TEST_PROXY_LABEL_PATH}"
        """,
    )

    # Validation threshold selection
    select_threshold = BashOperator(
        task_id="select_threshold",
        bash_command=f"""
            python {MODELING_ROOT}/evaluation/select_threshold.py
        """,
    )

    # Test anomaly scoring
    score_test_data = BashOperator(
        task_id="score_test_data",
        bash_command=f"""
            python {MODELING_ROOT}/scoring/scoring.py
        """,
    )

    # Final held-out test evaluation
    evaluate_test_threshold = BashOperator(
        task_id="evaluate_test_threshold",
        bash_command=f"""
            python {MODELING_ROOT}/evaluation/evaluate_test_threshold.py
        """,
    )

    # Statistical/proxy-score analysis
    compare_proxy_scores = BashOperator(
        task_id="compare_proxy_scores",
        bash_command=f"""
            python {MODELING_ROOT}/evaluation/compare_proxy_scores.py \
                --scores "{MODELING_ROOT}/data/gmm_results/test_anomaly_scores.csv" \
                --labels "{TEST_PROXY_LABEL_PATH}"
        """,
    )

    # -------------------------------------------------------------------------
    # Baseline models
    # -------------------------------------------------------------------------

    isolation_forest = BashOperator(
        task_id="isolation_forest",
        bash_command=f"""
            python {BASELINE_ROOT}/classical_ml/isolation_forest.py
        """,
    )

    one_class_svm = BashOperator(
        task_id="one_class_svm",
        bash_command=f"""
            python {BASELINE_ROOT}/classical_ml/one_class_svm.py
        """,
    )

    zscore = BashOperator(
        task_id="zscore",
        bash_command=f"""
            python {BASELINE_ROOT}/statistical/zscore.py
        """,
    )

    arima_garch = BashOperator(
        task_id="arima_garch",
        bash_command=f"""
            python {BASELINE_ROOT}/traditional/arima_garch.py
        """,
    )

    lstm_fixed = BashOperator(
        task_id="lstm_fixed",
        bash_command=f"""
            python {BASELINE_ROOT}/lstm_ablation/fixed_threshold.py
        """,
    )

    lstm_rank = BashOperator(
        task_id="lstm_rank",
        bash_command=f"""
            python {BASELINE_ROOT}/lstm_ablation/raw_error_ranking.py
        """,
    )

    # -------------------------------------------------------------------------
    # Statistical comparison score generation
    # -------------------------------------------------------------------------

    generate_isolation_forest_scores = BashOperator(
        task_id="generate_isolation_forest_scores",
        bash_command=f"""
            python {BASELINE_ROOT}/statistical_comparison/generate_isolation_forest_test_scores.py
        """,
    )

    generate_lstm_fixed_scores = BashOperator(
        task_id="generate_lstm_fixed_scores",
        bash_command=f"""
            python {BASELINE_ROOT}/statistical_comparison/generate_lstm_fixed_test_scores.py
        """,
    )

    generate_lstm_rank_scores = BashOperator(
        task_id="generate_lstm_rank_scores",
        bash_command=f"""
            python {BASELINE_ROOT}/statistical_comparison/generate_lstm_rank_test_scores.py
        """,
    )

    generate_one_class_svm_scores = BashOperator(
        task_id="generate_one_class_svm_scores",
        bash_command=f"""
            python {BASELINE_ROOT}/statistical_comparison/generate_one_class_svm_test_scores.py
        """,
    )

    generate_zscore_scores = BashOperator(
        task_id="generate_zscore_scores",
        bash_command=f"""
            python {BASELINE_ROOT}/statistical_comparison/generate_zscore_test_scores.py
        """,
    )

    # -------------------------------------------------------------------------
    # Model comparison summary
    # -------------------------------------------------------------------------

    summarize_all_models = BashOperator(
        task_id="summarize_all_models",
        bash_command=f"""
            python {BASELINE_ROOT}/summarize_all_models.py
        """,
    )

    # =========================================================================
    # DEPENDENCIES
    # =========================================================================

    # -------------------------------------------------------------------------
    # Validation proxy labels → threshold calibration
    # -------------------------------------------------------------------------

    generate_validation_proxy_labels >> select_threshold

    # -------------------------------------------------------------------------
    # Test scoring
    # -------------------------------------------------------------------------

    # scoring.py uses the frozen GMM and test reconstruction errors
    # produced by the finalized training pipeline.
    score_test_data >> evaluate_test_threshold

    # -------------------------------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------------------------------

    # Final evaluation requires both:
    #   1. the test anomaly scores
    #   2. the threshold selected using validation data
    select_threshold >> evaluate_test_threshold

    # Test proxy labels are also required by final evaluation.
    generate_test_proxy_labels >> evaluate_test_threshold

    # -------------------------------------------------------------------------
    # Proxy-score statistical analysis
    # -------------------------------------------------------------------------

    evaluate_test_threshold >> compare_proxy_scores

    # -------------------------------------------------------------------------
    # Baseline models
    # -------------------------------------------------------------------------

    generate_test_proxy_labels >> [
        isolation_forest,
        one_class_svm,
        zscore,
        arima_garch,
        lstm_fixed,
        lstm_rank,
    ]

    # Baseline score-generation tasks depend on their corresponding models.
    isolation_forest >> generate_isolation_forest_scores
    one_class_svm >> generate_one_class_svm_scores
    zscore >> generate_zscore_scores
    lstm_fixed >> generate_lstm_fixed_scores
    lstm_rank >> generate_lstm_rank_scores

    # ARIMA-GARCH does not currently have a statistical-comparison
    # score-generation script.

    # -------------------------------------------------------------------------
    # Final model summary
    # -------------------------------------------------------------------------

    [
        evaluate_test_threshold,
        isolation_forest,
        one_class_svm,
        zscore,
        arima_garch,
        lstm_fixed,
        lstm_rank,
    ] >> summarize_all_models