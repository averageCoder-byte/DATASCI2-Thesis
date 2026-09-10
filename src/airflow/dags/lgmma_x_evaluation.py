from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_ROOT = Path("/opt/airflow/project")
SRC_ROOT = PROJECT_ROOT / "src"
MODELING_ROOT = SRC_ROOT / "modeling"
BASELINE_ROOT = SRC_ROOT / "baseline"


default_args = {
    "owner": "datasci2",
    "retries": 0,
}


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

    generate_proxy_labels = BashOperator(
        task_id="generate_proxy_labels",
        bash_command=f"""
            python {MODELING_ROOT}/evaluation/proxy_labels.py
        """,
    )

    select_threshold = BashOperator(
        task_id="select_threshold",
        bash_command=f"""
            python {MODELING_ROOT}/evaluation/select_threshold.py
        """,
    )

    score_test_data = BashOperator(
        task_id="score_test_data",
        bash_command=f"""
            python {MODELING_ROOT}/scoring/scoring.py
        """,
    )

    evaluate_test_threshold = BashOperator(
        task_id="evaluate_test_threshold",
        bash_command=f"""
            python {MODELING_ROOT}/evaluation/evaluate_test_threshold.py
        """,
    )

    compare_proxy_scores = BashOperator(
        task_id="compare_proxy_scores",
        bash_command=f"""
            python {MODELING_ROOT}/evaluation/compare_proxy_scores.py
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

    # -------------------------------------------------------------------------
    # Dependencies
    # -------------------------------------------------------------------------

    # Proxy labels are required by threshold selection and baseline
    # validation-based threshold selection.
    generate_proxy_labels >> select_threshold

    generate_proxy_labels >> [
        isolation_forest,
        one_class_svm,
        zscore,
        arima_garch,
        lstm_fixed,
        lstm_rank,
    ]

    # LGMMA-X test scoring uses the frozen GMM and test reconstruction errors
    # produced by the training DAG.
    score_test_data >> evaluate_test_threshold
    select_threshold >> evaluate_test_threshold

    # Proxy-score severity analysis requires final test predictions/scores.
    evaluate_test_threshold >> compare_proxy_scores

    # Baseline score-generation tasks depend on their corresponding models.
    isolation_forest >> generate_isolation_forest_scores
    one_class_svm >> generate_one_class_svm_scores
    zscore >> generate_zscore_scores
    lstm_fixed >> generate_lstm_fixed_scores
    lstm_rank >> generate_lstm_rank_scores

    # ARIMA-GARCH does not have a statistical-comparison score-generation
    # script in the current repository.

    # Final model summary waits for the LGMMA-X evaluation and all baselines.
    [
        evaluate_test_threshold,
        isolation_forest,
        one_class_svm,
        zscore,
        arima_garch,
        lstm_fixed,
        lstm_rank,
    ] >> summarize_all_models