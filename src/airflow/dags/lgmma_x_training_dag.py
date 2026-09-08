from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_ROOT = Path("/opt/airflow/project")
SRC_ROOT = PROJECT_ROOT / "src"
PIPELINE_ROOT = SRC_ROOT / "pipeline"
MODELING_ROOT = SRC_ROOT / "modeling"


default_args = {
    "owner": "datasci2",
    "retries": 0,
}


with DAG(
    dag_id="lgmma_x_training",
    description="Train the LGMMA-X anomaly detection pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=6),
    tags=["lgmma-x", "training", "xauusd"],
) as dag:

    run_etl = BashOperator(
        task_id="run_etl",
        bash_command=f"""
            cd {PIPELINE_ROOT}
            python load.py
        """,
    )

    split_data = BashOperator(
        task_id="split_data",
        bash_command=f"""
            python {MODELING_ROOT}/split.py
        """,
    )

    scale_data = BashOperator(
        task_id="scale_data",
        bash_command=f"""
            python {MODELING_ROOT}/scaling.py
        """,
    )

    create_sequences = BashOperator(
        task_id="create_sequences",
        bash_command=f"""
            python {MODELING_ROOT}/sequence.py
        """,
    )

    train_lstm_autoencoder = BashOperator(
        task_id="train_lstm_autoencoder",
        bash_command=f"""
            python {MODELING_ROOT}/train.py
        """,
        execution_timeout=timedelta(hours=4),
    )

    generate_train_reconstruction = BashOperator(
        task_id="generate_train_reconstruction",
        bash_command=f"""
            python {MODELING_ROOT}/train_reconstruction.py
        """,
    )

    train_gmm = BashOperator(
        task_id="train_gmm",
        bash_command=f"""
            python {MODELING_ROOT}/scoring/gmm.py
        """,
    )


    (
        run_etl
        >> split_data
        >> scale_data
        >> create_sequences
        >> train_lstm_autoencoder
        >> generate_train_reconstruction
        >> train_gmm
    )