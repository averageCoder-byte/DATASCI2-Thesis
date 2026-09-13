# DATASCI2-Thesis Instructions

This document provides setup and usage instructions for the DATASCI2 thesis repository, including local Python development, Apache Airflow with Docker, and Jupyter/VS Code notebooks.

## 1. Prerequisites

Install the following before working with the repository:

- Git
- Python 3.13
- Docker Desktop
- Docker Compose (included with current Docker Desktop installations)
- VS Code (recommended)

The project uses Python for the modeling and data-processing pipeline and Dockerized Apache Airflow for workflow orchestration.

## 2. Clone the Repository

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/averageCoder-bit/data-sci-2-thesis.git
cd data-sci-2-thesis
```

The `main` branch contains the current stable implementation.

## 3. Create a Python Virtual Environment

Create and activate a local virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, adjust the execution policy for the current user as appropriate for your environment.

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

## 4. Install Python Dependencies

Install the project dependencies from the root requirements file:

```powershell
pip install -r requirements.txt
```

The local virtual environment is primarily intended for development, analysis, notebooks, and running individual scripts.

## 5. Repository Structure

The main source directories are:

```text
src/
├── airflow/       # Apache Airflow DAGs, Docker configuration, and Airflow requirements
├── baseline/      # Baseline anomaly-detection implementations and outputs
├── modeling/      # LGMMA-X modeling, scoring, and evaluation
├── notebooks/     # Analysis and visualization notebooks
└── pipeline/      # Data ingestion, preprocessing, and feature engineering
```

Generated runtime files, local configuration, and experiment-specific artifacts may be excluded through `.gitignore`.

## 6. Running the Data and Modeling Scripts Locally

Individual Python scripts can be executed from their corresponding directories while the virtual environment is active.

For example:

```powershell
python <script_name>.py
```

Run scripts from the repository paths expected by the source code so that relative paths resolve correctly.

The main LGMMA-X workflow consists of data preparation, chronological splitting, feature scaling, gap-aware sequence construction, LSTM Autoencoder training, reconstruction-error generation, GMM fitting, anomaly scoring, threshold calibration, proxy-label evaluation, and test evaluation.

## 7. Running Apache Airflow with Docker

The Airflow project is located under:

```text
src/airflow/
```

Move to that directory before using Docker Compose:

```powershell
cd src\airflow
```

Start the Airflow services in detached mode:

```powershell
docker compose up -d
```

Check the running containers:

```powershell
docker compose ps
```

To view Airflow service logs:

```powershell
docker compose logs -f
```

To stop the services:

```powershell
docker compose down
```

### Rebuilding Airflow Images

Rebuild the Airflow image when `src/airflow/requirements.txt`, the Dockerfile, or other image-level dependencies change:

```powershell
docker compose build --no-cache
```

Then start the services again:

```powershell
docker compose up -d
```

The Airflow image installs the dependencies defined in:

```text
src/airflow/requirements.txt
```

This environment is separate from the repository-level `.venv`.

## 8. Airflow DAGs

The main thesis workflows are defined in:

```text
src/airflow/dags/
```

The LGMMA-X workflow includes tasks for the training and evaluation stages used in the thesis experiments.

When making changes to a DAG:

1. Stop or restart the relevant Airflow services if necessary.
2. Confirm the DAG is parsed successfully.
3. Trigger the DAG manually from the Airflow UI when an experimental run is required.
4. Review task logs and output artifacts after completion.

Do not commit local Airflow logs, local `.env` files, or runtime cache files unless they are explicitly intended to be version-controlled.

## 9. Accessing the Airflow Web Interface

After the Docker services are running, use the port configured by the Airflow Compose setup to access the Airflow web interface in a browser.

The exact credentials and local configuration are environment-specific. Keep local credentials and secrets out of Git.

## 10. Running Jupyter / VS Code Notebooks

The analysis notebooks are stored under:

```text
src/notebooks/
```

The principal notebooks currently used for the thesis analysis include:

```text
01_comparative_model_analysis.ipynb
02_statistical_comparison.ipynb
03_preliminary_dataset_visualization.ipynb
```

### VS Code

Open the repository in VS Code:

```powershell
code .
```

Select the project virtual environment as the Python/Jupyter kernel:

```text
.venv\Scripts\python.exe
```

Then open the required notebook under `src/notebooks/` and run its cells in order.

### Command-Line Jupyter

Jupyter can also be launched from the repository root:

```powershell
jupyter notebook
```

or:

```powershell
jupyter lab
```

## 11. Notebook Roles

### Notebook 01 — Comparative Model Analysis

Used for comparing LGMMA-X and the baseline models using the final held-out test results and related descriptive analysis.

### Notebook 02 — Statistical Comparison

Used for paired block-level comparison across models. The workflow includes normality assessment and the paired statistical procedures defined in the thesis methodology, together with Holm adjustment for multiple comparisons.

### Notebook 03 — Preliminary Dataset Visualization

Used for dataset exploration, test-sequence category analysis, proxy-label distributions, and visualization of the LGMMA-X test results by proxy anomaly condition.

## 12. Reproducibility and Artifact Alignment

The thesis experiments rely on multiple generated artifacts. When rerunning experiments, make sure that the following remain aligned:

- chronological train/validation/test splits
- scaled datasets
- sequence arrays
- sequence metadata
- trained LSTM Autoencoder
- reconstruction-error files
- frozen GMM model and test anomaly scores
- validation-calibrated thresholds
- proxy-label files
- baseline score and prediction files

Do not mix artifacts from earlier experimental runs with the corrected final experiment. In particular, sequence metadata must correspond to the same processed data and sequence construction logic used to generate the sequence arrays.

For the current corrected experiment, the held-out test population contains 21,749 sequences and the LGMMA-X validation-calibrated threshold is 0.719680.

## 13. Git Workflow

The current `main` branch is the canonical working branch for future thesis revisions.

Check the current state before making changes:

```powershell
git status
git branch --show-current
git log --oneline -5
```

Stage and commit intentional changes:

```powershell
git add <file>
git commit -m "Describe the change"
```

Push to GitHub:

```powershell
git push origin main
```

Avoid committing generated logs, local secrets, virtual-environment files, temporary diagnostic scripts, or ignored statistical-comparison artifacts.

## 14. Cleaning Generated Runtime Files

Airflow generates logs, Python creates `__pycache__` directories, and experiments may produce local model/data artifacts. Before deleting anything, inspect Git's view of the working tree:

```powershell
git status
```

Use `git clean -nd` as a dry run before any cleanup of untracked files:

```powershell
git clean -nd
```

Do not use `git clean -fdx` casually because it also removes ignored files and may delete local configuration or environment files.

## 15. Troubleshooting

### Docker port or container conflicts

Check running containers:

```powershell
docker ps
```

Stop the project stack when appropriate:

```powershell
docker compose down
```

Then restart:

```powershell
docker compose up -d
```

### File-lock problems on Windows

If Git reports that a directory cannot be deleted during a branch checkout or merge, close VS Code terminals, Jupyter kernels, Python processes, and other applications that may be using files in that directory before retrying.

### Notebook output does not match the thesis

Stop and verify artifact alignment before using the output. Do not replace expected values with older results simply to make a notebook pass its assertions.

## 16. Thesis Reporting Rule

The repository contains implementation and experimental evidence for the thesis. The written thesis should use results from the corrected, reproducible experiment and should not mix values from obsolete runs.

When an artifact, output, or notebook result changes, update the corresponding documentation and analysis only after confirming that the underlying pipeline and evaluation population remain aligned.
