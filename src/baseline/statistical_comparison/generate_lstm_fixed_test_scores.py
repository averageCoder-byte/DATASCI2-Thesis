from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "modeling" / "data"
BASELINE_DIR = BASE_DIR / "baseline" / "data" / "lstm_ablation"
STATISTICAL_COMPARISON_DIR = (
	BASE_DIR / "baseline" / "statistical_comparison"
)

STATISTICAL_COMPARISON_DIR.mkdir(
	parents=True,
	exist_ok=True,
)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

TEST_ERRORS = (
	MODEL_DIR
	/ "test_results"
	/ "test_reconstruction_errors.npy"
)

TEST_LABELS = (
	MODEL_DIR
	/ "evaluation"
	/ "test_proxy_labels.csv"
)

TEST_METADATA = (
	MODEL_DIR
	/ "sequence_metadata"
	/ "test_sequence_metadata.parquet"
)

THRESHOLD_PATH = (
	BASELINE_DIR
	/ "fixed_threshold.txt"
)


# ------------------------------------------------------------
# Output file
# ------------------------------------------------------------

OUTPUT_PATH = (
	STATISTICAL_COMPARISON_DIR
	/ "lstm_fixed_test_scores.csv"
)


# ============================================================
# Load data
# ============================================================

def load_data():
	"""Load frozen LSTM reconstruction scores and test metadata."""

	input_paths = {
		"test reconstruction errors": TEST_ERRORS,
		"test proxy labels": TEST_LABELS,
		"test sequence metadata": TEST_METADATA,
		"fixed threshold": THRESHOLD_PATH,
	}

	for description, path in input_paths.items():
		if not path.exists():
			raise FileNotFoundError(
				f"Missing {description}:\n{path}"
			)

	scores = np.load(TEST_ERRORS)

	if scores.ndim != 1:
		raise ValueError(
			"Expected one reconstruction error per test sequence. "
			f"Received shape: {scores.shape}"
		)

	if not np.isfinite(scores).all():
		raise ValueError(
			"Test reconstruction errors contain non-finite values."
		)

	test_labels = pd.read_csv(TEST_LABELS)
	test_metadata = pd.read_parquet(TEST_METADATA)
	threshold = float(
		THRESHOLD_PATH.read_text(
			encoding="utf-8"
		).strip()
	)

	return (
		scores,
		test_labels,
		test_metadata,
		threshold,
	)


# ============================================================
# Validate inputs
# ============================================================

def validate_inputs(
	scores,
	test_labels,
	test_metadata,
):
	"""Verify sequence counts, required columns, and unique IDs."""

	required_label_columns = {
		"sequence_id",
		"proxy_anomaly",
	}

	required_metadata_columns = {
		"sequence_id",
		"segment_id",
		"start_timestamp",
		"end_timestamp",
	}

	missing_label_columns = (
		required_label_columns
		- set(test_labels.columns)
	)

	if missing_label_columns:
		raise ValueError(
			"Test proxy labels are missing required "
			f"columns: {sorted(missing_label_columns)}"
		)

	missing_metadata_columns = (
		required_metadata_columns
		- set(test_metadata.columns)
	)

	if missing_metadata_columns:
		raise ValueError(
			"Test sequence metadata is missing required "
			f"columns: {sorted(missing_metadata_columns)}"
		)

	n_sequences = len(scores)

	if len(test_labels) != n_sequences:
		raise ValueError(
			"Test sequence count does not match proxy labels. "
			f"Scores: {n_sequences:,}; "
			f"Labels: {len(test_labels):,}"
		)

	if len(test_metadata) != n_sequences:
		raise ValueError(
			"Test sequence count does not match sequence metadata. "
			f"Scores: {n_sequences:,}; "
			f"Metadata: {len(test_metadata):,}"
		)

	if test_labels["sequence_id"].duplicated().any():
		raise ValueError(
			"Duplicate sequence_id values found in "
			"test proxy labels."
		)

	if test_metadata["sequence_id"].duplicated().any():
		raise ValueError(
			"Duplicate sequence_id values found in "
			"test sequence metadata."
		)


# ============================================================
# Build standardized output
# ============================================================

def build_output(
	scores,
	threshold,
	test_labels,
	test_metadata,
):
	"""Build the standardized sequence-level score table."""

	scores_df = pd.DataFrame({
		"sequence_id": test_metadata[
			"sequence_id"
		].to_numpy(),
		"segment_id": test_metadata[
			"segment_id"
		].to_numpy(),
		"start_timestamp": test_metadata[
			"start_timestamp"
		].to_numpy(),
		"end_timestamp": test_metadata[
			"end_timestamp"
		].to_numpy(),
		"anomaly_score": scores,
	})

	output = scores_df.merge(
		test_labels[
			[
				"sequence_id",
				"proxy_anomaly",
			]
		],
		on="sequence_id",
		how="left",
		validate="one_to_one",
	)

	if output["proxy_anomaly"].isna().any():
		raise ValueError(
			"Some test sequences have no matching proxy label."
		)

	output["proxy_anomaly"] = (
		output["proxy_anomaly"].astype(int)
	)

	output["prediction"] = (
		output["anomaly_score"] >= threshold
	).astype(int)

	return output.sort_values(
		by=[
			"start_timestamp",
			"end_timestamp",
			"sequence_id",
		]
	).reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def main():

	print("=" * 70)
	print("LSTM FIXED-THRESHOLD TEST SCORE GENERATION")
	print("=" * 70)

	(
		scores,
		test_labels,
		test_metadata,
		threshold,
	) = load_data()

	print("\nLoaded:")
	print(f"Test scores:    {scores.shape}")
	print(f"Proxy labels:   {test_labels.shape}")
	print(f"Metadata:       {test_metadata.shape}")
	print(f"Threshold:      {threshold:.12f}")

	validate_inputs(
		scores,
		test_labels,
		test_metadata,
	)

	output = build_output(
		scores,
		threshold,
		test_labels,
		test_metadata,
	)

	required_output_columns = [
		"sequence_id",
		"segment_id",
		"start_timestamp",
		"end_timestamp",
		"anomaly_score",
		"proxy_anomaly",
		"prediction",
	]

	if list(output.columns) != required_output_columns:
		raise ValueError(
			"Unexpected output columns.\n"
			f"Expected: {required_output_columns}\n"
			f"Received: {list(output.columns)}"
		)

	if output["sequence_id"].duplicated().any():
		raise ValueError(
			"Duplicate sequence_id values found in final output."
		)

	output.to_csv(
		OUTPUT_PATH,
		index=False,
	)

	print("\n" + "=" * 70)
	print("TEST SCORE SUMMARY")
	print("=" * 70)
	print(f"Sequences:          {len(output):,}")
	print(f"Anomaly score mean: {output['anomaly_score'].mean():.6f}")
	print(f"Anomaly score std:  {output['anomaly_score'].std():.6f}")
	print(f"Anomaly score min:  {output['anomaly_score'].min():.6f}")
	print(f"Anomaly score max:  {output['anomaly_score'].max():.6f}")
	print(f"Proxy normal:       {(output['proxy_anomaly'] == 0).sum():,}")
	print(f"Proxy anomalous:    {(output['proxy_anomaly'] == 1).sum():,}")
	print(f"Predicted normal:   {(output['prediction'] == 0).sum():,}")
	print(f"Predicted anomalous:{(output['prediction'] == 1).sum():,}")
	print("\nOutput:")
	print(OUTPUT_PATH)
	print("\nFirst five rows:")
	print(output.head().to_string(index=False))
	print("\n" + "=" * 70)
	print("COMPLETE")
	print("=" * 70)


if __name__ == "__main__":
	main()
