from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "modeling" / "data"
BASELINE_DIR = BASE_DIR / "baseline" / "data" / "statistical"
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

TEST_SEQUENCES = (
	MODEL_DIR
	/ "sequences"
	/ "X_test.npy"
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

TRAINING_DISTRIBUTION_PATH = (
	BASELINE_DIR
	/ "zscore_training_distribution.csv"
)

THRESHOLD_PATH = (
	BASELINE_DIR
	/ "zscore_threshold.txt"
)


# ------------------------------------------------------------
# Output file
# ------------------------------------------------------------

OUTPUT_PATH = (
	STATISTICAL_COMPARISON_DIR
	/ "zscore_test_scores.csv"
)


# ============================================================
# Load data
# ============================================================

def load_data():
	"""Load frozen Z-score artifacts and test sequence metadata."""

	input_paths = {
		"test sequences": TEST_SEQUENCES,
		"test proxy labels": TEST_LABELS,
		"test sequence metadata": TEST_METADATA,
		"Z-score training distribution": TRAINING_DISTRIBUTION_PATH,
		"Z-score threshold": THRESHOLD_PATH,
	}

	for description, path in input_paths.items():
		if not path.exists():
			raise FileNotFoundError(
				f"Missing {description}:\n{path}"
			)

	X_test = np.load(TEST_SEQUENCES)
	test_labels = pd.read_csv(TEST_LABELS)
	test_metadata = pd.read_parquet(TEST_METADATA)
	distribution = pd.read_csv(TRAINING_DISTRIBUTION_PATH)
	threshold = float(
		THRESHOLD_PATH.read_text(
			encoding="utf-8"
		).strip()
	)

	required_distribution_columns = {
		"feature_index",
		"mean",
		"std",
	}

	missing_distribution_columns = (
		required_distribution_columns
		- set(distribution.columns)
	)

	if missing_distribution_columns:
		raise ValueError(
			"Z-score training distribution is missing required "
			f"columns: {sorted(missing_distribution_columns)}"
		)

	means = distribution.sort_values(
		"feature_index"
	)["mean"].to_numpy()

	stds = distribution.sort_values(
		"feature_index"
	)["std"].to_numpy()

	return (
		X_test,
		test_labels,
		test_metadata,
		means,
		stds,
		threshold,
	)


# ============================================================
# Anomaly scores
# ============================================================

def anomaly_scores(X, means, stds):
	"""Return the maximum absolute standardized deviation per sequence."""

	if X.ndim != 3:
		raise ValueError(
			"Expected test sequences with shape "
			"(samples, timesteps, features). "
			f"Received shape: {X.shape}"
		)

	if X.shape[-1] != len(means) or len(means) != len(stds):
		raise ValueError(
			"Z-score distribution feature count does not match "
			f"test sequences. Features: {X.shape[-1]}; "
			f"means: {len(means)}; stds: {len(stds)}"
		)

	return np.max(
		np.abs((X - means) / stds),
		axis=(1, 2),
	)


# ============================================================
# Validate inputs
# ============================================================

def validate_inputs(
	X_test,
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

	n_sequences = len(X_test)

	if len(test_labels) != n_sequences:
		raise ValueError(
			"Test sequence count does not match proxy labels. "
			f"Sequences: {n_sequences:,}; "
			f"Labels: {len(test_labels):,}"
		)

	if len(test_metadata) != n_sequences:
		raise ValueError(
			"Test sequence count does not match sequence metadata. "
			f"Sequences: {n_sequences:,}; "
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
	print("Z-SCORE TEST SCORE GENERATION")
	print("=" * 70)

	(
		X_test,
		test_labels,
		test_metadata,
		means,
		stds,
		threshold,
	) = load_data()

	print("\nLoaded:")
	print(f"Test sequences: {X_test.shape}")
	print(f"Proxy labels:   {test_labels.shape}")
	print(f"Metadata:       {test_metadata.shape}")
	print(f"Distribution:   {TRAINING_DISTRIBUTION_PATH}")
	print(f"Threshold:      {threshold:.12f}")

	validate_inputs(
		X_test,
		test_labels,
		test_metadata,
	)

	print("\nGenerating Z-score test anomaly scores...")

	scores = anomaly_scores(
		X_test,
		means,
		stds,
	)

	if len(scores) != len(X_test):
		raise ValueError(
			"Number of generated anomaly scores does not "
			"match number of test sequences."
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

	if len(output) != len(X_test):
		raise ValueError(
			"Final output row count does not match "
			"test sequence count."
		)

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
