from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent.parent

GMM_DIR = MODEL_DIR / "data" / "gmm_results"
TEST_DIR = MODEL_DIR / "data" / "test_results"

GMM_MODEL_PATH = GMM_DIR / "final_gmm.joblib"
TEST_ERRORS_PATH = TEST_DIR / "test_reconstruction_errors.npy"

OUTPUT_DIR = GMM_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading finalized GMM...")
gmm = joblib.load(GMM_MODEL_PATH)

print(f"GMM components: {gmm.n_components}")
print(f"Covariance type: {gmm.covariance_type}")


print("\nLoading test reconstruction errors...")
test_errors = np.load(TEST_ERRORS_PATH)

print(f"Test errors shape: {test_errors.shape}")

# GaussianMixture expects:
# (samples, features)
X_test_errors = test_errors.reshape(-1, 1)

print(f"GMM input shape: {X_test_errors.shape}")


print("\nCalculating test log-likelihoods...")

# score_samples() returns the log probability density
# of each reconstruction error under the fitted GMM.
log_likelihoods = gmm.score_samples(X_test_errors)


# Higher anomaly score = more unusual reconstruction error.
anomaly_scores = -log_likelihoods


print("\nTest anomaly-score statistics:")
print(f"Mean:   {anomaly_scores.mean():.6f}")
print(f"Std:    {anomaly_scores.std():.6f}")
print(f"Min:    {anomaly_scores.min():.6f}")
print(f"Median: {np.median(anomaly_scores):.6f}")
print(f"Max:    {anomaly_scores.max():.6f}")


# Save sequence-level scoring results.
results_df = pd.DataFrame(
    {
        "sequence_id": np.arange(len(test_errors)),
        "reconstruction_error": test_errors,
        "log_likelihood": log_likelihoods,
        "anomaly_score": anomaly_scores,
    }
)


OUTPUT_PATH = OUTPUT_DIR / "test_anomaly_scores.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\nScoring complete.")
print(f"Saved results to:")
print(OUTPUT_PATH)