from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


MODEL_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = MODEL_DIR / "data" / "validation_results"
OUTPUT_DIR = MODEL_DIR / "data" / "gmm_results"

VALIDATION_ERRORS_PATH = (
    VALIDATION_DIR / "validation_reconstruction_errors.npy"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Candidate number of Gaussian components.
COMPONENT_RANGE = range(2, 6)


print("Loading validation reconstruction errors...")

validation_errors = np.load(VALIDATION_ERRORS_PATH)

print(f"Validation errors shape: {validation_errors.shape}")

# GaussianMixture expects a 2D array:
# (samples, features)
X_validation = validation_errors.reshape(-1, 1)

print(f"GMM input shape: {X_validation.shape}")


print("\nFitting candidate GMM configurations...")

results = []
models = {}

for n_components in COMPONENT_RANGE:

    print(f"\nFitting GMM with K={n_components}...")

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
        n_init=10,
    )

    gmm.fit(X_validation)

    bic = gmm.bic(X_validation)
    aic = gmm.aic(X_validation)
    log_likelihood = gmm.score(X_validation)

    results.append(
        {
            "n_components": n_components,
            "bic": bic,
            "aic": aic,
            "average_log_likelihood": log_likelihood,
            "converged": gmm.converged_,
            "iterations": gmm.n_iter_,
        }
    )

    models[n_components] = gmm

    print(f"BIC:                   {bic:.6f}")
    print(f"AIC:                   {aic:.6f}")
    print(f"Average log-likelihood: {log_likelihood:.6f}")
    print(f"Converged:             {gmm.converged_}")
    print(f"Iterations:            {gmm.n_iter_}")


# Convert experiment results into a DataFrame.
results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("GMM MODEL SELECTION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)


# Select the configuration with the lowest BIC.
best_row = results_df.loc[results_df["bic"].idxmin()]
best_components = int(best_row["n_components"])

best_gmm = models[best_components]

MODEL_PATH = OUTPUT_DIR / "final_gmm.joblib"

joblib.dump(best_gmm, MODEL_PATH)

print("\nFinal GMM model saved to:")
print(MODEL_PATH)


print("\n" + "=" * 70)
print("SELECTED GMM")
print("=" * 70)

print(f"Selected components (K): {best_components}")
print(f"BIC:                     {best_gmm.bic(X_validation):.6f}")
print(f"AIC:                     {best_gmm.aic(X_validation):.6f}")
print(f"Converged:               {best_gmm.converged_}")
print(f"Iterations:              {best_gmm.n_iter_}")


print("\nGMM component parameters:")

for i in range(best_components):

    mean = best_gmm.means_[i, 0]
    variance = best_gmm.covariances_[i, 0, 0]
    std = np.sqrt(variance)
    weight = best_gmm.weights_[i]

    print(f"\nComponent {i + 1}:")
    print(f"  Weight:    {weight:.6f}")
    print(f"  Mean:      {mean:.6f}")
    print(f"  Variance:  {variance:.6f}")
    print(f"  Std:       {std:.6f}")


# Save model-selection results.
RESULTS_PATH = OUTPUT_DIR / "gmm_model_selection.csv"

results_df.to_csv(
    RESULTS_PATH,
    index=False,
)


print("\nGMM model-selection results saved to:")
print(RESULTS_PATH)