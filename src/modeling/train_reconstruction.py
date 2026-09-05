from pathlib import Path
import tempfile
import zipfile

import numpy as np

from architecture.autoencoder import LSTMAutoencoder


MODEL_DIR = Path(__file__).resolve().parent

SEQUENCE_DIR = MODEL_DIR / "data" / "sequences"
MODEL_PATH = MODEL_DIR / "data" / "models" / "lstm_autoencoder_exp04.keras"
OUTPUT_DIR = MODEL_DIR / "data" / "train_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load training sequences
print("Loading training sequences...")
X_train = np.load(SEQUENCE_DIR / "X_train.npy")

print(f"Training shape: {X_train.shape}")


# Load finalized Exp4 LSTM Autoencoder
print("\nLoading finalized LSTM Autoencoder...")

model = LSTMAutoencoder(
    hidden_units=128,
    latent_dim=32,
    dropout=0.10,
    sequence_length=60,
    feature_count=7,
)

model.build(input_shape=(None, 60, 7))

with tempfile.TemporaryDirectory() as temp_dir:
    with zipfile.ZipFile(MODEL_PATH, "r") as archive:
        archive.extract("model.weights.h5", temp_dir)

    weights_path = Path(temp_dir) / "model.weights.h5"

    model.load_weights(weights_path)

print("Model loaded successfully.")


# Generate reconstructions
print("\nGenerating training reconstructions...")

X_reconstructed = model.predict(
    X_train,
    batch_size=64,
    verbose=1,
)

print(f"Reconstructed shape: {X_reconstructed.shape}")


# Calculate reconstruction error
print("\nCalculating reconstruction errors...")

reconstruction_errors = np.mean(
    np.abs(X_train - X_reconstructed),
    axis=(1, 2),
)


# Display statistics
print("\nTraining reconstruction error statistics:")
print(f"Mean:   {reconstruction_errors.mean():.6f}")
print(f"Std:    {reconstruction_errors.std():.6f}")
print(f"Min:    {reconstruction_errors.min():.6f}")
print(f"Median: {np.median(reconstruction_errors):.6f}")
print(f"Max:    {reconstruction_errors.max():.6f}")


# Save reconstruction errors
OUTPUT_PATH = OUTPUT_DIR / "train_reconstruction_errors.npy"

np.save(
    OUTPUT_PATH,
    reconstruction_errors,
)

print("\nTraining reconstruction complete.")
print(f"Saved reconstruction errors to: {OUTPUT_PATH}")