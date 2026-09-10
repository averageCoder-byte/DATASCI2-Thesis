from pathlib import Path
import tempfile
import zipfile

import numpy as np
import tensorflow as tf

from architecture.autoencoder import LSTMAutoencoder


MODEL_DIR = Path(__file__).resolve().parent
SEQUENCE_DIR = MODEL_DIR / "data" / "sequences"
MODEL_PATH = MODEL_DIR / "data" / "models" / "lstm_autoencoder_exp04.keras"
OUTPUT_DIR = MODEL_DIR / "data" / "test_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading test sequences...")
X_test = np.load(SEQUENCE_DIR / "X_test.npy")

print(f"Test shape: {X_test.shape}")

print("\nLoading finalized LSTM Autoencoder...")
# Temporary recovery for incorrect serialization
# Uses manually encoded configs from Exp4
model = LSTMAutoencoder(
    hidden_units=128,
    latent_dim=32,
    dropout=0.10,
    sequence_length=60,
    feature_count=7,
)

# Actual model loader for Airflow
# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     custom_objects={
#         "LSTMAutoencoder": LSTMAutoencoder,
#         "LSTMEncoder": LSTMEncoder,
#         "LSTMDecoder": LSTMDecoder,
#     },
#     compile=False,
# )

model.build(input_shape=(None, 60, 7))

with tempfile.TemporaryDirectory() as temp_dir:
    with zipfile.ZipFile(MODEL_PATH, "r") as archive:
        archive.extract("model.weights.h5", temp_dir)

    weights_path = Path(temp_dir) / "model.weights.h5"

    model.load_weights(weights_path)

print("Model loaded successfully.")

print("\nGenerating reconstructions...")
X_reconstructed = model.predict(
    X_test,
    batch_size=64,
    verbose=1,
)

print(f"Reconstructed shape: {X_reconstructed.shape}")

print("\nCalculating reconstruction errors...")

# MAE for each individual 60-candle sequence
reconstruction_errors = np.mean(
    np.abs(X_test - X_reconstructed),
    axis=(1, 2),
)

print("\nTest reconstruction error statistics:")
print(f"Mean:   {reconstruction_errors.mean():.6f}")
print(f"Std:    {reconstruction_errors.std():.6f}")
print(f"Min:    {reconstruction_errors.min():.6f}")
print(f"Median: {np.median(reconstruction_errors):.6f}")
print(f"Max:    {reconstruction_errors.max():.6f}")

OUTPUT_PATH = OUTPUT_DIR / "test_reconstruction_errors.npy"

np.save(
    OUTPUT_PATH,
    reconstruction_errors,
)

print("\nTesting complete.")
print(f"Saved reconstruction errors to: {OUTPUT_PATH}")