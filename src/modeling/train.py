from pathlib import Path

import numpy as np
import tensorflow as tf

from architecture.autoencoder import LSTMAutoencoder

import time

# Paths

MODEL_DIR = Path(__file__).resolve().parent

SEQUENCE_DIR = MODEL_DIR / "data" / "sequences"
MODEL_OUTPUT_DIR = MODEL_DIR / "data" / "models"

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# Configuration

SEQUENCE_LENGTH = 60
FEATURE_COUNT = 7

HIDDEN_UNITS = 128
LATENT_DIM = 32
DROPOUT = 0.10

BATCH_SIZE = 128
LEARNING_RATE = 0.001

MAX_EPOCHS = 100
PATIENCE = 10

# Load sequences
print("Loading sequences...")

X_train = np.load(SEQUENCE_DIR / "X_train.npy")
X_validation = np.load(SEQUENCE_DIR / "X_validation.npy")

print(f"Training shape:   {X_train.shape}")
print(f"Validation shape: {X_validation.shape}")


# Build model

print("\nBuilding LSTM Autoencoder...")

model = LSTMAutoencoder(
    hidden_units=HIDDEN_UNITS,
    latent_dim=LATENT_DIM,
    dropout=DROPOUT,
    sequence_length=SEQUENCE_LENGTH,
    feature_count=FEATURE_COUNT,
)


# Compile

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.MeanAbsoluteError(),
)


# Build model with input shape

model.build(input_shape=(None, SEQUENCE_LENGTH, FEATURE_COUNT))

model.summary()


# Early stopping

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1,
)


# Train

print("\nStarting training...")
start_time = time.perf_counter()

history = model.fit(
    X_train,
    X_train,
    validation_data=(X_validation, X_validation),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    shuffle=True,
    verbose=1,
)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Save model
MODEL_PATH = MODEL_OUTPUT_DIR / "lstm_autoencoder_exp05.keras"

model.save(MODEL_PATH)

print("\nTraining complete.")
print(f"\nTraining time: {elapsed_time:.2f} seconds")
print(f"Training time: {elapsed_time / 60:.2f} minutes")
print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
print(f"Model saved to: {MODEL_PATH}")