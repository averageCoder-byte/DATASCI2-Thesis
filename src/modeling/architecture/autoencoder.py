import tensorflow as tf

from .encoder import LSTMEncoder
from .decoder import LSTMDecoder


class LSTMAutoencoder(tf.keras.Model):
    def __init__(
        self,
        hidden_units: int = 64,
        latent_dim: int = 16,
        dropout: float = 0.10,
        sequence_length: int = 60,
        feature_count: int = 7,
    ):
        super().__init__()

        self.encoder = LSTMEncoder(
            hidden_units=hidden_units,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        self.decoder = LSTMDecoder(
            hidden_units=hidden_units,
            sequence_length=sequence_length,
            feature_count=feature_count,
            dropout=dropout,
        )

    def call(self, inputs, training=False):
        latent = self.encoder(
            inputs,
            training=training,
        )

        reconstruction = self.decoder(
            latent,
            training=training,
        )

        return reconstruction