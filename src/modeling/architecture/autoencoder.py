import tensorflow as tf

from .encoder import LSTMEncoder
from .decoder import LSTMDecoder


@tf.keras.utils.register_keras_serializable()
class LSTMAutoencoder(tf.keras.Model):
    def __init__(
        self,
        hidden_units: int = 64,
        latent_dim: int = 16,
        dropout: float = 0.10,
        sequence_length: int = 60,
        feature_count: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.dropout_rate = dropout
        self.sequence_length = sequence_length
        self.feature_count = feature_count

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

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "hidden_units": self.hidden_units,
                "latent_dim": self.latent_dim,
                "dropout": self.dropout_rate,
                "sequence_length": self.sequence_length,
                "feature_count": self.feature_count,
            }
        )

        return config