import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class LSTMEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_units: int = 64,
        latent_dim: int = 16,
        dropout: float = 0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.dropout_rate = dropout

        self.lstm = tf.keras.layers.LSTM(
            hidden_units,
            return_sequences=False,
        )

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.latent = tf.keras.layers.Dense(latent_dim)

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dropout(x, training=training)

        return self.latent(x)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "hidden_units": self.hidden_units,
                "latent_dim": self.latent_dim,
                "dropout": self.dropout_rate,
            }
        )

        return config