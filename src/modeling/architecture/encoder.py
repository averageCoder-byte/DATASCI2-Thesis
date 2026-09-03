import tensorflow as tf

class LSTMEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_units: int = 64,
        latent_dim: int = 16,
        dropout: float = 0.10,
    ):
        super().__init__()

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