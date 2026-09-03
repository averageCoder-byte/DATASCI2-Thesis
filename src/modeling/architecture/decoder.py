import tensorflow as tf

class LSTMDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_units: int = 64,
        sequence_length: int = 60,
        feature_count: int = 7,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.feature_count = feature_count

        self.repeat = tf.keras.layers.RepeatVector(sequence_length)

        self.lstm = tf.keras.layers.LSTM(
            hidden_units,
            return_sequences=True,
        )

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(feature_count)
        )

    def call(self, latent, training=False):
        x = self.repeat(latent)
        x = self.lstm(x)

        x = self.dropout(x, training=training)

        return self.output_layer(x)