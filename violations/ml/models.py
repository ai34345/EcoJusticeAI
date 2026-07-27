import tensorflow as tf
from violations.ml.config import CONFIG
def build_model(input_shape=(60, 19)):
    """Build Bi-LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.BatchNormalization(),
# Layer 1
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(CONFIG.lstm_hidden_units, return_sequences=True)
),
        tf.keras.layers.Dropout(CONFIG.dropout_rate),
# Layer 2
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
),
        tf.keras.layers.Dropout(CONFIG.dropout_rate),
# Dense
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
)
    return model
if __name__ == '__main__':
    model = build_model()
    model.summary()