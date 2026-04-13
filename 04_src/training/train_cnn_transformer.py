from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

data_dir = Path(r"C:\SAYSAI\03_processed_data")
output_dir = Path(r"C:\SAYSAI\06_results")
output_dir.mkdir(parents=True, exist_ok=True)

train_X = np.load(data_dir / "train_X.npy")
train_y = np.load(data_dir / "train_y.npy")
val_X = np.load(data_dir / "val_X.npy")
val_y = np.load(data_dir / "val_y.npy")
test_X = np.load(data_dir / "test_X.npy")
test_y = np.load(data_dir / "test_y.npy")

with open(data_dir / "label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

num_classes = len(label_map["label_to_index"])
input_shape = train_X.shape[1:]  # (30, 126)

print("train_X:", train_X.shape)
print("train_y:", train_y.shape)
print("val_X:", val_X.shape)
print("val_y:", val_y.shape)
print("test_X:", test_X.shape)
print("test_y:", test_y.shape)
print("input_shape:", input_shape)
print("num_classes:", num_classes)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

inputs = layers.Input(shape=input_shape)

x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
x = layers.BatchNormalization()(x)

x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)

x = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.1)(x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

cb = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        filepath=str(output_dir / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    )
]

history = model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    epochs=50,
    batch_size=8,
    callbacks=cb,
    verbose=1
)

test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

model.save(output_dir / "final_model.keras")

with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
    json.dump(history.history, f, indent=2)

print("\nSaved files:")
print(output_dir / "best_model.keras")
print(output_dir / "final_model.keras")
print(output_dir / "training_history.json")