import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
RESULTS_DIR = ROOT / "results" / "model"
LOG_DIR = RESULTS_DIR / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === 1. Загрузка данных ===
data = pd.read_csv(DATA_PATH)

X = np.stack([np.fromstring(str(pix), sep=" ") for pix in data["pixels"]])
assert X.shape[1] == 48 * 48, f"Ожидалось 2304 пикселя, получили {X.shape[1]}"
X = X.reshape(-1, 48, 48, 1).astype("float32") / 255.0

y_labels = data["emotion"].astype(int).to_numpy()
num_classes = len(np.unique(y_labels))
y = to_categorical(y_labels, num_classes=num_classes)

print("Классы:", sorted(np.unique(y_labels)))

# === 2. Train/Val split ===
X_train, X_val, y_train, y_val, y_train_labels, y_val_labels = train_test_split(
    X, y, y_labels, test_size=0.1, stratify=y_labels, random_state=42
)

print("Train:", X_train.shape, " Val:", X_val.shape)

# === 3. Class weights ===
cw_values = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train_labels
)
class_weight = {i: float(cw_values[i]) for i in range(num_classes)}
print("class_weight:", class_weight)

# === 4. tf.data + аугментации ===
def aug_fn(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.resize(img, [52, 52])
    img = tf.image.random_crop(img, [48, 48, 1])
    return img, label

batch_size = 64

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(16384)
            .map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .batch(256)
          .prefetch(tf.data.AUTOTUNE))

# === 5. Модель (усиленная CNN) ===
wd = 1e-4

def conv_bn_relu(x, f):
    x = layers.Conv2D(
        f, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(wd)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

inputs = layers.Input((48, 48, 1))

x = conv_bn_relu(inputs, 64)
x = layers.MaxPooling2D(2)(x)

x = conv_bn_relu(x, 128)
x = layers.MaxPooling2D(2)(x)

x = conv_bn_relu(x, 256)
x = conv_bn_relu(x, 256)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(
    256, activation="relu",
    kernel_regularizer=regularizers.l2(wd)
)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

loss = CategoricalCrossentropy(label_smoothing=0.05)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=loss,
    metrics=["accuracy"],
)

model.summary()

# === 6. Колбэки ===
ckpt_path = RESULTS_DIR / "final_emotion_model.keras"
tensorboard_cb = TensorBoard(log_dir=str(LOG_DIR), histogram_freq=1)

early_stop_cb = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True,
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1,
)

ckpt_cb = ModelCheckpoint(
    filepath=str(ckpt_path),
    save_best_only=True,
    monitor="val_accuracy",
)

# === 7. Обучение ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=80,
    class_weight=class_weight,
    callbacks=[early_stop_cb, reduce_lr_cb, tensorboard_cb, ckpt_cb],
)

# === 8. Графики ===
plt.figure()
plt.plot(history.history["accuracy"], label="Train acc")
plt.plot(history.history["val_accuracy"], label="Val acc")
plt.legend()
plt.title("Accuracy")
plt.savefig(RESULTS_DIR / "learning_curves.png", bbox_inches="tight")

# === 9. Архитектура ===
with open(RESULTS_DIR / "final_emotion_model_arch.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("\n✅ Готово! Модель сохранена в:", ckpt_path)
