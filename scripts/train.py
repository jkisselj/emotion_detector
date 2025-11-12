import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results' / 'model'
LOG_DIR = RESULTS_DIR / 'logs'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === 1) Загрузка данных ===
data_path = ROOT / 'data' / 'train.csv'
data = pd.read_csv(data_path)
X = np.stack([np.fromstring(p, sep=' ') for p in data['pixels']])
assert X.shape[1] == 48*48, f"Ожидалось 2304 пикселя, получили {X.shape[1]}"
X = X.reshape(-1, 48, 48, 1).astype('float32') / 255.0
y_labels = data['emotion'].astype(int).to_numpy()
num_classes = len(np.unique(y_labels))
print("Классы:", sorted(np.unique(y_labels)))

# === 2) Train/Val split (стратификация) ===
X_train, X_val, y_train_labels, y_val_labels = train_test_split(
    X, y_labels, test_size=0.1, random_state=42, stratify=y_labels
)
y_train = tf.keras.utils.to_categorical(y_train_labels, num_classes=num_classes)
y_val   = tf.keras.utils.to_categorical(y_val_labels,   num_classes=num_classes)

# === 3) Class weights ===
cw_values = compute_class_weight(class_weight='balanced',
                                 classes=np.arange(num_classes),
                                 y=y_train_labels)
class_weight = {i: float(cw_values[i]) for i in range(num_classes)}
print("class_weight:", class_weight)

# === 4) Аугментации через tf.data ===
def aug_fn(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.resize(img, [54,54])
    img = tf.image.random_crop(img, [48,48,1])
    return img, label

batch_size = 64
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(8192)
            .map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(256)
            .prefetch(tf.data.AUTOTUNE))

# === 5) Модель ===
def make_model():
    inputs = layers.Input((48,48,1))
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

model = make_model()
loss = CategoricalCrossentropy(label_smoothing=0.05)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=loss,
              metrics=['accuracy'])
model.summary()

# === 6) Колбэки ===
ckpt_path = RESULTS_DIR / 'final_emotion_model.keras'
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    TensorBoard(log_dir=str(LOG_DIR)),
    ModelCheckpoint(str(ckpt_path), monitor='val_accuracy', save_best_only=True)
]

# === 7) Обучение ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=60,
    class_weight=class_weight,
    callbacks=callbacks
)

# === 8) Сохранение графика и архитектуры ===
import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Learning curves (accuracy)')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()
plt.savefig(RESULTS_DIR / 'learning_curves.png', bbox_inches='tight')

with open(RESULTS_DIR / 'final_emotion_model_arch.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("\nГотово! Лучшая модель сохранена в:", ckpt_path)
print("TensorBoard:", LOG_DIR)
