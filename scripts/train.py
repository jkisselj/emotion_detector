import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results' / 'model'
LOG_DIR = RESULTS_DIR / 'logs'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === 1. Загрузка данных ===
data_path = ROOT / 'data' / 'train.csv'
data = pd.read_csv(data_path)
print("Пример данных:")
print(data.head())

# Преобразуем пиксели в numpy массив
X = np.array([np.fromstring(pix, sep=' ') for pix in data['pixels']])
if X.shape[1] != 48*48:
    raise ValueError(f"Ожидалось 2304 пикселя, а получили {X.shape[1]}. Проверь data/train.csv")
X = X.reshape(-1, 48, 48, 1).astype('float32') / 255.0

num_classes = 7
y_labels = data['emotion'].astype(int)

# === 2. Train/Val split (стратификация по меткам) ===
X_train, X_val, y_train_labels, y_val_labels = train_test_split(
    X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# One-hot после сплита
y_train = tf.keras.utils.to_categorical(y_train_labels, num_classes=num_classes)
y_val   = tf.keras.utils.to_categorical(y_val_labels,   num_classes=num_classes)

# === 3. Модель CNN ===
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === 4. Колбэки ===
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR))

# === 5. Обучение ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop, tensorboard]
)

# === 6. Сохранение ===
model.save(RESULTS_DIR / 'final_emotion_model.keras')

# График обучения (accuracy)
plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Learning curves (accuracy)')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()
plt.savefig(RESULTS_DIR / 'learning_curves.png', bbox_inches='tight')

with open(RESULTS_DIR / 'final_emotion_model_arch.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("\nГотово! Модель и артефакты сохранены в:", RESULTS_DIR)
print("Для TensorBoard: tensorboard --logdir", LOG_DIR)
