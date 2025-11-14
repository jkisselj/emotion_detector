from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "model" / "final_emotion_model.keras"
TEST_PATH = ROOT / "data" / "test.csv"

if not MODEL_PATH.exists():
    raise SystemExit(f"Модель не найдена: {MODEL_PATH}")
if not TEST_PATH.exists():
    raise SystemExit(f"Тестовый файл не найден: {TEST_PATH}")

data = pd.read_csv(TEST_PATH)

X = np.stack([np.fromstring(str(pix), sep=" ") for pix in data["pixels"]])
if X.shape[1] != 48 * 48:
    raise SystemExit(f"Ожидалось 2304 пикселя (48x48), получили {X.shape[1]}")
X = X.reshape(-1, 48, 48, 1).astype("float32") / 255.0
y_true = data["emotion"].astype(int).to_numpy()

model = tf.keras.models.load_model(MODEL_PATH)

# TTA: оригинал + горизонтальный флип
X_flip = X[:, :, ::-1, :]

probs1 = model.predict(X, batch_size=256, verbose=0)
probs2 = model.predict(X_flip, batch_size=256, verbose=0)
probs = (probs1 + probs2) / 2.0

y_pred = probs.argmax(axis=1)
acc = (y_pred == y_true).mean()

print(f"Accuracy on test set: {acc*100:.2f}%")
