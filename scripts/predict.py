from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "model" / "final_emotion_model.keras"
TEST_PATH = ROOT / "data" / "test.csv"

if not MODEL_PATH.exists():
    raise SystemExit(f"Модель не найдена: {MODEL_PATH}")
if not TEST_PATH.exists():
    raise SystemExit(f"Тестовый файл не найден: {TEST_PATH}")

data = pd.read_csv(TEST_PATH)
if "emotion" not in data.columns or "pixels" not in data.columns:
    raise SystemExit("В test.csv должны быть колонки 'emotion' и 'pixels'")

X = np.stack([np.fromstring(str(pix), sep=" ") for pix in data["pixels"]])
if X.shape[1] != 48 * 48:
    raise SystemExit(f"Ожидалось 2304 пикселя (48x48), получили {X.shape[1]}")

X = X.reshape(-1, 48, 48, 1).astype("float32") / 255.0
y_true = data["emotion"].astype(int).to_numpy()

model = tf.keras.models.load_model(MODEL_PATH)

def make_tta_batch(X):
    """Генерим 6 версий каждого изображения:
       original, flip, dark, bright, center-crop, center-crop+flip.
    """
    imgs = []
    for img in X:
        # 0: оригинал
        base = img[:, :, 0]  # (48,48)
        # 1: flip
        flip = base[:, ::-1]
        # 2: темнее
        dark = np.clip(base * 0.9, 0.0, 1.0)
        # 3: светлее
        bright = np.clip(base * 1.1, 0.0, 1.0)
        # 4: небольшой центр-кроп 44x44 -> resize до 48x48
        crop = base[2:46, 2:46]
        crop_resized = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_LINEAR)
        # 5: crop + flip
        crop_flip = crop_resized[:, ::-1]

        variants = [base, flip, dark, bright, crop_resized, crop_flip]
        for v in variants:
            imgs.append(v[..., None])  # (48,48,1)
    return np.stack(imgs, axis=0)

X_tta = make_tta_batch(X)  # shape: (N*6, 48,48,1)

probs_all = model.predict(X_tta, batch_size=256, verbose=0)
num_classes = probs_all.shape[1]
num_samples = X.shape[0]

probs_all = probs_all.reshape(num_samples, 6, num_classes)
probs = probs_all.mean(axis=1)  # усредняем по 6 вариантам

y_pred = probs.argmax(axis=1)
acc = (y_pred == y_true).mean()
print(f"Accuracy on test set: {acc*100:.2f}%")
