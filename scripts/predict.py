import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'results' / 'model' / 'final_emotion_model.keras'
TEST_PATH  = ROOT / 'data' / 'test.csv'

def fail(msg):
    print(msg)
    raise SystemExit(1)

# 1) Проверки
if not MODEL_PATH.exists():
    fail(f"Модель не найдена: {MODEL_PATH}\nСначала обучи: python scripts/train.py")

if not TEST_PATH.exists():
    fail(f"Тестовый файл не найден: {TEST_PATH}\nСоздай его или положи сюда CSV формата FER (emotion,pixels).")

# 2) Загрузка данных
df = pd.read_csv(TEST_PATH)
if 'emotion' not in df.columns or 'pixels' not in df.columns:
    fail("Ожидались колонки 'emotion' и 'pixels' в test.csv")

try:
    X = np.stack([np.fromstring(str(p), sep=' ') for p in df['pixels']])
except Exception as e:
    fail(f"Не удалось преобразовать пиксели из test.csv: {e}")

if X.shape[1] != 48*48:
    fail(f"Ожидалось 2304 пикселя на изображение (48*48), получили {X.shape[1]}.")

X = X.reshape(-1, 48, 48, 1).astype('float32') / 255.0
y_true = df['emotion'].astype(int).to_numpy()

# 3) Загрузка модели
model = tf.keras.models.load_model(MODEL_PATH)

# 4) Предсказание и accuracy
probs = model.predict(X, batch_size=256, verbose=0)
y_pred = probs.argmax(axis=1)
acc = (y_pred == y_true).mean()

# 5) Вывод в нужном формате
print(f"Accuracy on test set: {round(acc * 100)}%")
