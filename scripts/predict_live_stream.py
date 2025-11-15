import time
import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Пути
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "model" / "final_emotion_model.keras"
CASCADE_PATH = ROOT / "data" / "haarcascade_frontalface_default.xml"
OUT_DIR = ROOT / "results" / "preprocessing_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise SystemExit(f"Не найдена модель: {MODEL_PATH}\nСначала запусти: python scripts/train.py")
if not CASCADE_PATH.exists():
    raise SystemExit(f"Не найден каскад Хаара: {CASCADE_PATH}\nСкачай его в папку data/.")

# Метки эмоций (порядок как в FER2013: 0..6)
EMOTION_LABELS = [
    "Angry",    # 0
    "Disgust",  # 1
    "Fear",     # 2
    "Happy",    # 3
    "Sad",      # 4
    "Surprise", # 5
    "Neutral"   # 6
]

print("Загружаю модель...")
model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

# Настраиваем видеозахват с вебкамеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Не удалось открыть вебкамеру (VideoCapture(0)).")

# Видеозапись входного потока
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_path = OUT_DIR / "input_video.mp4"
fps = 20.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
writer = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))

print("Reading video stream ...")
start_time = time.time()
last_second_printed = -1
image_idx = 0
duration_sec = 20  # минимум 20 секунд по условию

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось прочитать кадр с камеры.")
            break

        # Сохраняем исходный кадр в видео
        writer.write(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(40, 40)
        )

        elapsed = time.time() - start_time
        elapsed_sec = int(elapsed)

        # Обрабатываем только первое найденное лицо (или самое крупное)
        if len(faces) > 0:
            # Выбираем самое большое по площади
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))

            # Сохраняем preprocessed изображение
            img_path = OUT_DIR / f"image{image_idx}.png"
            cv2.imwrite(str(img_path), roi)
            image_idx += 1

            # Готовим для модели
            roi_norm = roi.astype("float32") / 255.0
            roi_norm = np.expand_dims(roi_norm, axis=-1)  # (48,48,1)
            roi_norm = np.expand_dims(roi_norm, axis=0)   # (1,48,48,1)

            # Предсказание
            preds = model.predict(roi_norm, verbose=0)[0]
            emotion_idx = int(np.argmax(preds))
            prob = float(np.max(preds))
            emotion = EMOTION_LABELS[emotion_idx]

            # Печатаем примерно раз в секунду
            if elapsed_sec != last_second_printed:
                last_second_printed = elapsed_sec
                # Форматируем "hh:mm:ss"
                ts = str(datetime.timedelta(seconds=elapsed_sec))
                print("Preprocessing ...")
                print(f"{ts}s : {emotion} , {prob*100:.0f}%")

        # Можно раскомментировать, если есть GUI и хочется видеть видео:
        # cv2.imshow("Emotion live", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("Остановлено пользователем (нажата 'q').")
        #     break

        if elapsed >= duration_sec:
            print(f"Достигнута длительность {duration_sec} секунд, останавливаемся.")
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print(f"Видео сохранено в: {video_path}")
print(f"Сохранено обработанных лиц: {image_idx} шт. в папке {OUT_DIR}")
print("Готово.")
