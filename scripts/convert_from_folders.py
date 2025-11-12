from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
TRAIN_DIR = DATA / 'train'
TEST_DIR  = DATA / 'test'
OUT_TRAIN = DATA / 'train.csv'
OUT_TEST  = DATA / 'test.csv'

# === Маппинг классов -> метки (FER порядок)
# 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
label_map = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6,
}

def collect_rows(split_dir: Path):
    rows = []
    if not split_dir.exists():
        raise SystemExit(f"Не найдена папка: {split_dir}")

    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    print(f"{split_dir.name} классы:", classes)
    for cls in classes:
        cls_lower = cls.lower()
        if cls_lower not in label_map:
            raise SystemExit(f"Неизвестный класс '{cls}'. "
                             f"Либо дополни label_map в скрипте, либо приведи имена папок к одному из: {list(label_map.keys())}")
        label = label_map[cls_lower]
        img_dir = split_dir / cls
        for p in img_dir.rglob('*'):
            if not p.is_file():
                continue
            # читаем изображение
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # пропускаем не-изображения
            # приводим к 48x48
            img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
            # вектор пикселей
            flat = img.reshape(-1)
            rows.append({'emotion': label, 'pixels': ' '.join(map(str, flat.tolist()))})
    return rows

print("Сканирую train...")
train_rows = collect_rows(TRAIN_DIR)
print("Сканирую test...")
test_rows  = collect_rows(TEST_DIR)

print(f"Сохраняю CSV: {OUT_TRAIN} (rows={len(train_rows)}), {OUT_TEST} (rows={len(test_rows)})")
pd.DataFrame(train_rows).to_csv(OUT_TRAIN, index=False)
pd.DataFrame(test_rows).to_csv(OUT_TEST, index=False)
print("Готово.")
