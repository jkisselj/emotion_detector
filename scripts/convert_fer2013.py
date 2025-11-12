from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'data' / 'fer2013.csv'
dst_train = ROOT / 'data' / 'train.csv'
dst_test  = ROOT / 'data' / 'test.csv'

if not src.exists():
    raise SystemExit(f"Нет файла: {src}\nСкачай fer2013.csv и положи в data/")

df = pd.read_csv(src)
need = {'emotion','pixels','Usage'}
if not need.issubset(df.columns):
    raise SystemExit(f"Ожидались колонки {need}, получили {set(df.columns)}")

# Оставляем базовые 7 классов (0..6)
df = df[df['emotion'].between(0,6)].dropna(subset=['pixels','emotion'])

train_df = df[df['Usage']=='Training'][['emotion','pixels']].reset_index(drop=True)
test_df  = df[df['Usage'].isin(['PublicTest','PrivateTest'])][['emotion','pixels']].reset_index(drop=True)

train_df.to_csv(dst_train, index=False)
test_df.to_csv(dst_test, index=False)

print(f"Готово:\n  {dst_train} (rows={len(train_df)})\n  {dst_test}  (rows={len(test_df)})")
