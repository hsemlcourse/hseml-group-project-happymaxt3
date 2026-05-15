import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


print("Загрузка данных")

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

# текст и метки
X_train = train_df["full_text"]
y_train = train_df["target"]

X_val = val_df["full_text"]
y_val = val_df["target"]


print("TF-IDF векторизация")

tfidf = TfidfVectorizer(
    max_features=60000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    stop_words="english",
    sublinear_tf=True,
    dtype=np.float32
)

X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)

print("Обучение модели")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_vec, y_train)

# предикт
y_pred = model.predict(X_val_vec)

print("\nRESULTS")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1-score:", f1_score(y_val, y_pred))

# сохранение модели
with open(MODELS_DIR / "fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

# сохранение tfidf
with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\nМодель сохранена в папку models/")