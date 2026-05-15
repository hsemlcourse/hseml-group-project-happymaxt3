import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = BASE_DIR / "models"


print("Загрузка модели")

with open(MODELS_DIR / "fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


print("\nInput news line:")
text = input("> ")

# vectorize
text_vec = tfidf.transform([text])

# predict
prediction = model.predict(text_vec)[0]

# probability
probability = model.predict_proba(text_vec)[0][prediction]

print()

if prediction == 1:
    print(f"FAKE NEWS ❌ ({probability:.2%})")
else:
    print(f"REAL NEWS ✅ ({probability:.2%})")