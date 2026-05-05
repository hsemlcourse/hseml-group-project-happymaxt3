import re
from pathlib import Path
import nltk
from nltk.stem import WordNetLemmatizer

import pandas as pd
from sklearn.model_selection import train_test_split

nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def clean_text(text: str) -> str:
    text = str(text).lower()

    # - ссылки
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"pic\.twitter\S+", " ", text)

    # - источники и мусорные сигналы
    patterns = [
        r"\breuters\b",
        r"\bfeatured image\b",
        r"\bfeatured video\b",
        r"\bread entire story\b",
        r"\bvia twitter\b",
        r"\bvia wikimedia\b",
        r"\bdailymailonline\b",
        r"\bfacebook\b",
        r"\btwitter\b",
        r"\bvideo\b",
        r"\bimage\b",
        r"\bflickr\b",
        r"\bwikimedia\b",
        r"\bpic\.twitter\.com\S*\b",

        # частовстречающиеся слова
        r"\bthe\b",
        r"\band\b",
        r"\bthat\b",
        r"\bfor\b",
        r"\btrump\b",

        r"\bthis\b",
        r"\byou\b",
        r"\bhis\b",
        r"\bthey\b",
        r"\bare\b",

        r"\bsaid\b",
        r"\bwa\b",
        r"\bha\b",
    ]

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # - пробелы
    text = re.sub(r"\s+", " ", text).strip()

    for pattern in patterns:
        text = re.sub(pattern, " ", text)

    words = text.split()

    # - первые 5 слов (часто заголовок / мусор / источник)
    if len(words) > 5:
        words = words[5:]

    # - ограничение длины (убирает влияние длины статьи)
    words = words[:250]

    # - короткие слова
    words = [word for word in words if len(word) > 2]

    # - лемматизация
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


def load_data():
    true_path = RAW_DIR / "True.csv"
    false_path = RAW_DIR / "Fake.csv"

    df_true = pd.read_csv(true_path)
    df_false = pd.read_csv(
        false_path,
        engine="python",
        quotechar='"',
        escapechar="\\"
    )

    df_true["target"] = 0  # реал
    df_false["target"] = 1  # факе

    return pd.concat([df_true, df_false], ignore_index=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\nПропуски до очистки:")
    print(df.isnull().sum())

    # - пустые строки
    df.dropna(subset=["title", "text"], inplace=True)

    # - дубли
    print("\nДублей до удаления:", df.duplicated().sum())
    df.drop_duplicates(inplace=True)

    # дата (чистим строку перед преобразованием)
    df["date"] = df["date"].astype(str).str.strip()
    df["date"] = df["date"].str.replace('"', "", regex=False)
    df["date"] = df["date"].str.replace("\r", "", regex=False)
    df["date"] = df["date"].str.replace("\n", "", regex=False)

    # дата (НЕ удаляем строки, иначе ломается класс fake)
    df["date"] = pd.to_datetime(
        df["date"],
        format="%B %d, %Y",
        errors="coerce"
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    # - title как источник утечки
    df["full_text"] = df["text"].fillna("")
    df["full_text"] = df["full_text"].apply(clean_text)

    df["text_length"] = df["full_text"].apply(len)
    df["word_count"] = df["full_text"].apply(lambda x: len(x.split()))
    df["title_length"] = df["title"].fillna("").apply(len)

    # фичи из даты (без удаления NaT)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday

    # короткие / пустые статьи убираем
    df = df[df["word_count"] > 25]

    # выбросы по длине текста (верхний 99 перцентиль)
    max_words = df["word_count"].quantile(0.9)
    df = df[df["word_count"] <= max_words]

    return df


def preprocess():
    print("\nЗагрузка данных...")
    df = load_data()

    print("\nРазмер до очистки:", df.shape)
    print("\nТипы данных:")
    print(df.dtypes)

    df = clean_data(df)
    df = create_features(df)

    print("\nРазмер после очистки:", df.shape)

    print("\nБаланс классов:")
    print(df["target"].value_counts())

    print("\nПустых дат после обработки:")
    print(df["date"].isna().sum())

    # train / val / test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["target"],
        random_state=42
    )

    # сохранение
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)  # обучу
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)  # подберу гиперпарамы
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)  # протесщу
    df.to_csv(PROCESSED_DIR / "full_dataset.csv", index=False)  # по приколу

    print("Файлы сохранены в data/processed/")
    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)


if __name__ == "__main__":
    preprocess()
    print(PROCESSED_DIR.resolve())