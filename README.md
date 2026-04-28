[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Определение фейковых новостей

**Студент:** Толмачев Маким Сергеевич

**Группа:** БИВ238


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Бинарная классификация

**Датасет:** [[fake-and-real-news-dataset\]](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Данные были найдены на платформе Kaggle.

Преимущества:
- готовая разметка fake/real
- достаточный объем данных
- актуальность

**Датасет после объединения и предобработки:**

- Размер: ~44,000 записей
- Признаки:
  - title — заголовок новости
  - text — основной текст новости
  - subject — тема новости (не используется в модели)
  - date — дата публикации (не используется в модели)
  - target — целевая переменная:
    - 0 — real news
    - 1 — fake news

**Целевые метрики:** 
- Accuracy
- F1-score (учитывает баланс precision/recall)
- ROC-AUC (оценка разделимости классов)

F1 выбрана как основная метрика, так как задача классификации может быть чувствительна к ошибкам обоих классов.


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

## Данные
- `data/raw/` — исходные файлы (скрыты)
- `data/processed/` — предобработанные данные (скрыты)


## Результаты
                model  accuracy  f1_score
2           LinearSVC  0.981006  0.981444
0  LogisticRegression  0.968686  0.969254
3        RandomForest  0.965349  0.966157
4             XGBoost  0.951489  0.952333
1       MultinomialNB  0.931211  0.932899


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
