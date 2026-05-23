[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Определение фейковых новостей

**Студент:** Толмачев Макcим Сергеевич

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
├── api                         # Скрипт fast api
├── bot                         # Скрипт tg-бота
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
├── .dockerignore               # Докер
├── Dockerfile
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

# 4. Запустить Fast Api
uvicorn api.main:app --reload

# 5. Запустить бота (предварительно вставить токен в bot.py)
python bot/bot.py
```

Создание Docker-контейнера (деплой на сервер)
```
cd <repo-name>

docker build -t fake-news-detector .
# запуск
docker run -it fake-news-detector
```

## Данные
- `data/raw/` — исходные файлы (скрыты)
- `data/processed/` — предобработанные данные (скрыты)


## Результаты

```
                model  accuracy  f1_score
2           LinearSVC  0.981006  0.981444
0  LogisticRegression  0.968686  0.969254
3        RandomForest  0.965349  0.966157
4             XGBoost  0.951489  0.952333
1       MultinomialNB  0.931211  0.932899


Fitting 3 folds for each of 9 candidates, totalling 27 fits

Лучшая модель:
{'model': LinearSVC(random_state=42), 'model__C': 1}
Accuracy: 0.981006160164271
F1: 0.981444332998997
Training: LogisticRegression
Training: MultinomialNB
Training: LinearSVC
Training: RandomForest
Training: XGBoost
         stage               model  \
2     baseline           LinearSVC   
0     baseline  LogisticRegression   
3     baseline        RandomForest   
4     baseline             XGBoost   
5  grid_search           LinearSVC   
1     baseline       MultinomialNB   

                                              params           features  \
2  {'C': 1.0, 'class_weight': None, 'dual': 'auto...  TF-IDF (1,2), 30k   
0  {'C': 1.0, 'class_weight': None, 'dual': False...  TF-IDF (1,2), 30k   
3  {'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...  TF-IDF (1,2), 30k   
4  {'objective': 'binary:logistic', 'base_score':...  TF-IDF (1,2), 30k   
5  {'model': LinearSVC(random_state=42), 'model__...  TF-IDF (1,2), 30k   
1  {'alpha': 1.0, 'class_prior': None, 'fit_prior...  TF-IDF (1,2), 30k
 

   accuracy  f1_score                 comment  
2    0.9810    0.9814             без тюнинга  
0    0.9687    0.9693             без тюнинга  
3    0.9653    0.9662             без тюнинга  
4    0.9515    0.9523             без тюнинга  
5    0.9515    0.9523  подбор гиперпараметров  
1    0.9312    0.9329             без тюнинга  
 ``` 

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
