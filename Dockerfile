FROM python:3.13-slim

WORKDIR /app

# системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# код проекта
COPY . .

# папка для моделей
RUN mkdir -p models

CMD ["python", "src/predict.py"]