# Используем официальный образ Python 3.12-slim
FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y ffmpeg flac && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Копируем исходный код и данные
COPY app/ ./app/
COPY data/ ./data/

# Устанавливаем переменную окружения для Python
ENV PYTHONPATH=/app

# Команда по умолчанию (будет переопределена в docker-compose.yml)
CMD ["python3", "app/bot.py"]