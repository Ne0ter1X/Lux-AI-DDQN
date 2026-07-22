# Lux AI Season 1 DDQN Agent + FastAPI

Проект представляет собой полный ML pipeline для соревнования Kaggle Lux AI Season 1.

В основе решения находится агент обучения с подкреплением DDQN (Double Deep Q-Network), который обучается взаимодействовать с игровой средой Lux AI.

После обучения модель сохраняется в формате `.h5` и используется backend-сервисом на базе FastAPI для выполнения inference-запросов.

## Проект включает:


- обучение DDQN агента;
- обработку игрового состояния;
- inference pipeline для готовой модели;
- REST API для получения действий агента;
- Docker-контейнеризацию сервиса.


**Цель проекта** - реализовать полный ML pipeline: от обучения RL-агента до предоставления модели через REST API.
## Структура проекта

```text
.
├── agent/              # Логика DDQN агента, inference и обработка состояния
├── app/                # FastAPI backend-сервис
├── training/           # Pipeline обучения DDQN модели
├── lux/                # Утилиты и игровые объекты Lux AI
├── models/             # Обученные модели (.h5)
├── scripts/            # Вспомогательные скрипты
├── test_client.py      # Клиент для оценки работы API
├── Dockerfile
└── requirements.txt
```

## Запуск проекта
**Вариант 1: Docker (рекомендуется)**

Сборка Docker image
```bash
docker build -t lux-ai-ddqn .
```

Запуск сервиса
```bash
docker run --rm -p 8000:8000 lux-ai-ddqn
```

После запуска API доступен:
```
http://localhost:8000
```
Swagger UI:
```
http://localhost:8000/docs
```
**Вариант 2: Локальный запуск Python**

Создание виртуального окружения:
```bash
python -m venv .venv
```

**Активация окружения:**

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

Установка зависимостей:
```bash
pip install -r requirements.txt
```

Запуск FastAPI:
```bash
uvicorn app.main:app --reload
```

## API

Основной endpoint:
```text
POST /predict
```

Endpoint принимает игровое состояние Lux AI и возвращает действие агента.

Описание API и примеры запросов доступны через Swagger:
```
http://localhost:8000/docs
```
## Обучение модели

Для запуска обучения DDQN:

```bash
python scripts/run_training.py
```
После завершения обучения модель сохраняется в формате `.h5`.

Используемая backend-сервисом модель находится в директории`models/`.

## Оценка агента

Для проверки качества модели используется симуляция игр через API.

Запуск оценки:
```bash
python test_client.py
```

Скрипт выполняет серию игр против базового агента и собирает:

- количество побед и поражений;
- среднюю награду;
- количество шагов;
- latency API.

## Технологии
- Python
- TensorFlow / Keras
- Reinforcement Learning (DDQN)
- FastAPI
- Docker
- Kaggle Lux AI Season 1 Environment
