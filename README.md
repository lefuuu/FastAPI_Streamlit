# 🚀 FastAPI + Streamlit в Docker

Этот проект содержит два сервиса:
1. **FastAPI** (бэкенд, обработка запросов)
2. **Streamlit** (фронтенд, взаимодействие с пользователем)

Сервисы работают в отдельных Docker-контейнерах и взаимодействуют через `docker-compose`.

## 📂 Структура проекта
```
/project
│── /api                  # Папка с FastAPI
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│── /front                # Папка со Streamlit
│   ├── streamlit_app.py
│   ├── Dockerfile
│   ├── requirements.txt
│── docker-compose.yml
│── README.md
```

---

## 🔧 Установка и запуск

### 1️⃣ Клонируем репозиторий
```sh
git clone https://github.com/your-repo.git
cd project
```

### 2️⃣ Запускаем контейнеры
```sh
docker compose up --build
```

---

## 🔹 Доступ к сервисам
После успешного запуска:
- **FastAPI** → [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit** → [http://localhost:8501](http://localhost:8501)

---

## 📌 API (FastAPI)

### 🔹 Эндпоинты

#### 1️⃣ **POST /clf_text** – анализ текста
- **Запрос:**
```json
{
    "text": "Пример текста"
}
```
- **Ответ:**
```json
{
    "label": "positive",
    "prob": 0.85
}
```

---

## 🎨 Frontend (Streamlit)
Приложение отправляет введённый текст в API и отображает ответ.

### 🔹 Запуск вручную (без Docker)
```sh
cd front
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---


🚀 **Готово к использованию!**
