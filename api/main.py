import logging
import random
from contextlib import asynccontextmanager
import PIL
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from utils.funcs import load_model, load_tokenizer, get_sentiment

logger = logging.getLogger('uvicorn.info')


# Определение класса запроса для классификации текста
class TextInput(BaseModel):
    text: str  # Текст, введенный пользователем для классификации

# Определение класса ответа для классификации текста
class TextResponse(BaseModel):
    label: str  # Метка класса, например, positive или negative
    prob: float # Вероятность, связанная с меткой


pt_model = None  # Глобальная переменная для PyTorch модели
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для инициализации и завершения работы FastAPI приложения.
    Загружает модели машинного обучения при запуске приложения и удаляет их после завершения.
    """
    global pt_model
    global tokenizer
    # Загрузка PyTorch модели
    pt_model = load_model()
    tokenizer = load_tokenizer()
    logger.info('Torch model loaded')
    logger.info('Tokenizer loaded')
    yield
    # Удаление моделей и освобождение ресурсов
    del pt_model, tokenizer

app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    """
    Возвращает приветственное сообщение при обращении к корневому маршруту API.
    """
    return 'My FastAPI is working goddamn!!'

@app.post('/clf_text')
def clf_text(data: TextInput):
    """
    Эндпоинт для классификации текста.
    Случайно генерирует метку класса и вероятность для демонстрационных целей.
    """
    result = get_sentiment(data.text)
    responce = TextResponse(
        label=result[0],
        prob=max(result[1])
    )
    return responce


if __name__ == "__main__":
    # Запуск приложения на localhost с использованием Uvicorn
    # производится из командной строки: python your/path/api/main.py
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)
