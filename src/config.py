"""Модуль конфигурации сервиса на базе Pydantic Settings."""

import logging
import sys
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Параметры подключения к PostgreSQL
    DB_HOST: str = Field(default="127.0.0.1", description="Хост базы данных")
    DB_PORT: int = Field(default=5432, description="Порт базы данных")
    DB_USER: str = Field(default="postgres", description="Пользователь БД")
    DB_PASSWORD: str = Field(default="mysecretpassword", description="Пароль БД")
    DB_NAME: str = Field(default="medical_db", description="Имя базы данных")

    @property
    def database_url(self) -> str:
        """Синхронный URL для подключения к базе данных через psycopg3."""
        return f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def async_database_url(self) -> str:
        """Асинхронный URL для подключения к базе данных через psycopg3."""
        return f"postgresql+psycopg_async://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # Конфигурация векторных эмбеддингов
    EMBEDDING_PROVIDER: Literal["fastembed", "ollama", "openai"] = Field(
        default="fastembed",
        description="Провайдер эмбеддингов",
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Название модели эмбеддингов",
    )
    EMBEDDING_DIM: int = Field(
        default=384,
        description="Размерность векторов эмбеддингов",
    )

    # Конфигурация LLM (Qwen / Ollama / OpenAI / LiteLLM)
    LLM_PROVIDER: Literal["ollama", "litellm", "openai"] = Field(
        default="ollama",
        description="Провайдер LLM",
    )
    LLM_MODEL: str = Field(
        default="qwen2.5:14b-instruct-q4_K_M",
        description="Идентификатор модели LLM (например, qwen2.5:14b-instruct-q4_K_M или qwen2.5:7b)",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://127.0.0.1:11434",
        description="Базовый URL Ollama сервиса",
    )
    OPENAI_API_KEY: str = Field(
        default="",
        description="API ключ OpenAI / стороннего провайдера",
    )
    OPENAI_BASE_URL: str = Field(
        default="",
        description="Пользовательский Base URL для OpenAI-совместимых сервисов",
    )

    # Пороги Matching Engine
    SIMILARITY_THRESHOLD_HIGH: float = Field(
        default=0.90,
        description="Порог высокой уверенности (точный синоним)",
    )
    SIMILARITY_THRESHOLD_LOW: float = Field(
        default=0.70,
        description="Порог низкой уверенности (ниже которого признается новым параметром)",
    )

    # Интерактивный арбитраж (Human-in-the-loop с модальным окном)
    HUMAN_IN_THE_LOOP: bool = Field(
        default=True,
        description="Включение модального окна подтверждения арбитража экспертом",
    )

    # Уровень логирования
    LOG_LEVEL: str = Field(default="INFO", description="Уровень логирования")



settings = Settings()


class RussianLogFormatter(logging.Formatter):
    """Форматтер для отображения уровней логирования на русском языке."""

    LEVEL_MAP = {
        "DEBUG": "ОТЛАДКА",
        "INFO": "ИНФО",
        "WARNING": "ВНИМАНИЕ",
        "ERROR": "ОШИБКА",
        "CRITICAL": "КРИТИЧНО",
    }

    def format(self, record: logging.LogRecord) -> str:
        level_ru = self.LEVEL_MAP.get(record.levelname, record.levelname)
        record.levelname = level_ru
        return super().format(record)


def setup_logger(name: str = "medical_ontology") -> logging.Logger:
    """Инициализация логгера с русскоязычным выводом."""
    logger = logging.getLogger(name)
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = RussianLogFormatter(
            fmt="[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
