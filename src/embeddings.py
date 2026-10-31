"""Модуль генерации векторных эмбеддингов для медицинских терминов."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import requests
from src.config import logger, settings


def l2_normalize(vector: List[float]) -> List[float]:
    """Нормализация вектора по евклидовой норме (L2)."""
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


class BaseEmbeddingProvider(ABC):
    """Абстрактный базовый класс для провайдеров эмбеддингов."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Получение вектора для одной текстовой строки."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Пакетное получение векторов."""
        pass


class FastEmbedProvider(BaseEmbeddingProvider):
    """Локальный провайдер на базе FastEmbed (ONNX Runtime, без нагрузки на GPU)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        logger.info(f"Загрузка локальной модели FastEmbed: {model_name}...")
        self.model = TextEmbedding(model_name=model_name)
        logger.info("Модель FastEmbed успешно инициализирована.")

    def embed_text(self, text: str) -> List[float]:
        clean_text = text.strip()
        embeddings = list(self.model.embed([clean_text]))
        return l2_normalize(embeddings[0].tolist())

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [t.strip() for t in texts]
        embeddings = list(self.model.embed(clean_texts))
        return [l2_normalize(e.tolist()) for e in embeddings]


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Провайдер эмбеддингов через локальный сервис Ollama."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        logger.info(f"Инициализация Ollama эмбеддингов: {model_name} ({self.base_url})")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model_name, "input": texts}
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            return [l2_normalize(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддингов из Ollama: {e}")
            raise


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Провайдер через OpenAI API / LiteLLM."""

    def __init__(self, model_name: str, api_key: str, base_url: str = ""):
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model_name = model_name

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [l2_normalize(item.embedding) for item in response.data]


_embedding_service: BaseEmbeddingProvider | None = None


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Синглтон фабрика для провайдера эмбеддингов."""
    global _embedding_service
    if _embedding_service is not None:
        return _embedding_service

    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "fastembed":
        _embedding_service = FastEmbedProvider(model_name=settings.EMBEDDING_MODEL)
    elif provider == "ollama":
        _embedding_service = OllamaEmbeddingProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model_name=settings.EMBEDDING_MODEL,
        )
    elif provider == "openai":
        _embedding_service = OpenAIEmbeddingProvider(
            model_name=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
    else:
        raise ValueError(f"Неизвестный провайдер эмбеддингов: {provider}")

    return _embedding_service
