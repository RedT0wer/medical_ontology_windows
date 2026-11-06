"""Базовые утилиты и фикстуры для тестирования приложения."""

import unittest
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from src.embeddings import BaseEmbeddingProvider
from src.models import Base, DocumentProfile


class DummyEmbeddingProvider(BaseEmbeddingProvider):
    """Тестовый провайдер эмбеддингов с детерминированным выводом."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        # Детерминированный единичный вектор на основе хэша строки
        val = (abs(hash(text)) % 1000) / 1000.0
        vec = [val] * self.dim
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            return [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class BaseTestCase(unittest.TestCase):
    """Базовый класс тестов с изолированной базой данных SQLite в памяти."""

    def setUp(self):
        # Создаем SQLite БД в памяти для каждого теста (StaticPool позволяет шарить соединение между потоками FastAPI)
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.session: Session = self.SessionLocal()
        self.embedding_provider = DummyEmbeddingProvider()

        # Создаем базовый профиль по умолчанию
        self.default_profile = DocumentProfile(
            code="us_abdomen",
            name="УЗИ органов брюшной полости",
            description="Тестовый профиль УЗИ ОБП",
        )
        self.session.add(self.default_profile)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()
