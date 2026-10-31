"""Управление подключением к PostgreSQL, расширением vector, сессиями и схемой БД."""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from src.config import logger, settings
from src.models import Base, DocumentProfile

# Создание синхронного движка SQLAlchemy
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Контекстный менеджер для безопасной работы с сессией БД."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка в транзакции базы данных: {e}")
        raise
    finally:
        session.close()


def init_db(drop_all: bool = False) -> None:
    """Инициализация базы данных:

    1. Подключение расширения pgvector.
    2. Создание таблиц (схемы признаков, документов, фактов, кэша JSON).
    3. Создание HNSW индексов для векторного поиска.
    4. Создание GIN индексов для JSONB полей.
    5. Инициализация профиля по умолчанию (us_abdomen).
    """
    logger.info("Проверка и инициализация базы данных PostgreSQL...")

    with engine.begin() as conn:
        # 1. Активация расширения pgvector
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        logger.info("Расширение 'vector' успешно проверено/подключено.")

        # 2. Очистка таблиц при необходимости
        if drop_all:
            logger.warning("Удаление существующих таблиц (drop_all=True)...")
            Base.metadata.drop_all(bind=conn)

        # 3. Создание таблиц
        Base.metadata.create_all(bind=conn)
        logger.info("Схема таблиц успешно создана.")

        # 4. Создание HNSW индексов для косинусного сходства векторов
        logger.info("Создание HNSW индексов для векторного поиска...")
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_feature_nodes_embedding_hnsw 
                ON feature_nodes USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_feature_aliases_embedding_hnsw 
                ON feature_aliases USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
        )

        # Legacy HNSW индексы (для обратной совместимости)
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_parameters_embedding_hnsw 
                ON parameters USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_parameter_synonyms_embedding_hnsw 
                ON parameter_synonyms USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
        )

        # 5. Создание GIN индексов для JSONB полей
        logger.info("Создание GIN индексов для JSONB метаданных...")
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_feature_nodes_metadata_gin 
                ON feature_nodes USING gin (metadata);
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_documents_administrative_gin 
                ON documents USING gin (administrative);
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_document_values_metadata_gin 
                ON document_values USING gin (metadata);
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_document_values_value_json_gin 
                ON document_values USING gin (value_json);
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_document_json_cache_gin 
                ON document_json_cache USING gin (compiled_json);
                """
            )
        )

        # Legacy GIN индексы
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_parameters_metadata_gin 
                ON parameters USING gin (metadata);
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_patient_records_metadata_gin 
                ON patient_records USING gin (metadata);
                """
            )
        )

        # 6. Базовый профиль документа по умолчанию (УЗИ органов брюшной полости)
        conn.execute(
            text(
                """
                INSERT INTO document_profiles (code, name, description)
                VALUES ('us_abdomen', 'Ультразвуковое исследование органов брюшной полости', 'Протокол УЗИ ОБП')
                ON CONFLICT (code) DO NOTHING;
                """
            )
        )

    logger.info("База данных успешно инициализирована со всеми индексами и базовым профилем.")


def get_default_profile(session: Session) -> DocumentProfile:
    """Получение или создание базового профиля протокола (us_abdomen)."""
    profile = session.query(DocumentProfile).filter(DocumentProfile.code == "us_abdomen").first()
    if not profile:
        profile = DocumentProfile(
            code="us_abdomen",
            name="Ультразвуковое исследование органов брюшной полости",
            description="Протокол УЗИ ОБП",
        )
        session.add(profile)
        session.flush()
    return profile
