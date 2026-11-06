"""Тесты для модуля конфигурации и логирования src/config.py."""

import logging
import unittest
from src.config import RussianLogFormatter, Settings, setup_logger


class TestConfig(unittest.TestCase):
    """Проверка конфигурации Pydantic Settings и форматтера логов."""

    def test_settings_default_values(self):
        """Проверка значений по умолчанию в Settings."""
        s = Settings()
        self.assertEqual(s.DB_HOST, "127.0.0.1")
        self.assertEqual(s.DB_PORT, 5432)
        self.assertEqual(s.DB_USER, "postgres")
        self.assertEqual(s.DB_NAME, "medical_db")
        self.assertEqual(s.EMBEDDING_PROVIDER, "fastembed")
        self.assertEqual(s.EMBEDDING_DIM, 384)
        self.assertEqual(s.LLM_PROVIDER, "ollama")
        self.assertAlmostEqual(s.SIMILARITY_THRESHOLD_HIGH, 0.90)
        self.assertAlmostEqual(s.SIMILARITY_THRESHOLD_LOW, 0.70)
        self.assertTrue(s.HUMAN_IN_THE_LOOP)

    def test_database_url_properties(self):
        """Проверка генерации синхронного и асинхронного URL подключения к БД."""
        s = Settings(
            DB_HOST="db.example.com",
            DB_PORT=5433,
            DB_USER="admin",
            DB_PASSWORD="secretpassword",
            DB_NAME="test_medical",
        )
        self.assertEqual(
            s.database_url,
            "postgresql+psycopg://admin:secretpassword@db.example.com:5433/test_medical",
        )
        self.assertEqual(
            s.async_database_url,
            "postgresql+psycopg_async://admin:secretpassword@db.example.com:5433/test_medical",
        )

    def test_russian_log_formatter(self):
        """Проверка перевода уровней логирования на русский язык."""
        formatter = RussianLogFormatter(fmt="[%(levelname)s] %(message)s")

        test_cases = [
            ("DEBUG", "ОТЛАДКА"),
            ("INFO", "ИНФО"),
            ("WARNING", "ВНИМАНИЕ"),
            ("ERROR", "ОШИБКА"),
            ("CRITICAL", "КРИТИЧНО"),
            ("CUSTOM", "CUSTOM"),
        ]

        for level_name, expected_name in test_cases:
            record = logging.LogRecord(
                name="test",
                level=getattr(logging, level_name, 20),
                pathname="test.py",
                lineno=10,
                msg="Тестовое сообщение",
                args=(),
                exc_info=None,
            )
            record.levelname = level_name
            formatted = formatter.format(record)
            self.assertIn(f"[{expected_name}]", formatted)

    def test_setup_logger(self):
        """Проверка инициализации логгера."""
        test_logger = setup_logger("test_custom_logger")
        self.assertIsInstance(test_logger, logging.Logger)
        self.assertTrue(len(test_logger.handlers) >= 1)
        self.assertIsInstance(test_logger.handlers[0].formatter, RussianLogFormatter)


if __name__ == "__main__":
    unittest.main()
