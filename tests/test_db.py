"""Тесты для модуля работы с базой данных src/db.py."""

import unittest
from unittest.mock import MagicMock, patch
from tests.test_base import BaseTestCase
from src.db import get_db_session, get_default_profile
from src.models import DocumentProfile


class TestDB(BaseTestCase):
    """Проверка функций инициализации, управления сессиями и профилями."""

    def test_get_default_profile_existing(self):
        """Проверка получения уже существующего профиля по умолчанию."""
        profile = get_default_profile(self.session)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.code, "us_abdomen")
        self.assertEqual(profile.id, self.default_profile.id)

    def test_get_default_profile_creates_when_missing(self):
        """Проверка создания профиля по умолчанию, если его нет в БД."""
        self.session.delete(self.default_profile)
        self.session.commit()

        # Теперь профиля нет
        found = self.session.query(DocumentProfile).filter_by(code="us_abdomen").first()
        self.assertIsNone(found)

        # Функция должна автоматически его создать
        created = get_default_profile(self.session)
        self.assertIsNotNone(created)
        self.assertEqual(created.code, "us_abdomen")
        self.assertEqual(created.name, "Ультразвуковое исследование органов брюшной полости")

    def test_get_db_session_success_commit(self):
        """Проверка фиксации транзакции контекстным менеджером get_db_session."""
        with patch("src.db.SessionLocal", return_value=self.session):
            with get_db_session() as sess:
                new_profile = DocumentProfile(
                    code="custom_profile",
                    name="Кастомный профиль",
                )
                sess.add(new_profile)

        # Проверяем, что объект зафиксирован в БД
        saved = self.session.query(DocumentProfile).filter_by(code="custom_profile").first()
        self.assertIsNotNone(saved)

    def test_get_db_session_rollback_on_exception(self):
        """Проверка отката транзакции при возникновении исключения."""
        mock_sess = MagicMock()
        mock_sess.commit.side_effect = Exception("DB Commit Failure")

        with patch("src.db.SessionLocal", return_value=mock_sess):
            with self.assertRaises(Exception):
                with get_db_session() as sess:
                    pass

        mock_sess.rollback.assert_called_once()
        mock_sess.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

