"""Тесты для веб-приложения FastAPI и API маршрутов src/web/app.py и src/web/routes.py."""

from contextlib import contextmanager
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from tests.test_base import BaseTestCase
from src.models import (
    Document,
    DocumentJsonCache,
    DocumentValue,
    FeatureAlias,
    FeatureNode,
)
from src.web.app import app
from src.web.routes import _get_nested_value


class TestWebApp(BaseTestCase):
    """Проверка представлений и всех REST API эндпоинтов веб-сервера."""

    def setUp(self):
        super().setUp()
        self.client = TestClient(app)

    @contextmanager
    def _test_db_session(self):
        """Контекстный менеджер сессии для FastAPI потоков с единым StaticPool."""
        sess = self.SessionLocal()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    def test_index_page(self):
        """Проверка отдачи главной HTML страницы."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Медицинская Онтология", response.text)

    def test_get_nested_value_helper(self):
        """Проверка вспомогательной функции _get_nested_value."""
        data = {"vitals": {"temperature": 36.6, "bp": {"systolic": 120}}}
        self.assertAlmostEqual(_get_nested_value(data, "vitals.temperature"), 36.6)
        self.assertEqual(_get_nested_value(data, "vitals.bp.systolic"), 120)
        self.assertIsNone(_get_nested_value(data, "vitals.missing"))
        self.assertIsNone(_get_nested_value("not_a_dict", "vitals"))

    def test_get_questionnaires_empty(self):
        """Проверка списка анкет при пустой базе данных."""
        with patch("src.web.routes.get_db_session", side_effect=self._test_db_session):
            response = self.client.get("/api/questionnaires")
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.json(), list)

    def test_get_questionnaires_with_data(self):
        """Проверка получения списка анкет с заполненными метаданными."""
        doc = Document(
            profile_id=self.default_profile.id,
            patient_id="PAT-WEB-1",
            patient_name="Васильев",
            raw_text="...",
            administrative={"doctor": "Врач 1", "age": 55},
        )
        self.session.add(doc)
        self.session.commit()

        cache = DocumentJsonCache(
            document_id=doc.id,
            compiled_json={"conclusion": ["гастрит"]},
        )
        self.session.add(cache)
        self.session.commit()

        with patch("src.web.routes.get_db_session", side_effect=self._test_db_session):
            response = self.client.get("/api/documents")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["patient_id"], "PAT-WEB-1")
            self.assertEqual(data[0]["patient_name"], "Васильев")
            self.assertEqual(data[0]["diagnosis_preview"], "гастрит")

    def test_get_single_questionnaire_and_raw_json(self):
        """Проверка получения детальной анкеты и чистого JSON."""
        doc = Document(
            profile_id=self.default_profile.id,
            patient_id="PAT-DETAIL-1",
            patient_name="Павлов",
            raw_text="Исходный текст",
            administrative={"doctor": "Терапевт"},
        )
        self.session.add(doc)
        self.session.commit()

        cache = DocumentJsonCache(
            document_id=doc.id,
            compiled_json={"liver": {"size": 135}, "patient": "Павлов"},
        )
        self.session.add(cache)
        self.session.commit()

        with patch("src.web.routes.get_db_session", side_effect=self._test_db_session):
            # 1. Детальная карточка
            resp = self.client.get(f"/api/questionnaires/{doc.id}")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["patient_name"], "Павлов")
            self.assertEqual(data["json"]["liver"]["size"], 135)

            # 2. Чистый JSON
            resp_raw = self.client.get(f"/api/questionnaires/{doc.id}/raw-json")
            self.assertEqual(resp_raw.status_code, 200)
            self.assertEqual(resp_raw.json()["liver"]["size"], 135)

            # 3. 404 для несуществующего документа
            resp_404 = self.client.get("/api/questionnaires/999999")
            self.assertEqual(resp_404.status_code, 404)

    def test_get_ontology_graph(self):
        """Проверка эндпоинта графа онтологии для Vis.js."""
        node1 = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="liver",
            full_path="liver",
            display_name="Печень",
            is_leaf=False,
        )
        self.session.add(node1)
        self.session.commit()

        node2 = FeatureNode(
            profile_id=self.default_profile.id,
            parent_id=node1.id,
            key_name="right_lobe",
            full_path="liver.right_lobe",
            display_name="Правая доля",
            data_type="integer",
            is_leaf=True,
        )
        self.session.add(node2)
        self.session.commit()

        with patch("src.web.routes.get_db_session", side_effect=self._test_db_session):
            response = self.client.get("/api/ontology/graph?include_root=true")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("nodes", data)
            self.assertIn("edges", data)
            self.assertIn("stats", data)

            node_ids = [n["id"] for n in data["nodes"]]
            self.assertIn("root_0", node_ids)
            self.assertIn(node1.id, node_ids)
            self.assertIn(node2.id, node_ids)

            # Проверка отсутствия сырого HTML в подсказках title
            for n in data["nodes"]:
                if "title" in n and n["title"]:
                    self.assertNotIn("<div", n["title"])
                    self.assertNotIn("<span", n["title"])

    def test_get_node_values_across_questionnaires(self):
        """Проверка мини-таблицы значений признака у всех пациентов."""
        doc1 = Document(profile_id=self.default_profile.id, patient_id="P1", raw_text="..")
        doc2 = Document(profile_id=self.default_profile.id, patient_id="P2", raw_text="..")
        self.session.add_all([doc1, doc2])
        self.session.commit()

        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="temp",
            full_path="vitals.temp",
            display_name="Температура",
            data_type="float",
            unit="°C",
            is_leaf=True,
        )
        self.session.add(node)
        self.session.commit()

        alias = FeatureAlias(feature_id=node.id, alias_text="температура тела")
        self.session.add(alias)

        # Значение заполнено только у doc1
        val1 = DocumentValue(
            document_id=doc1.id,
            feature_id=node.id,
            value_numeric=36.8,
            confidence=0.99,
            raw_context="t 36.8",
        )
        self.session.add(val1)
        self.session.commit()

        with patch("src.web.routes.get_db_session", side_effect=self._test_db_session):
            response = self.client.get(f"/api/ontology/nodes/{node.id}/values")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["node"]["full_path"], "vitals.temp")
            self.assertIn("температура тела", data["node"]["aliases"])
            self.assertEqual(data["stats"]["total_documents"], 2)
            self.assertEqual(data["stats"]["filled_documents"], 1)
            self.assertAlmostEqual(data["stats"]["numeric_avg"], 36.8)

            docs = {d["patient_id"]: d for d in data["documents"]}
            self.assertTrue(docs["P1"]["is_present"])
            self.assertIn("36.8", docs["P1"]["value"])
            self.assertFalse(docs["P2"]["is_present"])
            self.assertIsNone(docs["P2"]["value"])

    def test_rename_ontology_node_success(self):
        """Проверка успешного переименования узла онтологии через API."""
        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="duodenitis",
            full_path="digestive_system.duodenitis",
            display_name="Duodenitis",
            data_type="string",
            is_leaf=True,
        )
        self.session.add(node)
        self.session.commit()

        with patch("src.web.routes.get_db_session", side_effect=self._test_db_session):
            response = self.client.post(
                f"/api/ontology/nodes/{node.id}/rename",
                json={"display_name": "Дуоденит (воспаление 12-перстной кишки)"},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["display_name"], "Дуоденит (воспаление 12-перстной кишки)")

            # Проверка обновления в БД
            self.session.refresh(node)
            self.assertEqual(node.display_name, "Дуоденит (воспаление 12-перстной кишки)")

    def test_rename_ontology_node_validation(self):
        """Проверка валидации при переименовании узла (пустое имя, несуществующий узел)."""
        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="gastritis",
            full_path="digestive_system.gastritis",
            display_name="Гастрит",
            data_type="string",
            is_leaf=True,
        )
        self.session.add(node)
        self.session.commit()

        with patch("src.web.routes.get_db_session", side_effect=self._test_db_session):
            # Пустое имя -> 400
            resp_empty = self.client.post(
                f"/api/ontology/nodes/{node.id}/rename",
                json={"display_name": "   "},
            )
            self.assertEqual(resp_empty.status_code, 400)

            # Несуществующий узел -> 404
            resp_404 = self.client.post(
                "/api/ontology/nodes/999999/rename",
                json={"display_name": "Новое имя"},
            )
            self.assertEqual(resp_404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
