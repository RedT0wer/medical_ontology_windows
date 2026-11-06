"""Тесты для модуля сопоставления онтологии (Matching Engine) src/matcher.py."""

import json
import unittest
from unittest.mock import MagicMock, patch
from tests.test_base import BaseTestCase, DummyEmbeddingProvider
from src.matcher import (
    CandidateMatch,
    MatchDecision,
    MatchingEngine,
    normalize_value_types,
)
from src.models import FeatureAlias, FeatureNode


class TestMatcher(BaseTestCase):
    """Проверка нормализации путей, типов и алгоритма сопоставления признаков."""

    def setUp(self):
        super().setUp()
        self.matcher = MatchingEngine(
            session=self.session,
            embedding_provider=self.embedding_provider,
            profile_code="us_abdomen",
        )

    def test_dataclasses_compatibility(self):
        """Проверка dataclass CandidateMatch и MatchDecision, а также legacy-свойств."""
        cm = CandidateMatch(
            feature_id=10,
            full_path="liver.right_lobe",
            matched_text="правая доля",
            match_source="alias",
            similarity=0.95,
        )
        self.assertEqual(cm.parameter_id, 10)
        self.assertEqual(cm.canonical_name, "liver.right_lobe")

        md = MatchDecision(
            action="MATCH",
            feature_id=10,
            full_path="liver.right_lobe",
            similarity=0.95,
            reason="Точное совпадение",
        )
        self.assertEqual(md.parameter_id, 10)
        self.assertEqual(md.canonical_name, "liver.right_lobe")

    def test_canonicalize_path(self):
        """Проверка построения канонических путей с анатомическим маппингом."""
        # 1. С path_hint
        p1 = self.matcher._canonicalize_path("печень.желчный пузырь.стенка", "стенка", None)
        self.assertEqual(p1, "liver.gallbladder.wall")

        # 2. С категорией
        p2 = self.matcher._canonicalize_path(None, "правая доля", "печень")
        self.assertEqual(p2, "liver.right_lobe")

        # 3. Чисто raw_key
        p3 = self.matcher._canonicalize_path(None, "поджелудочная железа", None)
        self.assertEqual(p3, "pancreas")

    def test_normalize_value_types(self):
        """Проверка функции normalize_value_types для различных типов данных."""
        # None
        num, txt, jsn = normalize_value_types(None, "integer")
        self.assertIsNone(num)
        self.assertIsNone(txt)
        self.assertIsNone(jsn)

        # Словари / массивы -> value_json
        test_dict = {"status": "normal"}
        num, txt, jsn = normalize_value_types(test_dict, "object")
        self.assertEqual(jsn, test_dict)
        self.assertIsNone(num)

        # Числовые типы из строки с единицами измерения
        num, txt, jsn = normalize_value_types("140.5 мм", "float")
        self.assertAlmostEqual(num, 140.5)

        num, txt, jsn = normalize_value_types("150", "integer")
        self.assertEqual(num, 150.0)

        # Строка с запятой
        num, txt, jsn = normalize_value_types("36,6 °C", "float")
        self.assertAlmostEqual(num, 36.6)

        # Двухмерный габарит -> value_text
        num, txt, jsn = normalize_value_types("65  х  28 мм", "dimension_2d")
        self.assertIsNone(num)
        self.assertEqual(txt, "65  х  28 мм")

        # Обычная строка
        num, txt, jsn = normalize_value_types("диффузные изменения", "string")
        self.assertEqual(txt, "диффузные изменения")

    def test_ensure_feature_path_creates_hierarchy(self):
        """Проверка автоматического рекурсивного создания цепочки узлов онтологии."""
        leaf = self.matcher.ensure_feature_path(
            full_path="liver.gallbladder.wall",
            data_type="float",
            unit="мм",
            display_name="Толщина стенки желчного пузыря",
        )
        self.session.commit()

        self.assertIsNotNone(leaf.id)
        self.assertEqual(leaf.full_path, "liver.gallbladder.wall")
        self.assertEqual(leaf.data_type, "float")
        self.assertEqual(leaf.unit, "мм")
        self.assertTrue(leaf.is_leaf)

        # Проверяем промежуточные родительские узлы
        gallbladder = self.session.query(FeatureNode).filter_by(full_path="liver.gallbladder").first()
        self.assertIsNotNone(gallbladder)
        self.assertFalse(gallbladder.is_leaf)
        self.assertEqual(gallbladder.data_type, "object")

        liver = self.session.query(FeatureNode).filter_by(full_path="liver").first()
        self.assertIsNotNone(liver)
        self.assertIsNone(liver.parent_id)
        self.assertEqual(gallbladder.parent_id, liver.id)
        self.assertEqual(leaf.parent_id, gallbladder.id)

    def test_match_parameter_exact_path(self):
        """Проверка сопоставления при точном совпадении с существующим full_path."""
        node = self.matcher.ensure_feature_path(
            full_path="vitals.heart_rate",
            data_type="integer",
            unit="уд/мин",
        )
        self.session.commit()

        decision = self.matcher.match_parameter(
            raw_key="heart_rate",
            raw_value="75",
            path_hint="vitals.heart_rate",
        )

        self.assertEqual(decision.action, "MATCH")
        self.assertEqual(decision.feature_id, node.id)
        self.assertEqual(decision.value_numeric, 75.0)
        self.assertAlmostEqual(decision.similarity, 1.0)
        self.assertIn("Точное совпадение", decision.reason)

    def test_match_parameter_exact_alias(self):
        """Проверка сопоставления по точному псевдониму (alias)."""
        node = self.matcher.ensure_feature_path(
            full_path="vitals.heart_rate",
            data_type="integer",
            unit="уд/мин",
        )
        alias = FeatureAlias(feature_id=node.id, alias_text="пульс")
        self.session.add(alias)
        self.session.commit()

        with patch.object(self.matcher, "find_top_candidates") as mock_find:
            mock_find.return_value = [
                CandidateMatch(
                    feature_id=node.id,
                    full_path="vitals.heart_rate",
                    matched_text="пульс",
                    match_source="alias",
                    similarity=0.95,
                )
            ]
            decision = self.matcher.match_parameter(
                raw_key="пульс",
                raw_value=80,
                path_hint=None,
            )

        self.assertEqual(decision.action, "MATCH")
        self.assertEqual(decision.feature_id, node.id)
        self.assertEqual(decision.value_numeric, 80.0)

    def test_match_parameter_new_feature_creation(self):
        """Проверка динамического расширения схемы при новом признаке."""
        decision = self.matcher.match_parameter(
            raw_key="селезенка_размер",
            raw_value="110х45 мм",
            path_hint="spleen.size",
            data_type="dimension_2d",
        )
        self.session.commit()

        self.assertEqual(decision.action, "NEW_FEATURE")
        self.assertEqual(decision.full_path, "spleen.size")
        self.assertEqual(decision.value_text, "110х45 мм")

        # Узел должен быть создан в БД
        saved_node = self.session.query(FeatureNode).filter_by(full_path="spleen.size").first()
        self.assertIsNotNone(saved_node)
        self.assertEqual(saved_node.status, "candidate")

    def test_arbitrate_with_llm(self):
        """Проверка LLM-арбитража серой зоны."""
        node = self.matcher.ensure_feature_path("vitals.pulse", data_type="integer")
        self.session.commit()

        candidates = [
            CandidateMatch(
                feature_id=node.id,
                full_path="vitals.pulse",
                matched_text="пульс",
                match_source="feature",
                similarity=0.82,
            )
        ]

        # Тест ветки MATCH
        with patch.object(self.matcher, "_call_llm") as mock_llm:
            mock_llm.return_value = json.dumps({
                "action": "MATCH",
                "matched_feature_id": node.id,
                "reason": "Синоним пульса",
            })
            decision = self.matcher._arbitrate_with_llm(
                raw_key="ЧСС",
                path_hint="vitals.chss",
                raw_value=72,
                context="ЧСС 72 уд/мин",
                candidates=candidates,
            )
            self.assertEqual(decision.action, "MATCH")
            self.assertEqual(decision.feature_id, node.id)

        # Тест ветки NEW_FEATURE
        with patch.object(self.matcher, "_call_llm") as mock_llm:
            mock_llm.return_value = json.dumps({
                "action": "NEW_FEATURE",
                "full_path": "vitals.blood_pressure",
                "data_type": "string",
                "reason": "Отдельный параметр АД",
            })
            decision = self.matcher._arbitrate_with_llm(
                raw_key="АД",
                path_hint="vitals.ad",
                raw_value="120/80",
                context="АД 120/80 мм рт ст",
                candidates=candidates,
            )
            self.assertEqual(decision.action, "NEW_FEATURE")
            self.assertEqual(decision.full_path, "vitals.blood_pressure")


if __name__ == "__main__":
    unittest.main()
