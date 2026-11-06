"""Тесты для моделей данных SQLAlchemy src/models.py."""

from datetime import datetime
import unittest
from tests.test_base import BaseTestCase
from src.models import (
    Document,
    DocumentJsonCache,
    DocumentProfile,
    DocumentValue,
    FeatureAlias,
    FeatureNode,
    Parameter,
    ParameterSynonym,
    PatientRecord,
)


class TestModels(BaseTestCase):
    """Проверка создания, связей и валидации моделей данных."""

    def test_document_profile_creation(self):
        """Проверка модели DocumentProfile."""
        profile = DocumentProfile(
            code="echo_cg",
            name="Эхокардиография",
            description="Протокол исследования сердца",
        )
        self.session.add(profile)
        self.session.commit()

        self.assertIsNotNone(profile.id)
        self.assertIn("echo_cg", repr(profile))
        self.assertEqual(profile.name, "Эхокардиография")

    def test_feature_node_hierarchy(self):
        """Проверка иерархических связей FeatureNode (дерево признаков)."""
        parent = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="liver",
            full_path="liver",
            display_name="Печень",
            data_type="object",
            is_leaf=False,
        )
        self.session.add(parent)
        self.session.commit()

        child = FeatureNode(
            profile_id=self.default_profile.id,
            parent_id=parent.id,
            key_name="right_lobe",
            full_path="liver.right_lobe",
            display_name="Правая доля печени",
            data_type="integer",
            unit="мм",
            is_leaf=True,
            status="approved",
        )
        self.session.add(child)
        self.session.commit()

        self.assertIn("liver.right_lobe", repr(child))
        self.assertEqual(child.parent.key_name, "liver")
        self.assertEqual(len(parent.children), 1)
        self.assertEqual(parent.children[0].id, child.id)

    def test_feature_alias_relationship(self):
        """Проверка модели синонимов FeatureAlias."""
        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="heart_rate",
            full_path="vitals.heart_rate",
            display_name="ЧСС",
            data_type="integer",
            unit="уд/мин",
            is_leaf=True,
        )
        self.session.add(node)
        self.session.commit()

        alias1 = FeatureAlias(feature_id=node.id, alias_text="пульс")
        alias2 = FeatureAlias(feature_id=node.id, alias_text="частота сердечных сокращений")
        self.session.add_all([alias1, alias2])
        self.session.commit()

        self.assertIn("пульс", repr(alias1))
        self.assertEqual(len(node.aliases), 2)
        self.assertEqual(alias1.feature.full_path, "vitals.heart_rate")

    def test_document_and_values_and_cache(self):
        """Проверка создания документа, фактов (DocumentValue) и кэша JSON."""
        doc = Document(
            profile_id=self.default_profile.id,
            patient_id="PAT-001",
            patient_name="Иванов И.И.",
            consultation_date=datetime(2025, 5, 12, 10, 30),
            raw_text="Печень: правая доля 140 мм. Заключение: норма.",
            administrative={"doctor": "Петров", "age": 45},
        )
        self.session.add(doc)
        self.session.commit()

        self.assertIn("PAT-001", repr(doc))
        self.assertEqual(doc.patient_name, "Иванов И.И.")

        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="right_lobe",
            full_path="liver.right_lobe",
            display_name="Правая доля печени",
            data_type="integer",
            is_leaf=True,
        )
        self.session.add(node)
        self.session.commit()

        val = DocumentValue(
            document_id=doc.id,
            feature_id=node.id,
            value_numeric=140.0,
            value_text=None,
            value_json=None,
            confidence=0.98,
            raw_context="Печень: правая доля 140 мм",
            meta_data={"source": "test"},
        )
        self.session.add(val)

        cache = DocumentJsonCache(
            document_id=doc.id,
            compiled_json={"liver": {"right_lobe": 140}, "patient": "Иванов И.И."},
        )
        self.session.add(cache)
        self.session.commit()

        self.assertIn(str(doc.id), repr(val))
        self.assertIn(str(doc.id), repr(cache))
        self.assertEqual(len(doc.values), 1)
        self.assertEqual(doc.values[0].value_numeric, 140.0)
        self.assertEqual(doc.json_cache.compiled_json["liver"]["right_lobe"], 140)

    def test_legacy_models(self):
        """Проверка моделей обратной совместимости: Parameter, ParameterSynonym, PatientRecord."""
        param = Parameter(
            canonical_name="blood_pressure",
            description="Артериальное давление",
        )
        self.session.add(param)
        self.session.commit()

        syn = ParameterSynonym(parameter_id=param.id, synonym_text="АД")
        self.session.add(syn)
        self.session.commit()

        rec = PatientRecord(
            patient_id="P-99",
            patient_name="Сидоров",
            parameter_id=param.id,
            raw_key="АД",
            raw_value="120/80",
        )
        self.session.add(rec)
        self.session.commit()

        self.assertEqual(len(param.synonyms), 1)
        self.assertEqual(param.synonyms[0].synonym_text, "АД")
        self.assertEqual(rec.parameter.canonical_name, "blood_pressure")


if __name__ == "__main__":
    unittest.main()
