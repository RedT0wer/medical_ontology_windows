"""Тесты для пайплайна обработки медицинских протоколов src/pipeline.py."""

from datetime import datetime
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from tests.test_base import BaseTestCase
from src.extractor import AdministrativeInfo, ExtractedDocument, ExtractedEntity
from src.models import Document, DocumentValue, FeatureNode
from src.pipeline import (
    QuestionnaireProcessor,
    build_compiled_json,
    set_nested_dict_value,
)


class TestPipeline(BaseTestCase):
    """Проверка сборки JSON документов, разбора дат и полного цикла QuestionnaireProcessor."""

    def test_set_nested_dict_value(self):
        """Проверка рекурсивной установки значений в словаре по точечному пути."""
        target = {}
        set_nested_dict_value(target, ["liver", "gallbladder", "wall"], 3.5)
        self.assertEqual(target, {"liver": {"gallbladder": {"wall": 3.5}}})

        set_nested_dict_value(target, ["liver", "right_lobe"], 145)
        self.assertEqual(target["liver"]["right_lobe"], 145)
        self.assertEqual(target["liver"]["gallbladder"]["wall"], 3.5)

    def test_build_compiled_json(self):
        """Проверка сборки эталонного иерархического JSON документа."""
        admin = AdministrativeInfo(
            institution="Областная больница",
            patient="Смирнов",
            age=50,
            date_of_birth="1975-01-01",
            address="г. Москва",
            consultation_date="2025-05-15T10:00",
            examination="ультразвуковое_исследование_органов_брюшной_полости",
            doctor="Иванов",
            specialization="ультразвуковая_диагностика",
        )
        findings = [
            ("liver.right_lobe", "integer", 150),
            ("liver.gallbladder.wall", "float", 3.2),
            ("conclusion", "array", ["холецистит"]),  # Должно быть отфильтровано
        ]
        conclusions = ["хронический холецистит", "гепатомегалия"]

        compiled = build_compiled_json(admin, findings, conclusions)

        self.assertEqual(compiled["institution"], "Областная больница")
        self.assertEqual(compiled["patient"], "Смирнов")
        self.assertEqual(compiled["age"], 50)
        self.assertEqual(compiled["liver"]["right_lobe"], 150)
        self.assertAlmostEqual(compiled["liver"]["gallbladder"]["wall"], 3.2)
        self.assertEqual(compiled["conclusion"], conclusions)
        self.assertEqual(compiled["doctor"], "Иванов")

    def test_parse_consultation_date(self):
        """Проверка разбора дат различных форматов."""
        processor = QuestionnaireProcessor(embedding_provider=self.embedding_provider)

        d1 = processor._parse_consultation_date("2025-05-20T14:30")
        self.assertEqual(d1.year, 2025)
        self.assertEqual(d1.month, 5)
        self.assertEqual(d1.day, 20)
        self.assertEqual(d1.hour, 14)
        self.assertEqual(d1.minute, 30)

        d2 = processor._parse_consultation_date("15.06.2024")
        self.assertEqual(d2.year, 2024)
        self.assertEqual(d2.month, 6)
        self.assertEqual(d2.day, 15)

        # Fallback при пустой или некорректной дате -> текущее время
        d3 = processor._parse_consultation_date(None)
        self.assertIsInstance(d3, datetime)

    def test_process_questionnaire_full_flow(self):
        """Проверка полного сквозного цикла QuestionnaireProcessor с сохранением в БД."""
        mock_extractor = MagicMock()
        mock_extractor.extract_document.return_value = ExtractedDocument(
            administrative=AdministrativeInfo(
                institution="Городская поликлиника №5",
                patient="Семенов А.В.",
                age=62,
                consultation_date="2025-06-01T11:00",
                examination="лист_консультативного_осмотра",
                doctor="Ковалева",
                specialization="терапия",
            ),
            findings=[
                ExtractedEntity(
                    raw_key="right_lobe",
                    raw_value=148,
                    context="Правая доля печени 148 мм",
                    path_hint="liver.right_lobe",
                    data_type="integer",
                    unit="мм",
                ),
                ExtractedEntity(
                    raw_key="pulse",
                    raw_value=76,
                    context="Пульс 76 уд/мин",
                    path_hint="vitals.heart_rate",
                    data_type="integer",
                    unit="уд/мин",
                ),
            ],
            conclusion=["артериальная гипертензия"],
        )

        processor = QuestionnaireProcessor(
            extractor=mock_extractor,
            embedding_provider=self.embedding_provider,
            profile_code="us_abdomen",
        )

        result = processor.process_questionnaire(
            text="Текст медицинского протокола Семенова...",
            patient_id="PAT-SEM-01",
            session=self.session,
            save_json=False,
        )
        self.session.commit()

        self.assertEqual(result["patient_id"], "PAT-SEM-01")
        self.assertEqual(result["total_extracted"], 2)
        self.assertIsNotNone(result["document_id"])
        self.assertIn("liver", result["compiled_json"])
        self.assertEqual(result["compiled_json"]["liver"]["right_lobe"], 148)

        # Проверяем записи в базе данных
        doc = self.session.query(Document).filter_by(patient_id="PAT-SEM-01").first()
        self.assertIsNotNone(doc)
        self.assertEqual(doc.patient_name, "Семенов А.В.")
        self.assertEqual(len(doc.values), 3)  # 2 findings + 1 conclusion
        self.assertIsNotNone(doc.json_cache)

    def test_process_file(self):
        """Проверка обработки протокола из текстового файла."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write("Протокол УЗИ пациента. Печень увеличена.")
            tmp_path = f.name

        try:
            mock_extractor = MagicMock()
            mock_extractor.extract_document.return_value = ExtractedDocument(
                administrative=AdministrativeInfo(patient="Файловый Пациент"),
                findings=[],
                conclusion=[],
            )

            processor = QuestionnaireProcessor(
                extractor=mock_extractor,
                embedding_provider=self.embedding_provider,
                profile_code="us_abdomen",
            )

            res = processor.process_file(tmp_path, session=self.session, save_json=False)
            self.session.commit()
            self.assertEqual(res["patient_name"], "Файловый Пациент")

            # Проверка ошибки при отсутствии файла
            with self.assertRaises(FileNotFoundError):
                processor.process_file("non_existing_file_path.txt")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_get_ontology_summary(self):
        """Проверка формирования статистической сводки онтологии."""
        # Добавим узлы и документ
        node1 = FeatureNode(profile_id=self.default_profile.id, key_name="liver", full_path="liver", display_name="Печень", is_leaf=False)
        node2 = FeatureNode(profile_id=self.default_profile.id, key_name="size", full_path="liver.size", display_name="Размер печени", is_leaf=True)
        self.session.add_all([node1, node2])
        self.session.commit()

        processor = QuestionnaireProcessor(embedding_provider=self.embedding_provider)
        with patch("src.pipeline.get_db_session") as mock_sess:
            mock_sess.return_value.__enter__.return_value = self.session
            mock_sess.return_value.__exit__.return_value = None

            summary = processor.get_ontology_summary()
            self.assertEqual(summary["profile_code"], "us_abdomen")
            self.assertEqual(summary["total_features"], 2)
            self.assertEqual(summary["leaf_features"], 1)
            self.assertEqual(summary["branch_features"], 1)


if __name__ == "__main__":
    unittest.main()
