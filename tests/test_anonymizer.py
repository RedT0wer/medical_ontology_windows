"""Модульные тесты для модуля деидентификации и анонимизации данных (152-ФЗ / HIPAA)."""

import unittest
from src.anonymizer import DataAnonymizer, anonymize_administrative, anonymize_text
from src.extractor import AdministrativeInfo, ExtractedDocument, ExtractedEntity
from src.pipeline import QuestionnaireProcessor
from tests.test_base import BaseTestCase, DummyEmbeddingProvider


class TestAnonymizer(unittest.TestCase):
    """Тестирование правил маскирования персональных данных."""

    def setUp(self):
        self.anonymizer = DataAnonymizer(default_pseudonym_prefix="PATIENT_ANON")

    def test_anonymize_fio(self):
        text = "Пациент: Иванов Иван Иванович, 45 лет. Жалобы на боли в животе."
        masked, audit = self.anonymizer.anonymize_text(text, pseudonym="PAT_999")
        self.assertIn("[PAT_999]", masked)
        self.assertNotIn("Иванов Иван Иванович", masked)
        self.assertIn("fio", audit["audit"])

    def test_anonymize_contacts(self):
        text = "Тел: +7 (999) 123-45-67, email: test.patient@example.com, доп: 8-800-555-35-35"
        masked, audit = self.anonymizer.anonymize_text(text)
        self.assertNotIn("+7 (999) 123-45-67", masked)
        self.assertNotIn("test.patient@example.com", masked)
        self.assertNotIn("8-800-555-35-35", masked)
        self.assertIn("[ТЕЛЕФОН_СКРЫТ]", masked)
        self.assertIn("[EMAIL_СКРЫТ]", masked)
        self.assertEqual(len(audit["audit"]["phone"]), 2)
        self.assertEqual(len(audit["audit"]["email"]), 1)

    def test_anonymize_documents_snils_passport_polis(self):
        text = (
            "СНИЛС: 123-456-789 00. Паспорт РФ: 45 12 123456. "
            "Полис ОМС: 1234567890123456."
        )
        masked, audit = self.anonymizer.anonymize_text(text)
        self.assertNotIn("123-456-789 00", masked)
        self.assertNotIn("45 12 123456", masked)
        self.assertNotIn("1234567890123456", masked)
        self.assertIn("[СНИЛС_СКРЫТ]", masked)
        self.assertIn("[ПАСПОРТ_СКРЫТ]", masked)
        self.assertIn("[ПОЛИС_СКРЫТ]", masked)

    def test_anonymize_address_and_birthdate(self):
        text = "Дата рождения: 15.06.1985. Проживание: г. Москва, ул. Ленина, д. 5, кв. 12."
        masked, audit = self.anonymizer.anonymize_text(text)
        self.assertNotIn("15.06.1985", masked)
        self.assertIn("[ГОД_РОЖДЕНИЯ_1985]", masked)
        self.assertNotIn("ул. Ленина", masked)
        self.assertIn("[АДРЕС_СКРЫТ]", masked)

    def test_anonymize_administrative_block(self):
        admin = {
            "patient": "Сидоров Петр Алексеевич",
            "date_of_birth": "1990-05-12",
            "age": 34,
            "address": "г. Казань, ул. Баумана, д. 1",
            "institution": "Городская больница №1",
            "doctor": "Петров В.В.",
        }
        masked = anonymize_administrative(admin, pseudonym="PAT_MASKED_01")
        self.assertEqual(masked["patient"], "PAT_MASKED_01")
        self.assertEqual(masked["date_of_birth"], "1990-XX-XX")
        self.assertEqual(masked["address"], "[АДРЕС_СКРЫТ]")
        self.assertEqual(masked["institution"], "Городская больница №1")
        self.assertEqual(masked["doctor"], "Петров В.В.")


class TestPipelineAnonymization(BaseTestCase):
    """Интеграционное тестирование деидентификации в пайплайне обработки."""

    def test_pipeline_with_anonymization(self):
        processor = QuestionnaireProcessor(embedding_provider=self.embedding_provider)
        # Мокаем извлечение сущностей
        mock_doc = ExtractedDocument(
            administrative=AdministrativeInfo(
                patient="Смирнов Алексей Владимирович",
                date_of_birth="1980-04-10",
                age=44,
                consultation_date="2024-01-15",
                doctor="Врач УЗД Тестовый",
                specialization="Ультразвуковая диагностика",
                institution="Клиника Тест",
                address="г. Самара, ул. Мира, д. 10",
            ),
            findings=[
                ExtractedEntity(
                    raw_key="Печень",
                    raw_value="не увеличена",
                    path_hint="liver",
                    category="organ",
                    data_type="string",
                )
            ],
            conclusion=["Эхографические признаки нормы"],
        )
        processor.extractor.extract_document = lambda text, patient_id: mock_doc

        raw_text = (
            "Пациент: Смирнов Алексей Владимирович, тел: +7-900-111-22-33\n"
            "Дата рождения: 10.04.1980, СНИЛС: 111-222-333 44\n"
            "Адрес: г. Самара, ул. Мира, д. 10\n"
            "Печень: не увеличена. Заключение: Эхографические признаки нормы."
        )

        res = processor.process_questionnaire(
            text=raw_text,
            patient_id="ANON-PAT-001",
            session=self.session,
            save_json=False,
            anonymize=True,
        )

        self.assertTrue(res["anonymized"])
        self.assertIn("anonymization_audit", res)
        self.assertEqual(res["patient_id"], "ANON-PAT-001")
        self.assertEqual(res["compiled_json"]["patient"], "ANON-PAT-001")
        self.assertEqual(res["compiled_json"]["address"], "[АДРЕС_СКРЫТ]")
        self.assertEqual(res["compiled_json"]["date_of_birth"], "1980-XX-XX")


if __name__ == "__main__":
    unittest.main()
