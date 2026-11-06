"""Модульные тесты для модуля деидентификации и анонимизации данных (152-ФЗ / HIPAA).

Включает глубокое тестирование граничных случаев:
- Все варианты написания ФИО (смешанный регистр, ВЕРХНИЙ РЕГИСТР, инициалы со слитным и раздельным написанием).
- Множественные префиксы пациентов (Пациент, Пациентка, Больной, Больная, Обследуемый, ФИО, Ф.И.О.).
- Маскирование последующих упоминаний фамилии и инициалов в теле анамнеза.
- Все форматы контактных телефонов (+7, 8, скобки, пробелы, дефисы, точки, слитное написание).
- Все форматы полисов ОМС/ДМС (16 цифр слитно, 4x4 через пробелы, через дефисы).
- Все форматы СНИЛС (дефисы, пробелы, точки, 11 цифр подряд).
- Все форматы дат рождения (точная дата, сокращение д.р., род., родился, г.р.).
- Границы адресов проживания с защитой от захвата следующих секций (Жалобы, Диагноз и др.).
- Комплексный протокол реального вида с множественными ПДн.
"""

import unittest
from src.anonymizer import DataAnonymizer, anonymize_administrative, anonymize_text
from src.extractor import AdministrativeInfo, ExtractedDocument, ExtractedEntity
from src.pipeline import QuestionnaireProcessor
from tests.test_base import BaseTestCase


class TestAnonymizer(unittest.TestCase):
    """Тестирование правил маскирования персональных данных."""

    def setUp(self):
        self.anonymizer = DataAnonymizer(default_pseudonym_prefix="PATIENT_ANON")

    def test_anonymize_fio_standard(self):
        """Стандартное написание ФИО со смешанным регистром."""
        text = "Пациент: Иванов Иван Иванович, 45 лет. Жалобы на боли в животе."
        masked, audit = self.anonymizer.anonymize_text(text, pseudonym="PAT_999")
        self.assertIn("[PAT_999]", masked)
        self.assertNotIn("Иванов Иван Иванович", masked)
        self.assertIn("fio", audit["audit"])

    def test_anonymize_fio_uppercase(self):
        """ФИО заглавными буквами (частый формат в медицинских информационных системах)."""
        text = "Пациентка: ПЕТРОВА АННА СЕРГЕЕВНА, 1982 г.р."
        masked, audit = self.anonymizer.anonymize_text(text)
        self.assertIn("[PATIENT_ANON]", masked)
        self.assertNotIn("ПЕТРОВА АННА СЕРГЕЕВНА", masked)
        self.assertEqual(len(audit["audit"]["fio"]), 1)

    def test_anonymize_fio_initials_compact_and_spaced(self):
        """ФИО с инициалами (слитно 'И.И.' и через пробел 'И. И.')."""
        t1 = "Больной: Сидоров А.В., 50 лет."
        m1, _ = self.anonymizer.anonymize_text(t1)
        self.assertNotIn("Сидоров А.В.", m1)
        self.assertIn("[PATIENT_ANON]", m1)

        t2 = "Ф.И.О.: КУЗНЕЦОВ П. П., диагноз: гастрит."
        m2, _ = self.anonymizer.anonymize_text(t2)
        self.assertNotIn("КУЗНЕЦОВ П. П.", m2)
        self.assertIn("[PATIENT_ANON]", m2)

    def test_anonymize_fio_alternative_prefixes(self):
        """Различные префиксы: Обследуемый, Исследуемый, Больная, ФИО."""
        prefixes = [
            "Обследуемый: Васильев Дмитрий",
            "Исследуемый: Козлов Иван",
            "Больная: Николаева Мария",
            "ФИО: Григорьев Степан Александрович",
        ]
        for item in prefixes:
            masked, audit = self.anonymizer.anonymize_text(item)
            self.assertIn("[PATIENT_ANON]", masked)
            self.assertEqual(len(audit["audit"]["fio"]), 1)

    def test_anonymize_fio_subsequent_mentions_in_body(self):
        """Маскирование повторных упоминаний пациента в теле протокола."""
        text = (
            "Пациент: Смирнов Алексей Владимирович.\n"
            "В процессе осмотра Смирнов Алексей Владимирович жалоб не предъявлял.\n"
            "Результаты обследования переданы: Смирнов А.В. лично в руки.\n"
            "Повторный визит: Смирнов А. В."
        )
        masked, audit = self.anonymizer.anonymize_text(text, pseudonym="PAT_MASK")
        self.assertNotIn("Смирнов Алексей Владимирович", masked)
        self.assertNotIn("Смирнов А.В.", masked)
        self.assertNotIn("Смирнов А. В.", masked)
        # Все упоминания заменены на плейсхолдер
        self.assertEqual(masked.count("[PAT_MASK]"), 4)

    def test_anonymize_contacts_all_phone_formats(self):
        """Все варианты записи телефонных номеров."""
        phones = [
            "+7 (999) 123-45-67",
            "8-999-123-45-67",
            "+79991234567",
            "89991234567",
            "+7.999.123.45.67",
            "8 (495) 123 45 67",
        ]
        text = "Телефоны пациента: " + ", ".join(phones)
        masked, audit = self.anonymizer.anonymize_text(text)
        for ph in phones:
            self.assertNotIn(ph, masked)
        self.assertEqual(len(audit["audit"]["phone"]), len(phones))

    def test_anonymize_email_formats(self):
        """Маскирование адресов электронной почты."""
        text = "Контакты: doc.patient+test@clinic-service.org, ivanov@mail.ru"
        masked, audit = self.anonymizer.anonymize_text(text)
        self.assertNotIn("doc.patient+test@clinic-service.org", masked)
        self.assertNotIn("ivanov@mail.ru", masked)
        self.assertEqual(len(audit["audit"]["email"]), 2)

    def test_anonymize_snils_all_formats(self):
        """Различные форматы записи СНИЛС."""
        snils_list = [
            "123-456-789 00",
            "123-456-789-01",
            "123 456 789 02",
            "123.456.789.03",
            "СНИЛС: 12345678904",
        ]
        text = " ".join(snils_list)
        masked, audit = self.anonymizer.anonymize_text(text)
        for s in snils_list:
            clean_s = s.replace("СНИЛС: ", "")
            self.assertNotIn(clean_s, masked)
        self.assertEqual(len(audit["audit"]["snils"]), len(snils_list))

    def test_anonymize_polis_all_formats(self):
        """Полисы ОМС/ДМС (16 цифр слитно, 4x4 с пробелами и дефисами)."""
        polis_list = [
            "1234567890123456",
            "1111 2222 3333 4444",
            "5555-6666-7777-8888",
        ]
        text = "ОМС: 1234567890123456, ДМС: 1111 2222 3333 4444, полис: 5555-6666-7777-8888"
        masked, audit = self.anonymizer.anonymize_text(text)
        for p in polis_list:
            self.assertNotIn(p, masked)
        self.assertEqual(len(audit["audit"]["polis"]), len(polis_list))

    def test_anonymize_passports(self):
        """Серии и номера паспортов РФ."""
        text = "Паспорт РФ: 45 12 123456. Ранее выдан: 4512 654321."
        masked, audit = self.anonymizer.anonymize_text(text)
        self.assertNotIn("45 12 123456", masked)
        self.assertNotIn("4512 654321", masked)
        self.assertIn("[ПАСПОРТ_СКРЫТ]", masked)
        self.assertEqual(len(audit["audit"]["passport"]), 2)

    def test_anonymize_birthdate_formats(self):
        """Различные формулировки даты рождения."""
        cases = [
            "Дата рождения: 15.06.1985",
            "д.р.: 12/04/1990",
            "Родился: 23-11-1975",
            "д.р. 1988 г.",
        ]
        for c in cases:
            masked, audit = self.anonymizer.anonymize_text(c)
            self.assertTrue(
                "[ГОД_РОЖДЕНИЯ_" in masked or "[ДАТА_РОЖДЕНИЯ_СКРЫТА]" in masked
            )
            self.assertEqual(len(audit["audit"]["birth_date"]), 1)

    def test_anonymize_address_with_trailing_sections(self):
        """Адрес с последующими секциями без переноса строки."""
        text = (
            "Адрес проживания: г. Москва, ул. Тверская, д. 10, кв. 25. "
            "Жалобы: боли в животе."
        )
        masked, audit = self.anonymizer.anonymize_text(text)
        self.assertNotIn("Тверская", masked)
        self.assertIn("[АДРЕС_СКРЫТ]", masked)
        self.assertIn("Жалобы: боли в животе.", masked)

    def test_anonymize_full_complex_protocol(self):
        """Комплексный протокол реального вида со всеми типами ПДн."""
        protocol = (
            "ООО 'Медицинский Центр Здоровье'\n"
            "ЛИСТ КОНСУЛЬТАТИВНОГО ОСМОТРА\n"
            "Пациент: Морозов Константин Игоревич, 48 лет\n"
            "Дата рождения: 24.12.1976. Тел: +7 (916) 555-44-33, email: morozov@mail.ru\n"
            "Паспорт: 46 08 987654, СНИЛС: 222-333-444 55, полис ОМС: 9876543210123456\n"
            "Адрес: Московская область, г. Красногорск, ул. Ленина, д. 42, кв. 10\n"
            "Врач: Смирнова О.В.\n"
            "Жалобы: периодические боли в правом подреберье.\n"
            "Анамнез: Пациент Морозов Константин Игоревич наблюдается с 2021 года.\n"
            "Заключение: Эхопризнаки хронического калькулезного холецистита.\n"
            "Рекомендации переданы: Морозов К.И."
        )
        masked, audit = self.anonymizer.anonymize_text(protocol, pseudonym="PAT_ANON_007")

        # Проверка отсутствия персональных данных
        self.assertNotIn("Морозов Константин Игоревич", masked)
        self.assertNotIn("Морозов К.И.", masked)
        self.assertNotIn("+7 (916) 555-44-33", masked)
        self.assertNotIn("morozov@mail.ru", masked)
        self.assertNotIn("46 08 987654", masked)
        self.assertNotIn("222-333-444 55", masked)
        self.assertNotIn("9876543210123456", masked)
        self.assertNotIn("Красногорск", masked)
        self.assertNotIn("24.12.1976", masked)

        # Проверка сохранности клинических данных
        self.assertIn("ООО 'Медицинский Центр Здоровье'", masked)
        self.assertIn("Врач: Смирнова О.В.", masked)
        self.assertIn("боли в правом подреберье", masked)
        self.assertIn("хронического калькулезного холецистита", masked)
        self.assertIn("[ГОД_РОЖДЕНИЯ_1976]", masked)

        # Проверка счетчика скрытых элементов
        self.assertGreaterEqual(audit["masked_items"], 8)

    def test_anonymize_administrative_block_comprehensive(self):
        """Псевдонимизация административного словаря."""
        admin = {
            "patient": "Кузнецов Артем Николаевич",
            "date_of_birth": "1975-08-20",
            "age": 49,
            "address": "г. Екатеринбург, ул. Малышева, д. 101",
            "institution": "Городская больница №40",
            "doctor": "Врач Васильев К.А.",
            "specialization": "Терапия",
        }
        masked = anonymize_administrative(admin, pseudonym="PAT_MASKED_99")
        self.assertEqual(masked["patient"], "PAT_MASKED_99")
        self.assertEqual(masked["date_of_birth"], "1975-XX-XX")
        self.assertEqual(masked["address"], "[АДРЕС_СКРЫТ]")
        self.assertEqual(masked["institution"], "Городская больница №40")
        self.assertEqual(masked["doctor"], "Врач Васильев К.А.")
        self.assertTrue(masked["patient_original_masked"])


class TestPipelineAnonymization(BaseTestCase):
    """Интеграционное тестирование деидентификации в пайплайне обработки."""

    def test_pipeline_with_anonymization(self):
        processor = QuestionnaireProcessor(embedding_provider=self.embedding_provider)
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
