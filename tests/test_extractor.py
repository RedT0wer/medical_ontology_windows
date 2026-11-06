"""Тесты для модуля извлечения клинических сущностей src/extractor.py."""

import json
import unittest
from unittest.mock import MagicMock, patch
from src.extractor import (
    AdministrativeInfo,
    EntityExtractor,
    ExtractedDocument,
    ExtractedEntity,
)


class TestExtractor(unittest.TestCase):
    """Проверка очистки JSON, определения типов и парсинга структуры протоколов."""

    def setUp(self):
        self.extractor = EntityExtractor()

    def test_administrative_info_defaults(self):
        """Проверка значений по умолчанию в AdministrativeInfo."""
        admin = AdministrativeInfo()
        self.assertEqual(admin.institution, "---")
        self.assertEqual(admin.patient, "---")
        self.assertIsNone(admin.age)
        self.assertEqual(admin.examination, "лист_консультативного_осмотра")
        self.assertEqual(admin.specialization, "терапия")

    def test_clean_json_response_markdown(self):
        """Проверка очистки от markdown-оберток (```json ... ```)."""
        raw = '```json\n{"institution": "ГКБ 1", "age": 60}\n```'
        cleaned = self.extractor._clean_json_response(raw)
        self.assertEqual(cleaned, {"institution": "ГКБ 1", "age": 60})

    def test_clean_json_response_unclosed_braces(self):
        """Проверка автодополнения пропущенных закрывающих фигурных скобок."""
        raw = '{"institution": "ГКБ 1", "age": 60'
        cleaned = self.extractor._clean_json_response(raw)
        self.assertEqual(cleaned.get("institution"), "ГКБ 1")
        self.assertEqual(cleaned.get("age"), 60)

    def test_clean_json_response_wrapped_dict(self):
        """Проверка распаковки вложенных словарей верхнего уровня (data / protocol / response)."""
        raw = '{"protocol": {"institution": "ГКБ 2", "doctor": "Иванов"}}'
        cleaned = self.extractor._clean_json_response(raw)
        self.assertEqual(cleaned.get("institution"), "ГКБ 2")
        self.assertEqual(cleaned.get("doctor"), "Иванов")

    def test_find_context_snippet(self):
        """Проверка поиска контекстной цитаты по ключевым словам."""
        text = "Общее состояние удовлетворительное.\nПечень увеличена, правая доля 155 мм.\nСелезенка без особенностей."
        snippet = self.extractor._find_context_snippet(text, ["155", "правая доля"])
        self.assertEqual(snippet, "Печень увеличена, правая доля 155 мм.")

    def test_infer_type_and_unit(self):
        """Проверка вывода типов данных и единиц измерения по ключу и значению."""
        # 1. Двухмерный габарит
        t, u = self.extractor._infer_type_and_unit("size", "65 х 28 мм")
        self.assertEqual(t, "dimension_2d")
        self.assertEqual(u, "мм")

        # 2. Целое число (толщина/размер органа)
        t, u = self.extractor._infer_type_and_unit("wall", 3)
        self.assertEqual(t, "integer")
        self.assertEqual(u, "мм")

        # 3. Пульс / ЧСС
        t, u = self.extractor._infer_type_and_unit("heart_rate", 72)
        self.assertEqual(t, "integer")
        self.assertEqual(u, "уд/мин")

        # 4. Температура
        t, u = self.extractor._infer_type_and_unit("temperature", 36.6)
        self.assertEqual(t, "float")
        self.assertEqual(u, "°C")

        # 5. Массив
        t, u = self.extractor._infer_type_and_unit("symptoms", ["кашель", "одышка"])
        self.assertEqual(t, "array")

        # 6. Булево
        t, u = self.extractor._infer_type_and_unit("is_normal", True)
        self.assertEqual(t, "boolean")

    def test_flatten_clinical_tree(self):
        """Проверка рекурсивного разворачивания дерева органов в атомарные признаки."""
        tree = {
            "institution": "Клиника",  # Должно быть проигнорировано (admin key)
            "conclusion": ["холецистит"],  # Должно быть проигнорировано
            "liver": {
                "right_lobe": 150,
                "gallbladder": {
                    "wall": 3.5,
                    "concrements": "нет",
                },
            },
        }
        raw_text = "Печень: правая доля 150 мм. Желчный пузырь: стенка 3.5 мм, конкрементов нет."

        findings = self.extractor._flatten_clinical_tree(tree, raw_text)
        paths = {f.path_hint: f for f in findings}

        self.assertIn("liver.right_lobe", paths)
        self.assertEqual(paths["liver.right_lobe"].raw_value, 150)
        self.assertEqual(paths["liver.right_lobe"].data_type, "integer")

        self.assertIn("liver.gallbladder.wall", paths)
        self.assertEqual(paths["liver.gallbladder.wall"].raw_value, 3.5)
        self.assertEqual(paths["liver.gallbladder.wall"].data_type, "float")

        self.assertIn("liver.gallbladder.concrements", paths)
        self.assertEqual(paths["liver.gallbladder.concrements"].raw_value, "нет")
        self.assertEqual(paths["liver.gallbladder.concrements"].data_type, "string")

    def test_parse_administrative(self):
        """Проверка парсинга и валидации административной шапки."""
        data = {
            "institution": "ГКБ №1 им. Пирогова",
            "patient": "Сидоров В.П.",
            "age": "52",
            "date_of_birth": "1973-04-15",
            "consultation_date": "2025-05-10T14:30",
            "examination": "ультразвуковое_исследование_органов_брюшной_полости",
            "doctor": "Смирнов А.А.",
            "specialization": "ультразвуковая_диагностика",
        }
        admin = self.extractor._parse_administrative(data, patient_id="SID-01")
        self.assertEqual(admin.institution, "ГКБ №1 им. Пирогова")
        self.assertEqual(admin.patient, "Сидоров В.П.")
        self.assertEqual(admin.age, 52)
        self.assertEqual(admin.date_of_birth, "1973-04-15")

    def test_extract_document_mock_llm(self):
        """Проверка полного вызова extract_document с замоканным ответом модели."""
        mock_response = json.dumps({
            "institution": "Центральная больница",
            "patient": "Кузнецов",
            "age": 40,
            "conclusion": ["хронический гастрит"],
            "vitals": {
                "heart_rate": 68,
            },
        })

        with patch.object(self.extractor, "_call_ollama", return_value=mock_response):
            with patch("src.extractor.settings.LLM_PROVIDER", "ollama"):
                doc = self.extractor.extract_document(
                    text="Осмотр терапевта. ЧСС 68 уд/мин. Диагноз: хронический гастрит.",
                    patient_id="PAT-10",
                )

        self.assertIsInstance(doc, ExtractedDocument)
        self.assertEqual(doc.administrative.patient, "Кузнецов")
        self.assertEqual(doc.administrative.age, 40)
        self.assertEqual(doc.conclusion, ["хронический гастрит"])
        self.assertEqual(len(doc.findings), 1)
        self.assertEqual(doc.findings[0].path_hint, "vitals.heart_rate")
        self.assertEqual(doc.findings[0].raw_value, 68)


if __name__ == "__main__":
    unittest.main()
