"""Модульные тесты для расширенной фильтрации при экспорте датасета."""

from datetime import datetime
import tempfile
from pathlib import Path
import unittest
import pandas as pd
from src.export import build_dataset_dataframe, export_dataset
from src.models import Document, DocumentJsonCache, DocumentValue, FeatureNode
from tests.test_base import BaseTestCase


class TestExportFilters(BaseTestCase):
    """Тестирование параметров фильтрации экспорта анкет."""

    def setUp(self):
        super().setUp()

        # Создаем узлы признаков
        self.node_liver = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="liver_size",
            full_path="liver.size",
            display_name="Размер печени",
            data_type="float",
            unit="мм",
        )
        self.node_conclusion = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="conclusion",
            full_path="conclusion",
            display_name="Заключение",
            data_type="array",
        )
        self.session.add_all([self.node_liver, self.node_conclusion])
        self.session.flush()

        # Документ 1: 2024-01-10, Врач Иванов, Клиника А, Диагноз: Гепатомегалия
        self.doc1 = Document(
            profile_id=self.default_profile.id,
            patient_id="PAT_01",
            patient_name="Пациент 1",
            consultation_date=datetime(2024, 1, 10, 10, 0),
            raw_text="Печень увеличена. Заключение: Гепатомегалия",
            administrative={"doctor": "Иванов И.И.", "institution": "Городская больница №1"},
        )
        self.session.add(self.doc1)
        self.session.flush()
        self.session.add_all([
            DocumentValue(document_id=self.doc1.id, feature_id=self.node_liver.id, value_numeric=145.0),
            DocumentValue(document_id=self.doc1.id, feature_id=self.node_conclusion.id, value_json=["Гепатомегалия"]),
            DocumentJsonCache(
                document_id=self.doc1.id,
                compiled_json={
                    "patient": "PAT_01",
                    "doctor": "Иванов И.И.",
                    "institution": "Городская больница №1",
                    "conclusion": ["Гепатомегалия"],
                    "liver": {"size": 145.0},
                },
            ),
        ])

        # Документ 2: 2024-02-15, Врач Петров, Клиника Б, Диагноз: Хронический холецистит
        self.doc2 = Document(
            profile_id=self.default_profile.id,
            patient_id="PAT_02",
            patient_name="Пациент 2",
            consultation_date=datetime(2024, 2, 15, 14, 30),
            raw_text="Желчный пузырь с утолщенной стенкой. Заключение: Хронический холецистит",
            administrative={"doctor": "Петров П.П.", "institution": "Клиника Эксперт"},
        )
        self.session.add(self.doc2)
        self.session.flush()
        self.session.add_all([
            DocumentValue(document_id=self.doc2.id, feature_id=self.node_liver.id, value_numeric=120.0),
            DocumentValue(document_id=self.doc2.id, feature_id=self.node_conclusion.id, value_json=["Хронический холецистит"]),
            DocumentJsonCache(
                document_id=self.doc2.id,
                compiled_json={
                    "patient": "PAT_02",
                    "doctor": "Петров П.П.",
                    "institution": "Клиника Эксперт",
                    "conclusion": ["Хронический холецистит"],
                    "liver": {"size": 120.0},
                },
            ),
        ])

        # Документ 3: 2024-03-20, Врач Иванов, Клиника А, Диагноз: Без патологии
        self.doc3 = Document(
            profile_id=self.default_profile.id,
            patient_id="PAT_03",
            patient_name="Пациент 3",
            consultation_date=datetime(2024, 3, 20, 9, 15),
            raw_text="Эхоструктура однородная. Заключение: Эхопризнаков патологии не выявлено",
            administrative={"doctor": "Иванов И.И.", "institution": "Городская больница №1"},
        )
        self.session.add(self.doc3)
        self.session.flush()
        self.session.add_all([
            DocumentValue(document_id=self.doc3.id, feature_id=self.node_liver.id, value_numeric=110.0),
            DocumentValue(document_id=self.doc3.id, feature_id=self.node_conclusion.id, value_json=["Без патологии"]),
            DocumentJsonCache(
                document_id=self.doc3.id,
                compiled_json={
                    "patient": "PAT_03",
                    "doctor": "Иванов И.И.",
                    "institution": "Городская больница №1",
                    "conclusion": ["Без патологии"],
                    "liver": {"size": 110.0},
                },
            ),
        ])
        self.session.commit()

    def test_filter_by_date_range(self):
        features = ["patient_id", "liver.size"]
        # От 2024-02-01 до 2024-03-31 -> должны попасть doc2 и doc3
        df = build_dataset_dataframe(
            session=self.session,
            features=features,
            from_date="2024-02-01",
            to_date="2024-03-31",
        )
        self.assertEqual(len(df), 2)
        patients = set(df["patient_id"])
        self.assertEqual(patients, {"PAT_02", "PAT_03"})

    def test_filter_by_diagnosis(self):
        features = ["patient_id", "liver.size"]
        # Ищем по подстроке диагноза 'холецистит'
        df = build_dataset_dataframe(
            session=self.session,
            features=features,
            filter_diagnosis="холецистит",
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["patient_id"], "PAT_02")

    def test_filter_by_doctor(self):
        features = ["patient_id", "doctor"]
        # Фильтр по врачу 'Иванов' -> doc1 и doc3
        df = build_dataset_dataframe(
            session=self.session,
            features=features,
            filter_doctor="Иванов",
        )
        self.assertEqual(len(df), 2)
        patients = set(df["patient_id"])
        self.assertEqual(patients, {"PAT_01", "PAT_03"})

    def test_filter_by_institution(self):
        features = ["patient_id", "institution"]
        # Фильтр по учреждению 'Эксперт' -> doc2
        df = build_dataset_dataframe(
            session=self.session,
            features=features,
            filter_institution="Эксперт",
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["patient_id"], "PAT_02")


if __name__ == "__main__":
    unittest.main()

