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

    def test_parse_flexible_date_all_formats(self):
        """Проверка парсинга всех поддерживаемых форматов дат и валидации ошибок."""
        from src.export import parse_flexible_date

        # ISO и стандартные даты
        dt1 = parse_flexible_date("2024-05-20")
        self.assertEqual((dt1.year, dt1.month, dt1.day), (2024, 5, 20))

        # Российский формат DD.MM.YYYY
        dt2 = parse_flexible_date("20.05.2024")
        self.assertEqual((dt2.year, dt2.month, dt2.day), (2024, 5, 20))

        # Дата с временем
        dt3 = parse_flexible_date("2024-05-20 14:30:15")
        self.assertEqual((dt3.hour, dt3.minute, dt3.second), (14, 30, 15))

        dt4 = parse_flexible_date("20.05.2024 14:30")
        self.assertEqual((dt4.hour, dt4.minute), (14, 30))

        # ISO 8601 с 'T'
        dt5 = parse_flexible_date("2024-05-20T14:30:15.123456")
        self.assertEqual(dt5.microsecond, 123456)

        # Передача уже готового объекта datetime
        now = datetime.now()
        self.assertEqual(parse_flexible_date(now), now)

        # Некорректный формат
        with self.assertRaises(ValueError):
            parse_flexible_date("не_является_датой")

    def test_filter_combined_all_criteria(self):
        """Проверка одновременного применения всех критериев фильтрации."""
        features = ["patient_id", "doctor", "institution", "liver.size"]
        # Ищем: Иванов + Городская больница №1 + Гепатомегалия + диапазон дат
        df = build_dataset_dataframe(
            session=self.session,
            features=features,
            from_date="2024-01-01",
            to_date="2024-01-31",
            filter_diagnosis="гепатомегалия",
            filter_doctor="Иванов",
            filter_institution="Городская больница №1",
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["patient_id"], "PAT_01")
        self.assertAlmostEqual(df.iloc[0]["liver.size"], 145.0)

    def test_filter_no_matches_empty_df(self):
        """Проверка возврата корректного пустого DataFrame при отсутствии совпадений."""
        features = ["patient_id", "doctor", "liver.size"]
        df = build_dataset_dataframe(
            session=self.session,
            features=features,
            from_date="2099-01-01",  # Далекое будущее
        )
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), features)

    def test_filter_diagnosis_case_insensitivity_and_sources(self):
        """Проверка регистронезависимости и поиска диагноза в разных источниках."""
        # 1. Верхний регистр в запросе
        df_upper = build_dataset_dataframe(
            session=self.session,
            features=["patient_id"],
            filter_diagnosis="ХОЛЕЦИСТИТ",
        )
        self.assertEqual(len(df_upper), 1)
        self.assertEqual(df_upper.iloc[0]["patient_id"], "PAT_02")

        # 2. Поиск диагноза из сырого текста документа
        doc_raw = Document(
            profile_id=self.default_profile.id,
            patient_id="PAT_RAW",
            raw_text="Пациент направлен с подозрением на аппендицит острый.",
            administrative={},
        )
        self.session.add(doc_raw)
        self.session.commit()

        df_raw = build_dataset_dataframe(
            session=self.session,
            features=["patient_id"],
            filter_diagnosis="аппендицит",
        )
        self.assertEqual(len(df_raw), 1)
        self.assertEqual(df_raw.iloc[0]["patient_id"], "PAT_RAW")

    def test_export_dataset_file_formats_with_filters(self):
        """Проверка сборки файла датасета с фильтрацией через export_dataset и CLI cmd_export."""
        import argparse
        import io
        import json
        from unittest.mock import patch
        import yaml
        import cli

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            cfg_path = tmp_p / "features.yaml"
            cfg_path.write_text(
                yaml.dump({
                    "features": ["patient_id", "doctor", "liver.size"],
                    "output": str(tmp_p / "dataset.csv"),
                    "format": "csv",
                }),
                encoding="utf-8",
            )

            # 1. Вызов export_dataset с фильтром по врачу
            with patch("src.export.get_db_session") as mock_db:
                mock_db.return_value.__enter__.return_value = self.session
                res = export_dataset(
                    config=str(cfg_path),
                    filter_doctor="Иванов",
                )
                self.assertEqual(res["status"], "ok")
                self.assertEqual(res["total_documents"], 2)
                self.assertTrue(Path(res["file"]).exists())

                # Читаем получившийся CSV и проверяем состав
                df_csv = pd.read_csv(res["file"])
                self.assertEqual(len(df_csv), 2)
                self.assertEqual(set(df_csv["patient_id"]), {"PAT_01", "PAT_03"})

            # 2. Интеграция с CLI cmd_export
            out_cli = tmp_p / "cli_dataset.csv"
            cli_args = argparse.Namespace(
                config=str(cfg_path),
                config_pos=None,
                output=str(out_cli),
                format="csv",
                from_date="2024-02-01",
                to_date="2024-02-28",
                filter_diagnosis=None,
                filter_doctor=None,
                filter_institution=None,
                json=True,
            )
            with patch("src.export.get_db_session") as mock_db:
                mock_db.return_value.__enter__.return_value = self.session
                with patch("sys.stdout", new=io.StringIO()) as fake_out:
                    cli.cmd_export(cli_args)
                    cli_res = json.loads(fake_out.getvalue())
                    self.assertEqual(cli_res["status"], "ok")
                    self.assertEqual(cli_res["total_documents"], 1)
                    self.assertTrue(out_cli.exists())


if __name__ == "__main__":
    unittest.main()

