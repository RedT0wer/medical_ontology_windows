"""Тесты для модуля экспорта датасета признаков в CSV / Parquet src/export.py."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from tests.test_base import BaseTestCase
from src.export import (
    build_dataset_dataframe,
    clean_column_for_export,
    export_dataset,
    extract_document_values_map,
    is_null_value,
    load_feature_config,
    serialize_cell_value,
)
from src.models import (
    Document,
    DocumentJsonCache,
    DocumentValue,
    FeatureNode,
)


class TestExport(BaseTestCase):
    """Проверка загрузки конфигураций, сериализации ячеек и генерации CSV/Parquet датасетов."""

    def test_load_feature_config_yaml_dict(self):
        """Проверка загрузки конфигурации из YAML со словарем features."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump({"features": ["vitals.temperature", "liver.size"], "output": "out.csv"}, f)
            cfg_path = f.name

        try:
            cfg = load_feature_config(cfg_path)
            self.assertEqual(cfg["features"], ["vitals.temperature", "liver.size"])
            self.assertEqual(cfg["output"], "out.csv")
        finally:
            os.remove(cfg_path)

    def test_load_feature_config_yaml_list(self):
        """Проверка загрузки конфигурации из YAML со списком верхнего уровня."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(["vitals.heart_rate", "respiratory_system.chest"], f)
            cfg_path = f.name

        try:
            cfg = load_feature_config(cfg_path)
            self.assertEqual(cfg["features"], ["vitals.heart_rate", "respiratory_system.chest"])
        finally:
            os.remove(cfg_path)

    def test_load_feature_config_errors(self):
        """Проверка исключений при некорректных файлах конфигурации."""
        with self.assertRaises(FileNotFoundError):
            load_feature_config("non_existent_config.yaml")

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("")
            empty_path = f.name

        try:
            with self.assertRaises(ValueError):
                load_feature_config(empty_path)
        finally:
            os.remove(empty_path)

    def test_is_null_value(self):
        """Проверка функции распознавания пропущенных и пустых значений."""
        self.assertTrue(is_null_value(None))
        self.assertTrue(is_null_value(np.nan))
        self.assertTrue(is_null_value(""))
        self.assertTrue(is_null_value("   "))
        self.assertTrue(is_null_value("nan"))
        self.assertTrue(is_null_value("None"))
        self.assertTrue(is_null_value("null"))

        self.assertFalse(is_null_value(0))
        self.assertFalse(is_null_value(0.0))
        self.assertFalse(is_null_value(False))
        self.assertFalse(is_null_value("норма"))

    def test_serialize_cell_value(self):
        """Проверка сериализации значений для безопасного экспорта."""
        self.assertTrue(np.isnan(serialize_cell_value(None)))
        self.assertEqual(serialize_cell_value(120), 120)
        self.assertAlmostEqual(serialize_cell_value(36.6), 36.6)
        self.assertEqual(serialize_cell_value(["симптом 1", "симптом 2"]), "симптом 1; симптом 2")
        self.assertEqual(serialize_cell_value({"status": "ok"}), '{"status": "ok"}')

    def test_clean_column_for_export(self):
        """Проверка типизации колонок DataFrame."""
        # 1. Колонка чисел -> float64
        num_series = pd.Series([10.5, "12.0", np.nan])
        cleaned_num = clean_column_for_export(num_series, "feature_x")
        self.assertEqual(cleaned_num.dtype, "float64")
        self.assertAlmostEqual(cleaned_num.iloc[0], 10.5)

        # 2. Колонка целых чисел -> int64
        int_series = pd.Series([10, 20, 30])
        cleaned_int = clean_column_for_export(int_series, "feature_y")
        self.assertEqual(cleaned_int.dtype, "int64")

        # 3. Колонка строк -> object
        str_series = pd.Series(["норма", "патология", np.nan])
        cleaned_str = clean_column_for_export(str_series, "conclusion")
        self.assertEqual(cleaned_str.iloc[0], "норма")

    def test_extract_document_values_map(self):
        """Проверка извлечения карты значений из всех уровней анкеты."""
        doc = Document(
            profile_id=self.default_profile.id,
            patient_id="PAT-55",
            patient_name="Петров",
            raw_text="текст",
            administrative={"age": 60, "doctor": "Иванов"},
        )
        self.session.add(doc)
        self.session.commit()

        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="temperature",
            full_path="vitals.temperature",
            display_name="Температура",
            data_type="float",
            is_leaf=True,
        )
        self.session.add(node)
        self.session.commit()

        val = DocumentValue(
            document_id=doc.id,
            feature_id=node.id,
            value_numeric=37.2,
        )
        self.session.add(val)

        cache = DocumentJsonCache(
            document_id=doc.id,
            compiled_json={"liver": {"size": 140}},
        )
        self.session.add(cache)
        self.session.commit()

        vmap = extract_document_values_map(doc)
        self.assertEqual(vmap["patient_id"], "PAT-55")
        self.assertEqual(vmap["patient_name"], "Петров")
        self.assertEqual(vmap["age"], 60)
        self.assertAlmostEqual(vmap["vitals.temperature"], 37.2)
        self.assertEqual(vmap["temperature"], 37.2)
        self.assertEqual(vmap["liver.size"], 140)

    def test_build_dataset_dataframe_and_nan(self):
        """Проверка генерации датасета с заполнением отсутствующих признаков значением NaN."""
        # Создаем документ, у которого есть только vitals.temperature
        doc = Document(
            profile_id=self.default_profile.id,
            patient_id="P-01",
            raw_text="...",
        )
        self.session.add(doc)
        self.session.commit()

        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="temperature",
            full_path="vitals.temperature",
            display_name="Температура",
            data_type="float",
            is_leaf=True,
        )
        self.session.add(node)
        self.session.commit()

        val = DocumentValue(
            document_id=doc.id,
            feature_id=node.id,
            value_numeric=36.6,
        )
        self.session.add(val)
        self.session.commit()

        target_features = ["patient_id", "vitals.temperature", "liver.size"]
        df = build_dataset_dataframe(session=self.session, features=target_features)

        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), target_features)
        self.assertEqual(df["patient_id"].iloc[0], "P-01")
        self.assertAlmostEqual(df["vitals.temperature"].iloc[0], 36.6)
        # Отсутствующий у пациента признак liver.size должен стать NaN
        self.assertTrue(np.isnan(df["liver.size"].iloc[0]))

    def test_export_dataset_csv_and_parquet(self):
        """Проверка полного цикла экспорта в CSV и Parquet файлы."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = Path(tmp_dir) / "cfg.yaml"
            csv_out = Path(tmp_dir) / "output.csv"
            parquet_out = Path(tmp_dir) / "output.parquet"

            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.dump({"features": ["patient_id", "vitals.temperature"]}, f)

            with patch("src.export.get_db_session") as mock_get_sess:
                mock_get_sess.return_value.__enter__.return_value = self.session
                mock_get_sess.return_value.__exit__.return_value = None

                # 1. Экспорт в CSV
                res_csv = export_dataset(config=cfg_file, output=csv_out, format="csv")
                self.assertEqual(res_csv["status"], "ok")
                self.assertEqual(res_csv["format"], "csv")
                self.assertTrue(csv_out.exists())
                self.assertGreater(csv_out.stat().st_size, 0)

                # 2. Экспорт в Parquet (с моком для окружений без pyarrow)
                with patch.object(pd.DataFrame, "to_parquet") as mock_parquet:
                    def fake_parquet(path, **kwargs):
                        Path(path).write_bytes(b"PARQUET_MOCK_DATA")
                    mock_parquet.side_effect = fake_parquet

                    res_parquet = export_dataset(config=cfg_file, output=parquet_out, format="parquet")
                    self.assertEqual(res_parquet["status"], "ok")
                    self.assertEqual(res_parquet["format"], "parquet")
                    self.assertTrue(parquet_out.exists())
                    self.assertGreater(parquet_out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
