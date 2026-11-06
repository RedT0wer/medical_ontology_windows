"""Модульные тесты для расширенных CLI команд (курация схемы, инкрементальный режим, dump/load)."""

import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import cli
from src.models import Document, DocumentValue, FeatureAlias, FeatureNode
from tests.test_base import BaseTestCase


class TestCLIExtended(BaseTestCase):
    """Тестирование новых команд CLI: курация онтологии, дамп/лоад схемы, прогресс-бар и инкрементальный режим."""

    def test_console_progress_bar(self):
        """Проверка работы ConsoleProgressBar и форматирования."""
        buf = io.StringIO()
        pbar = cli.ConsoleProgressBar(total=3, width=10, stream=buf)
        pbar.update("file1.pdf", status="success")
        pbar.update("file2.docx", status="skipped")
        pbar.update("file3.txt", status="error")
        pbar.finish()

        output = buf.getvalue()
        self.assertIn("Успешно: 1", output)
        self.assertIn("Пропуск: 1", output)
        self.assertIn("Ошибки: 1", output)
        self.assertIn("Завершено: 3/3", output)

    def test_process_directory_incremental_resume(self):
        """Проверка инкрементального режима (--resume): пропуск уже существующих документов."""
        # Создаем документ в базе данных с patient_id="doc1"
        doc = Document(
            profile_id=self.default_profile.id,
            patient_id="doc1",
            patient_name="doc1",
            raw_text="Существующий протокол",
            administrative={},
        )
        self.session.add(doc)
        self.session.commit()

        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_p = Path(tmp_dir)
            f1 = dir_p / "doc1.txt"
            f2 = dir_p / "doc2.txt"
            f1.write_text("Текст 1", encoding="utf-8")
            f2.write_text("Текст 2", encoding="utf-8")

            mock_processor = MagicMock()
            mock_processor.process_questionnaire.return_value = {
                "patient_id": "doc2",
                "total_extracted": 2,
                "new_features": 0,
                "compiled_json": {},
            }

            with patch("cli.get_db_session") as mock_db:
                mock_db.return_value.__enter__.return_value = self.session

                # Запускаем с resume=True
                results = cli.process_directory(
                    dir_path=dir_p,
                    processor=mock_processor,
                    resume=True,
                    json_mode=True,
                )

                self.assertEqual(len(results), 2)
                statuses = {r["patient_id"]: r["status"] for r in results}
                self.assertEqual(statuses["doc1"], "skipped")
                self.assertEqual(statuses["doc2"], "success")

                # process_questionnaire должен был быть вызван ТОЛЬКО один раз (для doc2)
                self.assertEqual(mock_processor.process_questionnaire.call_count, 1)

    def test_cmd_ontology_rename(self):
        """Проверка переименования узла через CLI ontology rename."""
        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="kidney_right",
            full_path="kidney.right",
            display_name="Правая почка",
            data_type="object",
        )
        self.session.add(node)
        self.session.commit()

        args = argparse.Namespace(
            node="kidney.right",
            new_name="Правая почка (лат. Ren dexter)",
            json=True,
        )

        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                cli.cmd_ontology_rename(args)
                resp = json.loads(fake_out.getvalue())
                self.assertEqual(resp["status"], "ok")
                self.assertEqual(resp["new_display_name"], "Правая почка (лат. Ren dexter)")

        self.session.refresh(node)
        self.assertEqual(node.display_name, "Правая почка (лат. Ren dexter)")

    def test_cmd_ontology_merge(self):
        """Проверка слияния узлов через CLI ontology merge."""
        src = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="cholecyst",
            full_path="cholecyst",
            display_name="Холецист",
            data_type="object",
        )
        tgt = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="gallbladder",
            full_path="gallbladder",
            display_name="Желчный пузырь",
            data_type="object",
        )
        self.session.add_all([src, tgt])
        self.session.flush()

        doc = Document(
            profile_id=self.default_profile.id,
            patient_id="P_MERGE",
            raw_text="...",
            administrative={},
        )
        self.session.add(doc)
        self.session.flush()

        val = DocumentValue(
            document_id=doc.id,
            feature_id=src.id,
            value_text="деформирован",
        )
        self.session.add(val)
        self.session.commit()

        args = argparse.Namespace(
            source="cholecyst",
            target="gallbladder",
            json=True,
        )

        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                cli.cmd_ontology_merge(args)
                resp = json.loads(fake_out.getvalue())
                self.assertEqual(resp["status"], "ok")
                self.assertEqual(resp["values_reassigned"], 1)

        # Исходный узел удален
        self.assertIsNone(self.session.query(FeatureNode).filter(FeatureNode.full_path == "cholecyst").first())
        # Значение документа перепривязано к целевому узлу
        self.session.refresh(val)
        self.assertEqual(val.feature_id, tgt.id)

        # Синоним добавлен к целевому узлу
        aliases = [a.alias_text for a in tgt.aliases]
        self.assertIn("Холецист", aliases)

    def test_cmd_ontology_delete_with_and_without_force(self):
        """Проверка удаления узла онтологии с защитой от удаления используемых значений."""
        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="spleen",
            full_path="spleen",
            display_name="Селезенка",
            data_type="object",
        )
        self.session.add(node)
        self.session.flush()

        doc = Document(
            profile_id=self.default_profile.id,
            patient_id="P_DEL",
            raw_text="...",
            administrative={},
        )
        self.session.add(doc)
        self.session.flush()

        val = DocumentValue(
            document_id=doc.id,
            feature_id=node.id,
            value_text="норма",
        )
        self.session.add(val)
        self.session.commit()

        args_no_force = argparse.Namespace(node="spleen", force=False, json=False)
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with self.assertRaises(SystemExit):
                cli.cmd_ontology_delete(args_no_force)

        # С флагом force=True удаление должно пройти успешно
        args_force = argparse.Namespace(node="spleen", force=True, json=True)
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                cli.cmd_ontology_delete(args_force)
                resp = json.loads(fake_out.getvalue())
                self.assertEqual(resp["status"], "ok")
                self.assertEqual(resp["deleted_values_count"], 1)

        self.assertIsNone(self.session.query(FeatureNode).filter(FeatureNode.full_path == "spleen").first())

    def test_dump_and_load_schema(self):
        """Проверка экспорта схемы онтологии в JSON и обратной загрузки."""
        parent_node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="liver",
            full_path="liver",
            display_name="Печень",
            data_type="object",
        )
        child_node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="left_lobe",
            full_path="liver.left_lobe",
            display_name="Левая доля печени",
            data_type="float",
            unit="мм",
        )
        self.session.add_all([parent_node, child_node])
        self.session.flush()
        alias = FeatureAlias(feature_id=child_node.id, alias_text="ЛДП")
        self.session.add(alias)
        self.session.commit()

        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_file = Path(tmp_dir) / "schema.json"

            # 1. Дамп схемы
            dump_args = argparse.Namespace(output=str(dump_file), json=False)
            with patch("cli.get_db_session") as mock_db:
                mock_db.return_value.__enter__.return_value = self.session
                cli.cmd_ontology_dump_schema(dump_args)

            self.assertTrue(dump_file.exists())
            with open(dump_file, "r", encoding="utf-8") as f:
                dump_json = json.load(f)

            self.assertEqual(dump_json["version"], "1.0")
            profile_entry = dump_json["profiles"][0]
            self.assertEqual(profile_entry["code"], "us_abdomen")
            feature_paths = [f["full_path"] for f in profile_entry["features"]]
            self.assertIn("liver", feature_paths)
            self.assertIn("liver.left_lobe", feature_paths)

            # 2. Очищаем узлы и загружаем заново
            self.session.query(FeatureAlias).delete()
            self.session.query(FeatureNode).delete()
            self.session.commit()
            self.assertEqual(self.session.query(FeatureNode).count(), 0)

            # 3. Загрузка схемы обратно
            load_args = argparse.Namespace(input=str(dump_file), overwrite=True, json=True)
            with patch("cli.get_db_session") as mock_db:
                mock_db.return_value.__enter__.return_value = self.session
                with patch("src.embeddings.get_embedding_provider", return_value=self.embedding_provider):
                    with patch("sys.stdout", new=io.StringIO()) as fake_out:
                        cli.cmd_ontology_load_schema(load_args)
                        load_resp = json.loads(fake_out.getvalue())
                        self.assertEqual(load_resp["status"], "ok")
                        self.assertEqual(load_resp["features_created"], 2)
                        self.assertEqual(load_resp["aliases_added"], 1)

            # Проверяем, что узлы восстановились в БД
            restored = self.session.query(FeatureNode).filter(FeatureNode.full_path == "liver.left_lobe").first()
            self.assertIsNotNone(restored)
            self.assertEqual(restored.display_name, "Левая доля печени")
            self.assertEqual(len(restored.aliases), 1)
            self.assertEqual(restored.aliases[0].alias_text, "ЛДП")


if __name__ == "__main__":
    unittest.main()
