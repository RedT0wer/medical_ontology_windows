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

    def test_console_progress_bar_comprehensive(self):
        """Комплексная проверка ConsoleProgressBar: граничные значения total, обрезка, ETA и tty."""
        # 1. Граничный случай: total = 0
        buf0 = io.StringIO()
        pbar0 = cli.ConsoleProgressBar(total=0, width=10, stream=buf0)
        pbar0.update("sample.txt", status="success")
        pbar0.finish()
        self.assertIn("Завершено: 1/1", buf0.getvalue())

        # 2. Длинное имя файла (> 25 символов)
        buf1 = io.StringIO()
        pbar1 = cli.ConsoleProgressBar(total=2, width=10, stream=buf1)
        long_filename = "very_long_protocol_examination_ultra_sound_scan_2024.pdf"
        pbar1.update(long_filename, status="processing")
        pbar1.update(long_filename, status="success")
        pbar1.finish()
        out1 = buf1.getvalue()
        self.assertIn("very_long_protocol_exa...", out1)

        # 3. Режим TTY и ограничение по ширине экрана терминала
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = True
        pbar_tty = cli.ConsoleProgressBar(total=10, width=10, stream=mock_stream)
        with patch("shutil.get_terminal_size", return_value=type("Term", (), {"columns": 40})()):
            pbar_tty.update("short.txt", status="processing")
            # Проверяем, что вывод усечен до ширины терминала с \r в начале
            written_lines = [call.args[0] for call in mock_stream.write.call_args_list if call.args]
            self.assertTrue(any(line.startswith("\r") and len(line) <= 41 for line in written_lines))

    def test_process_directory_empty_or_unsupported(self):
        """Проверка реакции process_directory на пустую директорию и файлы неподдерживаемых форматов."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_p = Path(tmp_dir)
            # Файлы неподдерживаемых расширений
            (dir_p / "image.png").write_bytes(b"\x89PNG")
            (dir_p / "archive.zip").write_bytes(b"PK\x03\x04")

            mock_processor = MagicMock()
            results = cli.process_directory(
                dir_path=dir_p,
                processor=mock_processor,
                json_mode=True,
            )
            self.assertEqual(results, [])
            mock_processor.process_questionnaire.assert_not_called()

    def test_process_directory_skip_existing_by_patient_name(self):
        """Проверка пропуска анкеты по patient_name и извлечения сохраненного JSON кэша."""
        doc = Document(
            profile_id=self.default_profile.id,
            patient_id="PAT_ID_999",
            patient_name="protocol_existing",
            raw_text="...",
            administrative={},
        )
        self.session.add(doc)
        self.session.commit()

        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_p = Path(tmp_dir)
            out_p = dir_p / "output"
            out_p.mkdir()
            # Файл, имя которого совпадает с patient_name
            f = dir_p / "protocol_existing.txt"
            f.write_text("Протокол УЗИ", encoding="utf-8")
            cached_json = out_p / "protocol_existing_normalized.json"
            cached_json.write_text('{"status": "cached"}', encoding="utf-8")

            mock_processor = MagicMock()
            with patch("cli.get_db_session") as mock_db:
                mock_db.return_value.__enter__.return_value = self.session
                results = cli.process_directory(
                    dir_path=dir_p,
                    processor=mock_processor,
                    output_dir=str(out_p),
                    resume=True,
                    json_mode=True,
                )

                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["status"], "skipped")
                self.assertEqual(results[0]["saved_file"], str(cached_json))
                mock_processor.process_questionnaire.assert_not_called()

    def test_process_directory_partial_error_recovery(self):
        """Проверка отказоустойчивости пакетной обработки при возникновении исключения в отдельном файле."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_p = Path(tmp_dir)
            f_ok = dir_p / "doc_ok.txt"
            f_err = dir_p / "doc_err.txt"
            f_ok.write_text("ОК", encoding="utf-8")
            f_err.write_text("Сбой", encoding="utf-8")

            mock_processor = MagicMock()
            def side_effect(**kwargs):
                if kwargs.get("patient_id") == "doc_err":
                    raise RuntimeError("Критическая ошибка парсинга структуры")
                return {
                    "patient_id": "doc_ok",
                    "total_extracted": 1,
                    "new_features": 0,
                    "compiled_json": {},
                }
            mock_processor.process_questionnaire.side_effect = side_effect

            with patch("cli.get_db_session") as mock_db:
                mock_db.return_value.__enter__.return_value = self.session
                results = cli.process_directory(
                    dir_path=dir_p,
                    processor=mock_processor,
                    resume=True,
                    json_mode=True,
                )

                self.assertEqual(len(results), 2)
                statuses = {r["patient_id"] if "patient_id" in r else Path(r["file"]).stem: r["status"] for r in results}
                self.assertEqual(statuses["doc_ok"], "success")
                self.assertEqual(statuses["doc_err"], "error")
                err_item = next(r for r in results if r["status"] == "error")
                self.assertIn("Критическая ошибка", err_item["error"])

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


    def test_cmd_ontology_rename_by_id_and_not_found(self):
        """Проверка переименования узла по числовому ID и обработка отсутствующих узлов."""
        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="pancreas",
            full_path="pancreas",
            display_name="Поджелудочная",
            data_type="object",
        )
        self.session.add(node)
        self.session.commit()

        # 1. Переименование по числовому ID
        args_id = argparse.Namespace(
            node=str(node.id),
            new_name="Поджелудочная железа",
            json=True,
        )
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                cli.cmd_ontology_rename(args_id)
                resp = json.loads(fake_out.getvalue())
                self.assertEqual(resp["status"], "ok")
                self.assertEqual(resp["node_id"], node.id)
                self.assertEqual(resp["new_display_name"], "Поджелудочная железа")

        # 2. Несуществующий узел в режиме --json
        args_nf_json = argparse.Namespace(node="missing_node", new_name="Тест", json=True)
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                with self.assertRaises(SystemExit):
                    cli.cmd_ontology_rename(args_nf_json)
                resp_err = json.loads(fake_out.getvalue())
                self.assertEqual(resp_err["status"], "error")

        # 3. Несуществующий узел в обычном режиме
        args_nf = argparse.Namespace(node="missing_node", new_name="Тест", json=False)
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with patch("sys.stderr", new=io.StringIO()) as fake_err:
                with self.assertRaises(SystemExit):
                    cli.cmd_ontology_rename(args_nf)
                self.assertIn("не найден", fake_err.getvalue())

    def test_cmd_ontology_merge_edge_cases(self):
        """Комплексная проверка слияния узлов: совпадение узлов, несуществующие узлы, перенос детей и значений."""
        parent_src = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="src_branch",
            full_path="src_branch",
            display_name="Исходная ветка",
            data_type="object",
        )
        parent_tgt = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="tgt_branch",
            full_path="tgt_branch",
            display_name="Целевая ветка",
            data_type="object",
        )
        self.session.add_all([parent_src, parent_tgt])
        self.session.flush()

        # Дочерний узел исходной ветки
        child_node = FeatureNode(
            profile_id=self.default_profile.id,
            parent_id=parent_src.id,
            key_name="child",
            full_path="src_branch.child",
            display_name="Дочерний элемент",
            data_type="string",
        )
        self.session.add(child_node)
        self.session.flush()

        # Документ со значениями в обоих узлах (проверка объединения контекстов)
        doc = Document(profile_id=self.default_profile.id, patient_id="PAT_COL", raw_text="...", administrative={})
        self.session.add(doc)
        self.session.flush()
        val_src = DocumentValue(
            document_id=doc.id,
            feature_id=parent_src.id,
            value_text="признак А",
            raw_context="Контекст А",
        )
        val_tgt = DocumentValue(
            document_id=doc.id,
            feature_id=parent_tgt.id,
            value_text="признак Б",
            raw_context="Контекст Б",
        )
        self.session.add_all([val_src, val_tgt])
        self.session.commit()

        # 1. Ошибка: исходный и целевой узлы совпадают
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with self.assertRaises(SystemExit):
                cli.cmd_ontology_merge(argparse.Namespace(source="src_branch", target="src_branch", json=False))

        # 2. Ошибка: несуществующий узел
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with self.assertRaises(SystemExit):
                cli.cmd_ontology_merge(argparse.Namespace(source="non_existent", target="tgt_branch", json=False))

        # 3. Успешное слияние с объединением конфликтующих значений и переносом детей
        args_merge = argparse.Namespace(source="src_branch", target="tgt_branch", json=True)
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                cli.cmd_ontology_merge(args_merge)
                resp = json.loads(fake_out.getvalue())
                self.assertEqual(resp["status"], "ok")

        # Дочерний узел перепривязан к parent_tgt
        self.session.refresh(child_node)
        self.assertEqual(child_node.parent_id, parent_tgt.id)

        # Контекст объединен: 'Контекст Б; Контекст А'
        self.session.refresh(val_tgt)
        self.assertIn("Контекст Б", val_tgt.raw_context)
        self.assertIn("Контекст А", val_tgt.raw_context)

    def test_cmd_ontology_delete_unused_and_not_found(self):
        """Проверка удаления неиспользуемого узла без force, по ID и обработка 404."""
        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="temp_node",
            full_path="temp_node",
            display_name="Временный узел",
            data_type="string",
        )
        self.session.add(node)
        self.session.commit()
        node_id = node.id

        # 1. Удаление неиспользуемого узла без флага force (должно пройти успешно)
        args_del = argparse.Namespace(node=str(node_id), force=False, json=True)
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                cli.cmd_ontology_delete(args_del)
                resp = json.loads(fake_out.getvalue())
                self.assertEqual(resp["status"], "ok")
                self.assertEqual(resp["node_id"], node_id)

        self.assertIsNone(self.session.get(FeatureNode, node_id))

        # 2. Удаление несуществующего узла
        args_del_missing = argparse.Namespace(node="missing_path", force=False, json=False)
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with self.assertRaises(SystemExit):
                cli.cmd_ontology_delete(args_del_missing)

    def test_dump_and_load_schema_advanced(self):
        """Расширенная проверка dump-schema в stdout, ошибки load-schema и режима overwrite."""
        node = FeatureNode(
            profile_id=self.default_profile.id,
            key_name="bladder",
            full_path="bladder",
            display_name="Мочевой пузырь",
            data_type="object",
        )
        self.session.add(node)
        self.session.commit()

        # 1. Дамп схемы без указания output файла (вывод в stdout)
        dump_stdout_args = argparse.Namespace(output=None, json=False)
        with patch("cli.get_db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = self.session
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                cli.cmd_ontology_dump_schema(dump_stdout_args)
                raw_json = json.loads(fake_out.getvalue())
                self.assertIn("profiles", raw_json)
                self.assertEqual(raw_json["version"], "1.0")

        # 2. Загрузка схемы из несуществующего файла
        load_missing_args = argparse.Namespace(input="non_existent_schema_file.json", overwrite=False, json=False)
        with self.assertRaises(SystemExit):
            cli.cmd_ontology_load_schema(load_missing_args)

        # 3. Загрузка схемы с флагом overwrite=False (не перезаписывает) vs overwrite=True (обновляет)
        with tempfile.TemporaryDirectory() as tmp_dir:
            schema_file = Path(tmp_dir) / "test_schema.json"
            schema_content = {
                "version": "1.0",
                "profiles": [
                    {
                        "code": "us_abdomen",
                        "features": [
                            {
                                "key_name": "bladder",
                                "full_path": "bladder",
                                "display_name": "Мочевой пузырь (обновленный)",
                                "data_type": "string",
                                "unit": "мл",
                                "aliases": ["Пузырь"],
                            }
                        ],
                    }
                ],
            }
            schema_file.write_text(json.dumps(schema_content), encoding="utf-8")

            # Без overwrite: display_name не должен измениться
            load_no_ow = argparse.Namespace(input=str(schema_file), overwrite=False, json=True)
            with patch("cli.get_db_session") as mock_db:
                mock_db.return_value.__enter__.return_value = self.session
                with patch("src.embeddings.get_embedding_provider", return_value=self.embedding_provider):
                    cli.cmd_ontology_load_schema(load_no_ow)

            self.session.refresh(node)
            self.assertEqual(node.display_name, "Мочевой пузырь")

            # С overwrite=True: display_name и unit должны обновиться
            load_ow = argparse.Namespace(input=str(schema_file), overwrite=True, json=True)
            with patch("cli.get_db_session") as mock_db:
                mock_db.return_value.__enter__.return_value = self.session
                with patch("src.embeddings.get_embedding_provider", return_value=self.embedding_provider):
                    cli.cmd_ontology_load_schema(load_ow)

            self.session.refresh(node)
            self.assertEqual(node.display_name, "Мочевой пузырь (обновленный)")
            self.assertEqual(node.unit, "мл")


if __name__ == "__main__":
    unittest.main()


