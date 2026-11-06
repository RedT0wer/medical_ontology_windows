"""Тесты для командного интерфейса (CLI) cli.py и точки запуска web_app.py."""

import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from cli import configure_cli_logging, read_text_file
import cli


class TestCLI(unittest.TestCase):
    """Проверка вспомогательных функций CLI и парсера командной строки."""

    def test_configure_cli_logging(self):
        """Проверка перенаправления логов в stderr в JSON-режиме."""
        configure_cli_logging(json_mode=True)
        # Проверяем, что хэндлер пишет в stderr
        for handler in cli.logger.handlers:
            if isinstance(handler, cli.logging.StreamHandler):
                self.assertEqual(handler.stream, cli.sys.stderr)

    def test_read_text_file_utf8_and_cp1251(self):
        """Проверка чтения файлов с автоопределением кодировки (UTF-8 и CP1251)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            utf8_file = Path(tmp_dir) / "test_utf8.txt"
            cp1251_file = Path(tmp_dir) / "test_cp1251.txt"

            test_text = "Тестовый медицинский протокол: УЗИ ОБП."

            with open(utf8_file, "w", encoding="utf-8") as f:
                f.write(test_text)

            with open(cp1251_file, "w", encoding="cp1251") as f:
                f.write(test_text)

            self.assertEqual(read_text_file(utf8_file), test_text)
            self.assertEqual(read_text_file(cp1251_file), test_text)

    def test_cli_argument_parsing_subcommands(self):
        """Проверка разбора аргументов всех подкоманд CLI."""
        # 1. process
        with patch("sys.argv", ["cli.py", "process", "protocol.txt", "--patient-id", "PAT-1"]):
            with patch("cli.cmd_process") as mock_cmd:
                mock_cmd.return_value = None
                cli.main()
                mock_cmd.assert_called_once()

        # 2. ontology
        with patch("sys.argv", ["cli.py", "ontology", "--json"]):
            with patch("cli.cmd_ontology") as mock_cmd:
                mock_cmd.return_value = None
                cli.main()
                mock_cmd.assert_called_once()

        # 3. export
        with patch("sys.argv", ["cli.py", "export", "--format", "csv", "--output", "out.csv"]):
            with patch("cli.cmd_export") as mock_cmd:
                mock_cmd.return_value = None
                cli.main()
                mock_cmd.assert_called_once()

        # 4. search
        with patch("sys.argv", ["cli.py", "search", "печень"]):
            with patch("cli.cmd_search") as mock_cmd:
                mock_cmd.return_value = None
                cli.main()
                mock_cmd.assert_called_once()

        # 5. match
        with patch("sys.argv", ["cli.py", "match", "ЧСС", "--value", "72"]):
            with patch("cli.cmd_match") as mock_cmd:
                mock_cmd.return_value = None
                cli.main()
                mock_cmd.assert_called_once()

        # 6. init-db
        with patch("sys.argv", ["cli.py", "init-db"]):
            with patch("cli.cmd_init_db") as mock_cmd:
                mock_cmd.return_value = None
                cli.main()
                mock_cmd.assert_called_once()

        # 7. web
        with patch("sys.argv", ["cli.py", "web", "--port", "8080"]):
            with patch("cli.cmd_web") as mock_cmd:
                mock_cmd.return_value = None
                cli.main()
                mock_cmd.assert_called_once()

    def test_web_app_main_parser(self):
        """Проверка аргументов запуска в web_app.py."""
        import web_app

        with patch("sys.argv", ["web_app.py", "--host", "0.0.0.0", "--port", "9000", "--reload"]):
            with patch("web_app.run_server") as mock_run:
                web_app.main()
                mock_run.assert_called_once_with(host="0.0.0.0", port=9000, reload=True)


    def test_process_single_file_multiformat(self):
        """Проверка вызова process_single_file для разных форматов файлов (PDF, DOCX, TXT)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_p = Path(tmp_dir) / "protocol.pdf"
            file_p.write_bytes(b"%PDF-1.4 dummy")

            mock_processor = MagicMock()
            mock_processor.process_questionnaire.return_value = {
                "patient_id": "P-PDF",
                "total_extracted": 3,
                "new_features": 1,
                "compiled_json": {"status": "ok"},
            }

            with patch("cli.read_document_text", return_value="Текст извлеченного PDF протокола"):
                res = cli.process_single_file(file_p, mock_processor, save_json=False)

            self.assertEqual(res["patient_id"], "P-PDF")
            mock_processor.process_questionnaire.assert_called_once_with(
                text="Текст извлеченного PDF протокола",
                patient_id="protocol",
                save_json=False,
            )

    def test_process_directory_multiformat(self):
        """Проверка пакетной обработки файлов разных форматов (.pdf, .docx, .txt)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_p = Path(tmp_dir)
            (dir_p / "doc1.txt").write_text("Текст 1", encoding="utf-8")
            (dir_p / "doc2.docx").write_bytes(b"PK dummy docx")
            (dir_p / "doc3.pdf").write_bytes(b"%PDF dummy")
            (dir_p / "ignore.exe").write_bytes(b"MZ binary")

            mock_processor = MagicMock()
            mock_processor.process_questionnaire.return_value = {
                "patient_id": "PID",
                "total_extracted": 1,
                "new_features": 0,
                "compiled_json": {},
            }

            with patch("cli.read_document_text", return_value="Текст протокола"):
                results = cli.process_directory(dir_p, mock_processor, save_json=False, json_mode=True)

            self.assertEqual(len(results), 3)
            processed_exts = {Path(r["file"]).suffix.lower() for r in results}
            self.assertEqual(processed_exts, {".txt", ".docx", ".pdf"})


if __name__ == "__main__":
    unittest.main()
