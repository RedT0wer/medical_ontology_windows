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


if __name__ == "__main__":
    unittest.main()
