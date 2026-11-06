"""Тесты для модуля универсального чтения документов src/document_reader.py."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile
from src.document_reader import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    extract_text_from_docx,
    extract_text_from_html,
    extract_text_from_json,
    extract_text_from_odt,
    extract_text_from_pdf,
    extract_text_from_rtf,
    read_document_text,
    read_text_file,
)


class TestDocumentReader(unittest.TestCase):
    """Проверка извлечения текста из различных форматов файлов (PDF, DOCX, ODT, RTF, HTML, TXT, JSON)."""

    def test_supported_extensions(self):
        """Проверка наличия всех ключевых расширений в SUPPORTED_DOCUMENT_EXTENSIONS."""
        expected = {".pdf", ".docx", ".doc", ".txt", ".odt", ".rtf", ".html", ".htm", ".json", ".md"}
        for ext in expected:
            self.assertIn(ext, SUPPORTED_DOCUMENT_EXTENSIONS)

    def test_read_text_file_encodings(self):
        """Проверка чтения текстовых файлов в UTF-8 и CP1251."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            utf8_p = Path(tmp_dir) / "test_utf8.txt"
            cp1251_p = Path(tmp_dir) / "test_cp1251.txt"

            sample_text = "Протокол консультативного осмотра: жалоб нет."
            utf8_p.write_text(sample_text, encoding="utf-8")
            cp1251_p.write_text(sample_text, encoding="cp1251")

            self.assertEqual(read_text_file(utf8_p), sample_text)
            self.assertEqual(read_text_file(cp1251_p), sample_text)

    def test_extract_text_from_docx_paragraphs_and_tables(self):
        """Проверка извлечения текста из DOCX (параграфы и таблицы) через встроенный XML парсер."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            docx_p = Path(tmp_dir) / "protocol.docx"

            doc_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Протокол УЗИ органов брюшной полости.</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Орган</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Размер</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Печень</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>150 мм</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:r><w:t>Заключение: без патологии.</w:t></w:r></w:p>
  </w:body>
</w:document>"""

            with zipfile.ZipFile(docx_p, "w") as z:
                z.writestr("word/document.xml", doc_xml)

            text = extract_text_from_docx(docx_p)
            self.assertIn("Протокол УЗИ органов брюшной полости.", text)
            self.assertIn("Печень | 150 мм", text)
            self.assertIn("Заключение: без патологии.", text)

    def test_extract_text_from_odt(self):
        """Проверка извлечения текста из OpenDocument (.odt)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            odt_p = Path(tmp_dir) / "protocol.odt"

            content_xml = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
                         xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
  <office:body>
    <office:text>
      <text:h>Осмотр кардиолога</text:h>
      <text:p>Пульс: 70 уд/мин, ритмичный.</text:p>
    </office:text>
  </office:body>
</office:document-content>"""

            with zipfile.ZipFile(odt_p, "w") as z:
                z.writestr("content.xml", content_xml)

            text = extract_text_from_odt(odt_p)
            self.assertIn("Осмотр кардиолога", text)
            self.assertIn("Пульс: 70 уд/мин, ритмичный.", text)

    def test_extract_text_from_rtf(self):
        """Проверка извлечения текста из RTF файла."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rtf_p = Path(tmp_dir) / "protocol.rtf"
            rtf_content = r"{\rtf1\ansi\deff0 {\fonttbl{\f0 Arial;}} \f0\fs24 Лист осмотра.\par Жалобы: одышка.\par}"
            rtf_p.write_bytes(rtf_content.encode("cp1251"))

            text = extract_text_from_rtf(rtf_p)
            self.assertIn("Лист осмотра.", text)
            self.assertIn("Жалобы: одышка.", text)

    def test_extract_text_from_html(self):
        """Проверка очистки HTML протокола от скриптов и тегов."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_p = Path(tmp_dir) / "protocol.html"
            html_content = """
            <html>
              <head><style>body { color: red; }</style></head>
              <body>
                <script>console.log("secret");</script>
                <h1>Консультация хирурга</h1>
                <p>Живот мягкий, безболезненный.</p>
              </body>
            </html>
            """
            html_p.write_text(html_content, encoding="utf-8")

            text = extract_text_from_html(html_p)
            self.assertIn("Консультация хирурга", text)
            self.assertIn("Живот мягкий, безболезненный.", text)
            self.assertNotIn("console.log", text)
            self.assertNotIn("color: red", text)

    def test_extract_text_from_json(self):
        """Проверка форматирования JSON протокола."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_p = Path(tmp_dir) / "protocol.json"
            data = {"patient": "Иванов", "vitals": {"temperature": 36.6}}
            json_p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            text = extract_text_from_json(json_p)
            self.assertIn('"patient": "Иванов"', text)
            self.assertIn('"temperature": 36.6', text)

    def test_extract_text_from_pdf_pdftotext(self):
        """Проверка извлечения текста из PDF через pdftotext."""
        with patch("subprocess.run") as mock_run:
            mock_res = MagicMock()
            mock_res.stdout = "Протокол ЭКГ: синусовый ритм."
            mock_run.return_value = mock_res

            text = extract_text_from_pdf("fake.pdf")
            self.assertEqual(text, "Протокол ЭКГ: синусовый ритм.")

    def test_read_document_text_dispatcher(self):
        """Проверка диспетчера read_document_text для разных форматов."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_p = Path(tmp_dir) / "sample.txt"
            txt_p.write_text("Текстовый файл анкеты", encoding="utf-8")

            res = read_document_text(txt_p)
            self.assertEqual(res, "Текстовый файл анкеты")

            # Ошибка при несуществующем файле
            with self.assertRaises(FileNotFoundError):
                read_document_text("non_existent_file.pdf")

            # Ошибка при пустом файле
            empty_p = Path(tmp_dir) / "empty.txt"
            empty_p.write_text("   \n  ", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_document_text(empty_p)


if __name__ == "__main__":
    unittest.main()
