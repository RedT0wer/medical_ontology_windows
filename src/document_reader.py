"""Модуль универсального чтения и извлечения текста из различных форматов медицинских протоколов.

Поддерживает форматы:
- PDF (.pdf) через pdftotext / pypdf / pdfplumber / fitz
- Word (.docx) через python-docx или встроенный парсер zipfile + XML
- Word (.doc) через antiword / catdoc / pdftotext
- OpenDocument (.odt) через zipfile + XML
- Rich Text Format (.rtf)
- HTML (.html, .htm) через BeautifulSoup
- JSON (.json)
- Текстовые файлы (.txt, .text, .md, .markdown, .log) с автоопределением кодировок (UTF-8, CP1251, UTF-8-SIG)

Принцип: один файл = одна медицинская анкета / один протокол исследования.
Приводит любой поддерживаемый файл к единому чистому тексту для дальнейшего извлечения признаков.
"""

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Set, Union
from src.config import logger

SUPPORTED_DOCUMENT_EXTENSIONS: Set[str] = {
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".log",
    ".pdf",
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    ".html",
    ".htm",
    ".json",
}


def read_text_file(path: Union[str, Path]) -> str:
    """Чтение текстового файла с автоматическим перебором кодировок (utf-8, cp1251, utf-8-sig, latin-1)."""
    p = Path(path)
    for enc in ("utf-8", "cp1251", "utf-8-sig", "latin-1"):
        try:
            with open(p, "r", encoding=enc) as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось определить кодировку текстового файла: {p}")


def extract_text_from_pdf(path: Union[str, Path]) -> str:
    """Извлечение текста из файла PDF."""
    p = Path(path)

    # 1. Попытка через установленные библиотеки Python (pypdf, pdfplumber, fitz)
    try:
        import pypdf

        reader = pypdf.PdfReader(str(p))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(page.strip() for page in pages if page.strip())
        if text.strip():
            return text
    except (ImportError, Exception) as e:
        logger.debug(f"pypdf недоступен или ошибка: {e}")

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(p))
        pages = [page.get_text() or "" for page in doc]
        text = "\n\n".join(page.strip() for page in pages if page.strip())
        if text.strip():
            return text
    except (ImportError, Exception) as e:
        logger.debug(f"PyMuPDF недоступен или ошибка: {e}")

    # 2. Попытка через системную утилиту pdftotext (poppler-utils)
    try:
        res = subprocess.run(
            ["pdftotext", "-layout", str(p), "-"],
            capture_output=True,
            text=True,
            check=True,
            timeout=45,
        )
        if res.stdout and res.stdout.strip():
            return res.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.debug(f"pdftotext недоступен: {e}")

    # 3. Fallback: поиск текстовых блоков в бинарном потоке PDF
    try:
        with open(p, "rb") as f:
            raw = f.read().decode("latin-1", errors="ignore")
        # Поиск текста внутри скобок операторов Tj / TJ в PDF потоках
        extracted = []
        for match in re.finditer(r"\((.*?)\)\s*Tj", raw):
            s = match.group(1).strip()
            if len(s) > 2:
                extracted.append(s)
        if extracted:
            return "\n".join(extracted)
    except Exception:
        pass

    raise RuntimeError(
        f"Не удалось извлечь текст из PDF '{p.name}'. "
        f"Убедитесь, что файл содержит текстовый слой (не является сканом) или установлена утилита pdftotext."
    )


def extract_text_from_docx(path: Union[str, Path]) -> str:
    """Извлечение текста из документа Word (.docx)."""
    p = Path(path)

    # 1. Попытка через библиотеку python-docx
    try:
        import docx

        doc = docx.Document(str(p))
        lines = []
        for par in doc.paragraphs:
            if par.text.strip():
                lines.append(par.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    lines.append(" | ".join(row_cells))
        if lines:
            return "\n".join(lines)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Ошибка python-docx: {e}")

    # 2. Встроенный парсинг XML без сторонних зависимостей (через zipfile)
    import xml.etree.ElementTree as ET
    import zipfile

    with zipfile.ZipFile(p) as z:
        if "word/document.xml" not in z.namelist():
            raise ValueError(f"Файл {p.name} не является корректным документом DOCX (нет word/document.xml).")
        xml_content = z.read("word/document.xml")

    root = ET.fromstring(xml_content)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    body = root.find(".//w:body", ns)
    if body is None:
        body = root

    lines = []
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            texts = [t.text for t in child.iterfind(".//w:t", ns) if t.text]
            if texts:
                lines.append("".join(texts).strip())
        elif tag == "tbl":
            for row in child.iterfind(".//w:tr", ns):
                cell_texts = []
                for tc in row.iterfind(".//w:tc", ns):
                    p_texts = [t.text for t in tc.iterfind(".//w:t", ns) if t.text]
                    if p_texts:
                        cell_texts.append("".join(p_texts).strip())
                if cell_texts:
                    lines.append(" | ".join(cell_texts))

    return "\n".join(line for line in lines if line)


def extract_text_from_doc(path: Union[str, Path]) -> str:
    """Извлечение текста из устаревшего формата Word (.doc)."""
    p = Path(path)
    for tool in ("antiword", "catdoc"):
        try:
            res = subprocess.run(
                [tool, str(p)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            if res.stdout and res.stdout.strip():
                return res.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            continue

    raise RuntimeError(
        f"Для чтения файлов старого формата .doc ('{p.name}') требуется утилита antiword или catdoc. "
        f"Рекомендуется сохранить файл в современном формате .docx или .txt."
    )


def extract_text_from_odt(path: Union[str, Path]) -> str:
    """Извлечение текста из OpenDocument (.odt)."""
    import xml.etree.ElementTree as ET
    import zipfile

    p = Path(path)
    with zipfile.ZipFile(p) as z:
        if "content.xml" not in z.namelist():
            raise ValueError(f"Файл {p.name} не является корректным документом ODT (нет content.xml).")
        xml_content = z.read("content.xml")

    root = ET.fromstring(xml_content)
    paragraphs = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in ("p", "h"):
            text = "".join(elem.itertext()).strip()
            if text:
                paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_text_from_rtf(path: Union[str, Path]) -> str:
    """Извлечение текста из файла Rich Text Format (.rtf)."""
    p = Path(path)

    try:
        from striprtf.striprtf import rtf_to_text

        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return rtf_to_text(f.read())
    except ImportError:
        pass

    # Встроенная очистка RTF разметки
    with open(p, "rb") as f:
        raw_bytes = f.read()

    text = raw_bytes.decode("cp1251", errors="ignore")
    text = re.sub(r"\\par[d]?\s*", "\n", text)
    text = re.sub(r"\\line\s*", "\n", text)
    text = re.sub(r"\\tab\s*", "\t", text)
    text = re.sub(r"\{\\[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+(-?\d+)?\s?", "", text)
    text = text.replace("{", "").replace("}", "")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def extract_text_from_html(path: Union[str, Path]) -> str:
    """Извлечение чистого текста из HTML страницы протокола."""
    from bs4 import BeautifulSoup

    raw = read_text_file(path)
    soup = BeautifulSoup(raw, "html.parser")
    for s in soup(["script", "style", "head", "title", "meta"]):
        s.extract()
    return soup.get_text(separator="\n", strip=True)


def extract_text_from_json(path: Union[str, Path]) -> str:
    """Чтение JSON протокола с форматированием в читаемый текстовый вид."""
    raw = read_text_file(path)
    try:
        data = json.loads(raw)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return raw


def read_document_text(file_path: Union[str, Path]) -> str:
    """Главная точка входа: определение типа файла и извлечение чистого текста анкеты.

    Принцип: один файл = одна анкета.

    Поддерживает: .pdf, .docx, .doc, .odt, .rtf, .html, .htm, .json, .txt, .md, .log
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл протокола не найден: {path.resolve()}")
    if not path.is_file():
        raise ValueError(f"Указанный путь не является файлом: {path.resolve()}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = extract_text_from_pdf(path)
    elif suffix == ".docx":
        text = extract_text_from_docx(path)
    elif suffix == ".doc":
        text = extract_text_from_doc(path)
    elif suffix == ".odt":
        text = extract_text_from_odt(path)
    elif suffix == ".rtf":
        text = extract_text_from_rtf(path)
    elif suffix in (".html", ".htm"):
        text = extract_text_from_html(path)
    elif suffix == ".json":
        text = extract_text_from_json(path)
    else:
        # По умолчанию читаем как текстовый файл (.txt, .md, .text, .log и др.)
        text = read_text_file(path)

    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"Файл '{path.name}' не содержит извлекаемого текста (пустой документ или скан без OCR).")

    return cleaned

