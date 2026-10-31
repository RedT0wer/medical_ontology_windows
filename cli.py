#!/usr/bin/env python3
"""Консольный программный интерфейс (CLI) для динамической онтологии медицинских протоколов.

Позволяет:
1. Обрабатывать отдельные файлы протоколов (например, '1 copy.txt').
2. Пакетно обрабатывать директории с файлами протоколов.
3. Принимать текст через аргументы или stdin.
4. Выполнять векторный семантический поиск по онтологии.
5. Выгружать структуру онтологии и экспортировать датасет признаков в CSV с NaN.
6. Инициализировать базу данных (pgvector, HNSW, GIN).
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from src.config import logger, settings
from src.db import get_db_session, init_db
from src.matcher import MatchingEngine
from src.models import (
    Document,
    DocumentJsonCache,
    DocumentProfile,
    DocumentValue,
    FeatureAlias,
    FeatureNode,
)
from src.pipeline import QuestionnaireProcessor


def configure_cli_logging(json_mode: bool) -> None:
    """При выводе JSON перенаправляет все текстовые логи в stderr, оставляя stdout чистым."""
    if json_mode:
        for l in [logger, logging.root, logging.getLogger("medical_ontology")]:
            for h in list(l.handlers):
                l.removeHandler(h)
            new_h = logging.StreamHandler(sys.stderr)
            new_h.setLevel(logging.INFO)
            from src.config import RussianLogFormatter

            new_h.setFormatter(
                RussianLogFormatter(
                    fmt="[%(levelname)s] %(asctime)s - %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            l.addHandler(new_h)


def read_text_file(path: Path) -> str:
    """Чтение текстового файла с автоматическим определением кодировки (utf-8 / cp1251)."""
    for enc in ("utf-8", "cp1251", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось декодировать файл: {path}")


def process_single_file(
    file_path: Path,
    processor: QuestionnaireProcessor,
    patient_id: Optional[str] = None,
    save_json: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Обработка одного файла протокола."""
    content = read_text_file(file_path)
    # Если patient_id не задан явно, пробуем взять имя файла без расширения
    inferred_id = patient_id or file_path.stem

    result = processor.process_questionnaire(
        text=content,
        patient_id=inferred_id,
        save_json=False,  # Сохраняем локально сами в нужную директорию
    )

    saved_path = None
    if save_json:
        out_dir = Path(output_dir or "analysis_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{file_path.stem}_normalized.json"
        saved_path = str(out_dir / safe_name)
        with open(saved_path, "w", encoding="utf-8") as f:
            json.dump(result["compiled_json"], f, ensure_ascii=False, indent=2)
        result["saved_file"] = saved_path

    return result


def process_directory(
    dir_path: Path,
    processor: QuestionnaireProcessor,
    recursive: bool = False,
    save_json: bool = True,
    output_dir: Optional[str] = None,
    json_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Пакетная обработка всех файлов протоколов в указанной директории."""
    pattern = "**/*" if recursive else "*"
    all_files = [
        p for p in dir_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in (".txt", ".text", ".md", ".log")
    ]
    all_files.sort()

    if not all_files:
        sys.stderr.write(f"В директории {dir_path} не найдено текстовых файлов (.txt, .text, .md).\n")
        return []

    if not json_mode:
        print(f"Найдено файлов для обработки: {len(all_files)} в папке '{dir_path}'.")

    results = []
    for i, file_p in enumerate(all_files, 1):
        if not json_mode:
            print(f"[{i}/{len(all_files)}] Обработка файла: {file_p.name}...")

        try:
            res = process_single_file(
                file_path=file_p,
                processor=processor,
                save_json=save_json,
                output_dir=output_dir,
            )
            results.append(
                {
                    "file": str(file_p),
                    "status": "success",
                    "patient_id": res["patient_id"],
                    "total_extracted": res["total_extracted"],
                    "new_features": res["new_features"],
                    "saved_file": res.get("saved_file"),
                    "compiled_json": res["compiled_json"],
                }
            )
            if not json_mode:
                print(
                    f"   -> Успех: пациент '{res['patient_id']}', "
                    f"извлечено признаков: {res['total_extracted']} (новых: {res['new_features']}). "
                    f"JSON: {res.get('saved_file', 'не сохранен')}"
                )
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_p}: {e}")
            results.append(
                {
                    "file": str(file_p),
                    "status": "error",
                    "error": str(e),
                }
            )
            if not json_mode:
                print(f"   -> Ошибка: {e}")

    return results


def cmd_process(args) -> None:
    """Обработка одного файла, директории с файлами или сырого текста."""
    configure_cli_logging(args.json)
    processor = QuestionnaireProcessor()

    # 1. Проверка директории (через --dir или позиционный аргумент)
    target_dir: Optional[Path] = None
    if getattr(args, "dir", None):
        target_dir = Path(args.dir)
    elif args.input_path and Path(args.input_path).is_dir():
        target_dir = Path(args.input_path)

    if target_dir:
        if not target_dir.exists():
            sys.stderr.write(f"Ошибка: директория не существует: {target_dir}\n")
            sys.exit(1)

        results = process_directory(
            dir_path=target_dir,
            processor=processor,
            recursive=args.recursive,
            save_json=not args.no_save,
            output_dir=args.output_dir,
            json_mode=args.json,
        )
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            success_cnt = sum(1 for r in results if r["status"] == "success")
            print(f"\nГотово: успешно обработано {success_cnt} из {len(results)} файлов.")
        return

    # 2. Проверка файла (через --file или позиционный аргумент)
    target_file: Optional[Path] = None
    if getattr(args, "file", None):
        target_file = Path(args.file)
    elif args.input_path and Path(args.input_path).is_file():
        target_file = Path(args.input_path)

    if target_file:
        if not target_file.exists():
            sys.stderr.write(f"Ошибка: файл не найден: {target_file}\n")
            sys.exit(1)

        result = process_single_file(
            file_path=target_file,
            processor=processor,
            patient_id=args.patient_id,
            save_json=not args.no_save,
            output_dir=args.output_dir,
        )

        if args.json:
            print(json.dumps(result["compiled_json"], ensure_ascii=False, indent=2))
        else:
            print(f"Протокол {target_file.name} успешно обработан:")
            print(f"  Пациент: {result['patient_id']} ({result['patient_name'] or 'Не указан'})")
            print(f"  Извлечено признаков: {result['total_extracted']}")
            print(f"  Новых узлов схемы: {result['new_features']}")
            print(f"  Сопоставлено признаков: {result['matched_features']}")
            if result.get("saved_file"):
                print(f"  Нормализованный JSON сохранен: {result['saved_file']}")
        return

    # 3. Чтение сырого текста или stdin
    text_input = args.text
    if args.stdin or (not text_input and not sys.stdin.isatty()):
        text_input = sys.stdin.read().strip()

    if not text_input:
        sys.stderr.write(
            "Ошибка: укажите путь к файлу (--file или аргумент), папке (--dir), "
            "передайте текст (--text) или используйте stdin.\n"
        )
        sys.exit(1)

    result = processor.process_questionnaire(
        text=text_input,
        patient_id=args.patient_id,
        save_json=not args.no_save,
    )

    if args.json:
        print(json.dumps(result["compiled_json"], ensure_ascii=False, indent=2))
    else:
        print(f"Протокол {args.patient_id or 'пациента'} успешно обработан:")
        print(f"  Извлечено признаков: {result['total_extracted']}")
        print(f"  Новых узлов схемы: {result['new_features']}")
        print(f"  Сопоставлено: {result['matched_features']}")
        if result.get("saved_file"):
            print(f"  JSON сохранен: {result['saved_file']}")


def cmd_match(args) -> None:
    """Тестирование сопоставления отдельного клинического термина."""
    configure_cli_logging(args.json)

    with get_db_session() as session:
        matcher = MatchingEngine(session=session)
        decision = matcher.match_parameter(
            raw_key=args.term,
            raw_value=args.value or "",
            context=args.context or "",
        )

        output = {
            "term": args.term,
            "action": decision.action,
            "feature_id": decision.feature_id,
            "full_path": decision.full_path,
            "data_type": decision.data_type,
            "similarity": round(decision.similarity, 4),
            "reason": decision.reason,
            "candidates": [
                {
                    "feature_id": c.feature_id,
                    "full_path": c.full_path,
                    "matched_text": c.matched_text,
                    "match_source": c.match_source,
                    "similarity": round(c.similarity, 4),
                }
                for c in decision.candidates
            ],
        }

        if args.json:
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"Термин: '{args.term}' -> {decision.action} [Путь: {decision.full_path}] (сходство: {decision.similarity:.4f})")
            print(f"  Причина: {decision.reason}")


def cmd_search(args) -> None:
    """Векторный поиск ближайших узлов онтологии (Top-K)."""
    configure_cli_logging(args.json)

    with get_db_session() as session:
        matcher = MatchingEngine(session=session)
        candidates = matcher.find_top_candidates(args.query, top_k=args.top)

        results = [
            {
                "feature_id": c.feature_id,
                "full_path": c.full_path,
                "matched_text": c.matched_text,
                "match_source": c.match_source,
                "similarity": round(c.similarity, 4),
            }
            for c in candidates
        ]

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for i, c in enumerate(results, 1):
                print(f"[{i}] ID {c['feature_id']}: '{c['full_path']}' (сходство {c['similarity']:.4f} через '{c['matched_text']}')")


def cmd_ontology(args) -> None:
    """Выгрузка текущей схемы и дерева признаков онтологии."""
    configure_cli_logging(args.json)

    processor = QuestionnaireProcessor()
    summary = processor.get_ontology_summary()

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Профиль: {summary['profile_name']} ({summary['profile_code']})")
        print(f"Всего признаков: {summary['total_features']} (листьев: {summary['leaf_features']}, веток: {summary['branch_features']})")
        print(f"Всего синонимов: {summary['total_aliases']}, документов: {summary['total_documents']}, значений: {summary['total_values']}\n")
        print("Дерево признаков:")
        for f in summary["features"]:
            prefix = "  [ЛИСТ]" if f["is_leaf"] else "[ВЕТКА]"
            unit_info = f", ед: {f['unit']}" if f["unit"] else ""
            recs_info = f", записей: {f['records_count']}" if f["is_leaf"] else ""
            print(f"  {prefix} {f['full_path']:<40} (тип: {f['data_type']}{unit_info}{recs_info})")


def cmd_export_csv(args) -> None:
    """Сборка и выгрузка датасета признаков в CSV с автоматическим заполнением NaN для пропущенных полей."""
    configure_cli_logging(args.json)
    out_file = Path(args.output)

    with get_db_session() as session:
        profile = session.query(DocumentProfile).filter(DocumentProfile.code == "us_abdomen").first()
        if not profile:
            sys.stderr.write("Профиль документов не найден в базе данных.\n")
            sys.exit(1)

        # 1. Определение списка запрашиваемых признаков
        target_paths: List[str] = []
        if args.features:
            target_paths = [f.strip() for f in args.features.split(",") if f.strip()]
        else:
            # Все листовые признаки профиля
            leaf_nodes = (
                session.query(FeatureNode)
                .filter(FeatureNode.profile_id == profile.id, FeatureNode.is_leaf == True)
                .order_by(FeatureNode.full_path)
                .all()
            )
            target_paths = [n.full_path for n in leaf_nodes if n.full_path != "conclusion"]

        if not target_paths:
            sys.stderr.write("В онтологии нет зарегистрированных признаков для экспорта.\n")
            sys.exit(1)

        # 2. Получение всех документов профиля
        documents = (
            session.query(Document)
            .filter(Document.profile_id == profile.id)
            .order_by(Document.id)
            .all()
        )

        if not documents:
            sys.stderr.write("В базе нет обработанных документов для выгрузки.\n")
            sys.exit(1)

        # 3. Сбор разреженной матрицы
        fieldnames = ["document_id", "patient_id", "consultation_date"] + target_paths
        rows: List[Dict[str, Any]] = []

        for doc in documents:
            row: Dict[str, Any] = {
                "document_id": doc.id,
                "patient_id": doc.patient_id,
                "consultation_date": doc.consultation_date.isoformat() if doc.consultation_date else "NaN",
            }
            # Сопоставляем значения документа
            val_map = {
                v.feature.full_path: (
                    v.value_numeric if v.value_numeric is not None else (v.value_text or "NaN")
                )
                for v in doc.values
                if v.feature
            }
            for path in target_paths:
                row[path] = val_map.get(path, "NaN")

            rows.append(row)

        # 4. Запись в CSV
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="NaN")
            writer.writeheader()
            writer.writerows(rows)

        if args.json:
            print(json.dumps({"status": "ok", "total_documents": len(rows), "total_features": len(target_paths), "file": str(out_file)}, ensure_ascii=False, indent=2))
        else:
            print(f"Датасет успешно собран и экспортирован в '{out_file}':")
            print(f"  Всего пациентов/документов: {len(rows)}")
            print(f"  Колонок признаков: {len(target_paths)}")
            print(f"  Отсутствующие значения заполнены как NaN.")


def cmd_init_db(args) -> None:
    """Инициализация базы данных PostgreSQL (pgvector, HNSW, GIN)."""
    configure_cli_logging(args.json)
    if args.drop_all:
        print("ВНИМАНИЕ: Все существующие таблицы будут удалены и созданы заново!")
    init_db(drop_all=args.drop_all)
    if args.json:
        print(json.dumps({"status": "ok", "message": "Database initialized successfully"}, ensure_ascii=False))
    else:
        print("База данных успешно инициализирована со всеми индексами и базовым профилем.")


def main():
    if "--json" in sys.argv:
        configure_cli_logging(True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Переопределить модель LLM (например: qwen2.5:14b-instruct-q4_K_M)",
    )

    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Консольный CLI-интерфейс для динамического структурирования медицинских протоколов.",
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Команды")

    # 1. Команда: process (обработка файла, папки или текста)
    p_proc = subparsers.add_parser(
        "process",
        parents=[parent_parser],
        help="Обработка клинического протокола (файл, папка или сырой текст)",
    )
    p_proc.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Путь к файлу протокола (например: '1 copy.txt') или папке с файлами",
    )
    p_proc.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Явный путь к файлу протокола для обработки",
    )
    p_proc.add_argument(
        "--dir",
        "-d",
        type=str,
        default=None,
        help="Явный путь к папке с файлами для пакетной обработки",
    )
    p_proc.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Рекурсивный поиск файлов в поддиректориях",
    )
    p_proc.add_argument(
        "--text",
        "-t",
        type=str,
        default="",
        help="Текст протокола строкой",
    )
    p_proc.add_argument(
        "--stdin",
        action="store_true",
        help="Считать текст из стандартного потока ввода",
    )
    p_proc.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Идентификатор пациента (если не указан, извлекается из текста/файла)",
    )
    p_proc.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="analysis_output",
        help="Папка для сохранения нормализованных JSON файлов",
    )
    p_proc.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять скомпилированные JSON файлы на диск",
    )
    p_proc.add_argument(
        "--json",
        action="store_true",
        help="Вывод чистого JSON в stdout",
    )
    p_proc.set_defaults(func=cmd_process)

    # 2. Команда: ontology (схема и дерево)
    p_onto = subparsers.add_parser("ontology", parents=[parent_parser], help="Выгрузка дерева признаков онтологии")
    p_onto.add_argument("--json", action="store_true", help="Вывод чистого JSON в stdout")
    p_onto.set_defaults(func=cmd_ontology)

    # 3. Команда: export-csv (выгрузка датасета)
    p_exp = subparsers.add_parser("export-csv", parents=[parent_parser], help="Экспорт датасета признаков в CSV с NaN")
    p_exp.add_argument("--output", "-o", type=str, default="dataset.csv", help="Путь к результирующему CSV-файлу")
    p_exp.add_argument("--features", type=str, default="", help="Список признаков через запятую (по умолчанию: все)")
    p_exp.add_argument("--json", action="store_true", help="Вывод статуса в формате JSON")
    p_exp.set_defaults(func=cmd_export_csv)

    # 4. Команда: search (поиск по вектору)
    p_search = subparsers.add_parser("search", parents=[parent_parser], help="Семантический поиск признака в схеме")
    p_search.add_argument("query", type=str, help="Поисковый запрос (например: 'толщина стенки')")
    p_search.add_argument("--top", type=int, default=3, help="Количество кандидатов (Top-K)")
    p_search.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    p_search.set_defaults(func=cmd_search)

    # 5. Команда: match (тест матчинга термина)
    p_match = subparsers.add_parser("match", parents=[parent_parser], help="Тест сопоставления сырого термина")
    p_match.add_argument("term", type=str, help="Сырой термин или путь")
    p_match.add_argument("--value", type=str, default="", help="Значение показателя")
    p_match.add_argument("--context", type=str, default="", help="Контекст")
    p_match.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    p_match.set_defaults(func=cmd_match)

    # 6. Команда: init-db (инициализация БД)
    p_init = subparsers.add_parser("init-db", parents=[parent_parser], help="Инициализация схемы БД и индексов pgvector")
    p_init.add_argument("--drop-all", action="store_true", help="Удалить существующие таблицы перед созданием")
    p_init.add_argument("--json", action="store_true", help="Вывод статуса в формате JSON")
    p_init.set_defaults(func=cmd_init_db)

    args = parser.parse_args()

    # Переопределение модели LLM при необходимости
    model_name = getattr(args, "model", "")
    if not model_name and "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_name = sys.argv[idx + 1]

    if model_name:
        settings.LLM_MODEL = model_name

    if "--json" in sys.argv:
        configure_cli_logging(True)

    args.func(args)


if __name__ == "__main__":
    main()
