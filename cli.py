#!/usr/bin/env python3
"""Консольный программный интерфейс (CLI API) для интеграции с внешними приложениями.

Позволяет вызывать функции онтологии из любых скриптов (Bash, Python, Node.js, Go),
поддерживает передачу данных через аргументы и stdin, и возвращает чистый JSON в stdout.
"""

import argparse
import json
import logging
import sys
from typing import Optional
from src.config import logger, settings
from src.db import get_db_session
from src.embeddings import get_embedding_provider
from src.matcher import MatchingEngine
from src.models import Parameter, ParameterSynonym
from src.pipeline import QuestionnaireProcessor


def configure_cli_logging(json_mode: bool):
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



def cmd_process(args):
    """Обработка текста анкеты через CLI."""
    configure_cli_logging(args.json)

    text_input = args.text
    if args.stdin or (not text_input and not sys.stdin.isatty()):
        text_input = sys.stdin.read().strip()

    if not text_input:
        sys.stderr.write("Ошибка: текст анкеты не передан (используйте --text или передайте через stdin)\n")
        sys.exit(1)

    processor = QuestionnaireProcessor()
    result = processor.process_questionnaire(text=text_input, patient_id=args.patient_id)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Обработана анкета {args.patient_id}: извлечено {result['total_extracted']} параметров, создано новых: {result['new_parameters']}, синонимов: {result['synonyms_linked']}.")


def cmd_match(args):
    """Сопоставление отдельного сырого термина с базой онтологии."""
    configure_cli_logging(args.json)

    with get_db_session() as session:
        matcher = MatchingEngine(session=session)
        decision = matcher.match_parameter(
            raw_key=args.term,
            raw_value=args.value or "",
            context=args.context or "",
        )

        candidates_data = [
            {
                "parameter_id": c.parameter_id,
                "canonical_name": c.canonical_name,
                "matched_text": c.matched_text,
                "match_source": c.match_source,
                "similarity": round(c.similarity, 4),
            }
            for c in decision.candidates
        ]

        output = {
            "term": args.term,
            "action": decision.action,
            "parameter_id": decision.parameter_id,
            "canonical_name": decision.canonical_name,
            "similarity": round(decision.similarity, 4),
            "reason": decision.reason,
            "candidates": candidates_data,
        }

        if args.json:
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"Термин: '{args.term}' -> {decision.action} [{decision.canonical_name}] (сходство: {decision.similarity:.4f})")


def cmd_search(args):
    """Векторный поиск ближайших параметров в онтологии (Top-K)."""
    configure_cli_logging(args.json)

    with get_db_session() as session:
        matcher = MatchingEngine(session=session)
        candidates = matcher.find_top_candidates(args.query, top_k=args.top)

        results = [
            {
                "parameter_id": c.parameter_id,
                "canonical_name": c.canonical_name,
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
                print(f"[{i}] ID {c['parameter_id']}: '{c['canonical_name']}' (сходство {c['similarity']} через '{c['matched_text']}')")


def cmd_ontology(args):
    """Выгрузка текущего дерева онтологии."""
    configure_cli_logging(args.json)

    processor = QuestionnaireProcessor()
    summary = processor.get_ontology_summary()

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Всего параметров: {summary['total_parameters']}, синонимов: {summary['total_synonyms']}, записей: {summary['total_patient_records']}")
        for p in summary["parameters"]:
            syns = ", ".join(p["synonyms"]) if p["synonyms"] else "нет"
            print(f"  [{p['id']}] {p['canonical_name']} ({p['category']}) | синонимы: {syns}")


def cmd_add_synonym(args):
    """Связывание нового синонима с существующим параметром."""
    configure_cli_logging(args.json)

    processor = QuestionnaireProcessor()
    with get_db_session() as session:
        processor._ensure_synonym(session, args.parameter_id, args.synonym)

    output = {
        "status": "ok",
        "parameter_id": args.parameter_id,
        "synonym": args.synonym,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"Синоним '{args.synonym}' успешно привязан к параметру ID {args.parameter_id}")


def cmd_excel(args):
    """Пакетная обработка пациентов из файла input.xlsm."""
    configure_cli_logging(args.json)

    processor = QuestionnaireProcessor()
    results = processor.process_excel_file(
        file_path=args.file,
        sheet_name=args.sheet,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"Обработка завершена: обработано {len(results)} пациентов из {args.file}.")


def main():
    if "--json" in sys.argv:
        configure_cli_logging(True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Переопределить модель LLM (например: qwen2.5:7b)",
    )

    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Консольный программный API для сервиса онтологии медицинских параметров.",
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Команды CLI API")

    # 1. Команда: process (обработка анкеты)
    p_proc = subparsers.add_parser("process", parents=[parent_parser], help="Обработка клинической анкеты пациента")
    p_proc.add_argument("--text", type=str, default="", help="Текст анкеты")
    p_proc.add_argument("--stdin", action="store_true", help="Считать текст анкеты из стандартного потока ввода (stdin)")
    p_proc.add_argument("--patient-id", type=str, default="CLI-ПАЦИЕНТ", help="Идентификатор пациента")
    p_proc.add_argument("--json", action="store_true", help="Вывод строгого машиночитаемого JSON в stdout")
    p_proc.set_defaults(func=cmd_process)

    # 2. Команда: match (сопоставление одного термина)
    p_match = subparsers.add_parser("match", parents=[parent_parser], help="Сопоставление одного сырого медицинского термина")
    p_match.add_argument("term", type=str, help="Сырой термин (например: 'АД на входе')")
    p_match.add_argument("--value", type=str, default="", help="Значение показателя (например: '140/90')")
    p_match.add_argument("--context", type=str, default="", help="Контекстное предложение")
    p_match.add_argument("--json", action="store_true", help="Вывод строгого JSON в stdout")
    p_match.set_defaults(func=cmd_match)

    # 3. Команда: search (векторный поиск)
    p_search = subparsers.add_parser("search", parents=[parent_parser], help="Векторный поиск ближайших параметров по смыслу")
    p_search.add_argument("query", type=str, help="Текст для векторного поиска (например: 'головная боль')")
    p_search.add_argument("--top", type=int, default=3, help="Количество возвращаемых кандидатов (Top-K)")
    p_search.add_argument("--json", action="store_true", help="Вывод строгого JSON в stdout")
    p_search.set_defaults(func=cmd_search)

    # 4. Команда: ontology (выгрузка базы знаний)
    p_onto = subparsers.add_parser("ontology", parents=[parent_parser], help="Выгрузка текущего дерева онтологии")
    p_onto.add_argument("--json", action="store_true", help="Вывод строгого JSON в stdout")
    p_onto.set_defaults(func=cmd_ontology)

    # 5. Команда: add-synonym (добавление синонима)
    p_syn = subparsers.add_parser("add-synonym", parents=[parent_parser], help="Привязка нового синонима к параметру")
    p_syn.add_argument("--parameter-id", type=int, required=True, help="ID канонического параметра")
    p_syn.add_argument("--synonym", type=str, required=True, help="Текст добавляемого синонима")
    p_syn.add_argument("--json", action="store_true", help="Вывод строгого JSON в stdout")
    p_syn.set_defaults(func=cmd_add_synonym)

    # 6. Команда: excel (пакетная обработка Excel)
    p_xl = subparsers.add_parser("excel", parents=[parent_parser], help="Пакетная обработка пациентов из input.xlsm")
    p_xl.add_argument("--file", type=str, default="input.xlsm", help="Путь к файлу")
    p_xl.add_argument("--sheet", type=str, default="Лист1", help="Имя листа")
    p_xl.add_argument("--limit", type=int, default=None, help="Лимит пациентов (по умолчанию: все анкеты файла)")
    p_xl.add_argument("--json", action="store_true", help="Вывод строгого JSON в stdout")
    p_xl.set_defaults(func=cmd_excel)

    args = parser.parse_args()
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

