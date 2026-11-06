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
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional
from src.anonymizer import anonymize_administrative, anonymize_text
from src.config import logger, settings
from src.db import get_db_session, init_db
from src.document_reader import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    read_document_text,
    read_text_file as _read_text_file,
)
from src.export import export_dataset, load_feature_config
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
    """Чтение текстового файла (обратная совместимость)."""
    return _read_text_file(path)


class ConsoleProgressBar:
    """Консольный индикатор прогресса пакетной обработки файлов."""

    def __init__(self, total: int, width: int = 24, stream=sys.stderr):
        self.total = max(1, total)
        self.width = width
        self.stream = stream
        self.current = 0
        self.success_count = 0
        self.skip_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.is_tty = hasattr(stream, "isatty") and stream.isatty()

    def update(self, filename: str, status: str = "processing") -> None:
        """Обновление индикатора прогресса."""
        if status == "success":
            self.success_count += 1
            self.current += 1
        elif status == "skipped":
            self.skip_count += 1
            self.current += 1
        elif status == "error":
            self.error_count += 1
            self.current += 1

        elapsed = time.time() - self.start_time
        pct = int((self.current / self.total) * 100) if self.total else 100
        pct = min(100, max(0, pct))

        filled = int(self.width * (self.current / self.total)) if self.total else self.width
        bar = "█" * filled + "░" * (self.width - filled)

        if self.current > 0 and self.current < self.total:
            rate = elapsed / self.current
            remaining_sec = int(rate * (self.total - self.current))
            eta_str = f"{remaining_sec // 60:02d}:{remaining_sec % 60:02d}"
        elif self.current >= self.total:
            eta_str = "00:00"
        else:
            eta_str = "--:--"

        disp_file = filename
        if len(disp_file) > 25:
            disp_file = disp_file[:22] + "..."

        line = (
            f"[{bar}] {pct:>3}% ({self.current}/{self.total}) "
            f"| Успешно: {self.success_count}, Пропуск: {self.skip_count}, Ошибки: {self.error_count} "
            f"| ETA: {eta_str} | {disp_file}"
        )

        if self.is_tty:
            term_width = shutil.get_terminal_size((80, 20)).columns
            if len(line) > term_width - 1:
                line = line[: term_width - 4] + "..."
            self.stream.write(f"\r{line}")
            self.stream.flush()
        else:
            if status != "processing":
                self.stream.write(f"{line}\n")
                self.stream.flush()

    def finish(self) -> None:
        """Печать финальной строки и перевод каретки."""
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        if self.is_tty:
            self.stream.write("\n")
        self.stream.write(
            f"Завершено: {self.current}/{self.total} файлов за {elapsed_str} "
            f"(Успешно: {self.success_count}, Пропущено: {self.skip_count}, Ошибок: {self.error_count})\n"
        )
        self.stream.flush()


def process_single_file(
    file_path: Path,
    processor: QuestionnaireProcessor,
    patient_id: Optional[str] = None,
    save_json: bool = True,
    output_dir: Optional[str] = None,
    anonymize: bool = False,
) -> Dict[str, Any]:
    """Обработка одного файла протокола (PDF, DOCX, TXT, ODT, RTF, HTML, JSON и др.).
    
    Принцип: один файл = одна анкета.
    """
    content = read_document_text(file_path)
    inferred_id = patient_id or file_path.stem

    call_kwargs: Dict[str, Any] = {
        "text": content,
        "patient_id": inferred_id,
        "save_json": False,
    }
    if anonymize:
        call_kwargs["anonymize"] = True

    result = processor.process_questionnaire(**call_kwargs)

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
    resume: bool = False,
    anonymize: bool = False,
) -> List[Dict[str, Any]]:
    """Пакетная обработка всех файлов протоколов в указанной директории (PDF, DOCX, TXT, RTF, ODT и др.)."""
    pattern = "**/*" if recursive else "*"
    all_files = [
        p for p in dir_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]
    all_files.sort()

    if not all_files:
        supported_str = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        sys.stderr.write(f"В директории {dir_path} не найдено поддерживаемых файлов протоколов ({supported_str}).\n")
        return []

    pbar: Optional[ConsoleProgressBar] = None
    if not json_mode:
        print(f"Найдено файлов для обработки: {len(all_files)} в папке '{dir_path}'.")
        pbar = ConsoleProgressBar(total=len(all_files))

    results = []
    for file_p in all_files:
        inferred_id = file_p.stem

        # Проверка инкрементального режима (--resume / --skip-existing)
        if resume:
            with get_db_session() as sess:
                existing_doc = (
                    sess.query(Document)
                    .filter((Document.patient_id == inferred_id) | (Document.patient_name == inferred_id))
                    .first()
                )
            if existing_doc:
                skipped_info = {
                    "file": str(file_p),
                    "status": "skipped",
                    "patient_id": inferred_id,
                    "reason": f"Анкета уже существует в базе данных (ID документа: {existing_doc.id})",
                    "document_id": existing_doc.id,
                }
                out_dir = Path(output_dir or "analysis_output")
                existing_json = out_dir / f"{file_p.stem}_normalized.json"
                if existing_json.exists():
                    skipped_info["saved_file"] = str(existing_json)
                results.append(skipped_info)
                if pbar:
                    pbar.update(filename=file_p.name, status="skipped")
                continue

        if pbar:
            pbar.update(filename=file_p.name, status="processing")

        try:
            res = process_single_file(
                file_path=file_p,
                processor=processor,
                save_json=save_json,
                output_dir=output_dir,
                anonymize=anonymize,
            )
            item = {
                "file": str(file_p),
                "status": "success",
                "patient_id": res["patient_id"],
                "total_extracted": res["total_extracted"],
                "new_features": res["new_features"],
                "saved_file": res.get("saved_file"),
                "compiled_json": res["compiled_json"],
            }
            if anonymize:
                item["anonymized"] = True
                if "anonymization_audit" in res:
                    item["anonymization_audit"] = res["anonymization_audit"]
            results.append(item)
            if pbar:
                pbar.update(filename=file_p.name, status="success")
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_p}: {e}")
            results.append(
                {
                    "file": str(file_p),
                    "status": "error",
                    "error": str(e),
                }
            )
            if pbar:
                pbar.update(filename=file_p.name, status="error")

    if pbar:
        pbar.finish()

    return results


def cmd_process(args) -> None:
    """Пакетная обработка директории с файлами медицинских анкет."""
    configure_cli_logging(args.json)

    dir_param = getattr(args, "dir", None)
    if not dir_param:
        sys.stderr.write(
            "Ошибка: отправляется не отдельный файл, а именно директория. "
            "Требуется указать параметр --dir (например: python cli.py process --dir /путь/к/папке).\n"
        )
        sys.exit(1)

    target_dir = Path(dir_param)
    if target_dir.is_file():
        sys.stderr.write(
            f"Ошибка: отправляется не файл, а именно директория. "
            f"Указанный путь '{target_dir}' является файлом. "
            f"Укажите директорию с файлами анкет: python cli.py process --dir <путь_к_папке>\n"
        )
        sys.exit(1)

    if not target_dir.exists():
        sys.stderr.write(f"Ошибка: директория не существует: {target_dir}\n")
        sys.exit(1)

    processor = QuestionnaireProcessor()
    resume_flag = getattr(args, "resume", False) or getattr(args, "skip_existing", False)
    anonymize_flag = getattr(args, "anonymize", False)

    results = process_directory(
        dir_path=target_dir,
        processor=processor,
        recursive=args.recursive,
        save_json=not args.no_save,
        output_dir=args.output_dir,
        json_mode=args.json,
        resume=resume_flag,
        anonymize=anonymize_flag,
    )
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        success_cnt = sum(1 for r in results if r["status"] == "success")
        skipped_cnt = sum(1 for r in results if r.get("status") == "skipped")
        print(f"\nГотово: успешно обработано {success_cnt}, пропущено {skipped_cnt} из {len(results)} файлов.")


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
    """Выгрузка текущей схемы и дерева признаков онтологии (поведение по умолчанию)."""
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


def cmd_ontology_rename(args) -> None:
    """Переименование отображаемого имени узла онтологии."""
    configure_cli_logging(args.json)
    node_ident = args.node.strip()
    new_name = args.new_name.strip()

    with get_db_session() as session:
        node = None
        if node_ident.isdigit():
            node = session.get(FeatureNode, int(node_ident))
        if not node:
            node = session.query(FeatureNode).filter(FeatureNode.full_path == node_ident).first()
        if not node:
            err_msg = f"Узел онтологии '{node_ident}' не найден."
            if args.json:
                print(json.dumps({"status": "error", "message": err_msg}, ensure_ascii=False))
            else:
                sys.stderr.write(f"Ошибка: {err_msg}\n")
            sys.exit(1)

        old_name = node.display_name
        node.display_name = new_name
        session.commit()

        res = {
            "status": "ok",
            "node_id": node.id,
            "full_path": node.full_path,
            "old_display_name": old_name,
            "new_display_name": new_name,
        }
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            print(f"Узел #{node.id} ('{node.full_path}') переименован: '{old_name}' -> '{new_name}'")


def cmd_ontology_merge(args) -> None:
    """Слияние двух узлов онтологии (--source переносится в --target, а затем удаляется)."""
    configure_cli_logging(args.json)
    src_ident = args.source.strip()
    tgt_ident = args.target.strip()

    if src_ident == tgt_ident:
        sys.stderr.write("Ошибка: исходный и целевой узлы не могут совпадать.\n")
        sys.exit(1)

    with get_db_session() as session:
        src_node = None
        if src_ident.isdigit():
            src_node = session.get(FeatureNode, int(src_ident))
        if not src_node:
            src_node = session.query(FeatureNode).filter(FeatureNode.full_path == src_ident).first()

        tgt_node = None
        if tgt_ident.isdigit():
            tgt_node = session.get(FeatureNode, int(tgt_ident))
        if not tgt_node:
            tgt_node = session.query(FeatureNode).filter(FeatureNode.full_path == tgt_ident).first()

        if not src_node:
            sys.stderr.write(f"Ошибка: исходный узел '{src_ident}' не найден.\n")
            sys.exit(1)
        if not tgt_node:
            sys.stderr.write(f"Ошибка: целевой узел '{tgt_ident}' не найден.\n")
            sys.exit(1)

        if src_node.id == tgt_node.id:
            sys.stderr.write("Ошибка: исходный и целевой узлы указывают на одну сущность.\n")
            sys.exit(1)

        # 1. Перенос значений DocumentValue
        src_values = session.query(DocumentValue).filter(DocumentValue.feature_id == src_node.id).all()
        reassigned_count = 0
        for sv in src_values:
            existing_tgt_val = (
                session.query(DocumentValue)
                .filter(DocumentValue.document_id == sv.document_id, DocumentValue.feature_id == tgt_node.id)
                .first()
            )
            if not existing_tgt_val:
                sv.feature = tgt_node
                sv.feature_id = tgt_node.id
                reassigned_count += 1
            else:
                if existing_tgt_val.value_numeric is None and sv.value_numeric is not None:
                    existing_tgt_val.value_numeric = sv.value_numeric
                if not existing_tgt_val.value_text and sv.value_text:
                    existing_tgt_val.value_text = sv.value_text
                if not existing_tgt_val.value_json and sv.value_json:
                    existing_tgt_val.value_json = sv.value_json
                if sv.raw_context and sv.raw_context not in (existing_tgt_val.raw_context or ""):
                    existing_tgt_val.raw_context = (f"{existing_tgt_val.raw_context or ''}; {sv.raw_context}").strip("; ")
                session.delete(sv)
                reassigned_count += 1

        # 2. Перепривязка дочерних узлов
        children = session.query(FeatureNode).filter(FeatureNode.parent_id == src_node.id).all()
        for ch in children:
            ch.parent_id = tgt_node.id

        # 3. Перенос синонимов
        aliases_to_add = {src_node.display_name, src_node.key_name, src_node.full_path}
        for a in src_node.aliases:
            aliases_to_add.add(a.alias_text)

        existing_aliases = {a.alias_text.lower() for a in tgt_node.aliases}
        added_aliases_count = 0
        for alias_text in aliases_to_add:
            if alias_text and alias_text.lower() not in existing_aliases:
                session.add(FeatureAlias(feature_id=tgt_node.id, alias_text=alias_text))
                existing_aliases.add(alias_text.lower())
                added_aliases_count += 1

        src_path = src_node.full_path
        tgt_path = tgt_node.full_path
        session.delete(src_node)
        session.commit()

        res = {
            "status": "ok",
            "source": src_path,
            "target": tgt_path,
            "values_reassigned": reassigned_count,
            "aliases_added": added_aliases_count,
        }
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            print(f"Слияние узлов успешно: '{src_path}' -> '{tgt_path}'")
            print(f"  Перенесено значений анкет: {reassigned_count}")
            print(f"  Добавлено синонимов к целевому узлу: {added_aliases_count}")


def cmd_ontology_delete(args) -> None:
    """Удаление узла онтологии."""
    configure_cli_logging(args.json)
    node_ident = args.node.strip()
    force = getattr(args, "force", False)

    with get_db_session() as session:
        node = None
        if node_ident.isdigit():
            node = session.get(FeatureNode, int(node_ident))
        if not node:
            node = session.query(FeatureNode).filter(FeatureNode.full_path == node_ident).first()

        if not node:
            sys.stderr.write(f"Ошибка: узел онтологии '{node_ident}' не найден.\n")
            sys.exit(1)

        val_count = session.query(DocumentValue).filter(DocumentValue.feature_id == node.id).count()
        if val_count > 0 and not force:
            sys.stderr.write(
                f"Ошибка: узел '{node.full_path}' используется в {val_count} значениях анкет. "
                f"Для принудительного удаления используйте флаг --force.\n"
            )
            sys.exit(1)

        node_path = node.full_path
        node_id = node.id
        session.delete(node)
        session.commit()

        res = {
            "status": "ok",
            "node_id": node_id,
            "full_path": node_path,
            "deleted_values_count": val_count,
        }
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            print(f"Узел #{node_id} ('{node_path}') успешно удален (удалено значений анкет: {val_count}).")


def cmd_ontology_dump_schema(args) -> None:
    """Экспорт схемы онтологии в формате JSON."""
    configure_cli_logging(args.json)
    with get_db_session() as session:
        profiles = session.query(DocumentProfile).all()
        dump_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "profiles": [],
        }
        for p in profiles:
            prof_data = {
                "code": p.code,
                "name": p.name,
                "description": p.description,
                "features": [],
            }
            features = (
                session.query(FeatureNode)
                .filter(FeatureNode.profile_id == p.id)
                .order_by(FeatureNode.full_path)
                .all()
            )
            for f in features:
                prof_data["features"].append(
                    {
                        "key_name": f.key_name,
                        "full_path": f.full_path,
                        "display_name": f.display_name,
                        "description": f.description,
                        "data_type": f.data_type,
                        "unit": f.unit,
                        "is_leaf": f.is_leaf,
                        "status": f.status,
                        "metadata": f.meta_data or {},
                        "aliases": [a.alias_text for a in f.aliases],
                    }
                )
            dump_data["profiles"].append(prof_data)

    if getattr(args, "output", None):
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, ensure_ascii=False, indent=2)
        if args.json:
            print(json.dumps({"status": "ok", "saved_file": str(out_p.resolve())}, ensure_ascii=False))
        else:
            print(f"Схема онтологии успешно выгружена в файл: {out_p.resolve()}")
    else:
        print(json.dumps(dump_data, ensure_ascii=False, indent=2))


def cmd_ontology_load_schema(args) -> None:
    """Импорт схемы онтологии из файла JSON."""
    configure_cli_logging(args.json)
    input_p = Path(args.input)
    if not input_p.exists():
        sys.stderr.write(f"Ошибка: файл схемы не найден: {input_p}\n")
        sys.exit(1)

    with open(input_p, "r", encoding="utf-8") as f:
        schema_data = json.load(f)

    profiles_count = 0
    created_count = 0
    updated_count = 0
    aliases_count = 0

    from src.embeddings import get_embedding_provider
    provider = get_embedding_provider()

    with get_db_session() as session:
        for prof_info in schema_data.get("profiles", []):
            code = prof_info.get("code", "us_abdomen")
            profile = session.query(DocumentProfile).filter(DocumentProfile.code == code).first()
            if not profile:
                profile = DocumentProfile(
                    code=code,
                    name=prof_info.get("name", code),
                    description=prof_info.get("description"),
                )
                session.add(profile)
                session.flush()
                profiles_count += 1

            features = prof_info.get("features", [])
            features.sort(key=lambda x: len(x.get("full_path", "").split(".")))

            for f_info in features:
                f_path = f_info.get("full_path", "")
                if not f_path:
                    continue

                existing_node = (
                    session.query(FeatureNode)
                    .filter(FeatureNode.profile_id == profile.id, FeatureNode.full_path == f_path)
                    .first()
                )

                if existing_node:
                    if getattr(args, "overwrite", False):
                        existing_node.display_name = f_info.get("display_name", existing_node.display_name)
                        existing_node.description = f_info.get("description", existing_node.description)
                        existing_node.data_type = f_info.get("data_type", existing_node.data_type)
                        existing_node.unit = f_info.get("unit", existing_node.unit)
                        existing_node.is_leaf = f_info.get("is_leaf", existing_node.is_leaf)
                        existing_node.status = f_info.get("status", existing_node.status)
                        updated_count += 1
                    target_node = existing_node
                else:
                    parts = f_path.split(".")
                    parent_id = None
                    if len(parts) > 1:
                        parent_path = ".".join(parts[:-1])
                        parent_node = (
                            session.query(FeatureNode)
                            .filter(FeatureNode.profile_id == profile.id, FeatureNode.full_path == parent_path)
                            .first()
                        )
                        if parent_node:
                            parent_id = parent_node.id

                    vec = None
                    if provider:
                        try:
                            vec = provider.embed_text(f_info.get("display_name", f_path))
                        except Exception:
                            vec = None

                    target_node = FeatureNode(
                        profile_id=profile.id,
                        parent_id=parent_id,
                        key_name=f_info.get("key_name", parts[-1]),
                        full_path=f_path,
                        display_name=f_info.get("display_name", parts[-1]),
                        description=f_info.get("description"),
                        data_type=f_info.get("data_type", "string"),
                        unit=f_info.get("unit"),
                        is_leaf=f_info.get("is_leaf", True),
                        status=f_info.get("status", "approved"),
                        embedding=vec,
                        meta_data=f_info.get("metadata", {}),
                    )
                    session.add(target_node)
                    session.flush()
                    created_count += 1

                existing_aliases = {a.alias_text.lower() for a in target_node.aliases}
                for alias_txt in f_info.get("aliases", []):
                    if alias_txt and alias_txt.lower() not in existing_aliases:
                        vec_a = None
                        if provider:
                            try:
                                vec_a = provider.embed_text(alias_txt)
                            except Exception:
                                vec_a = None
                        session.add(
                            FeatureAlias(
                                feature_id=target_node.id,
                                alias_text=alias_txt,
                                embedding=vec_a,
                            )
                        )
                        existing_aliases.add(alias_txt.lower())
                        aliases_count += 1

        session.commit()

    res = {
        "status": "ok",
        "profiles_created": profiles_count,
        "features_created": created_count,
        "features_updated": updated_count,
        "aliases_added": aliases_count,
    }
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(f"Схема онтологии успешно загружена:")
        print(f"  Создано новых профилей: {profiles_count}")
        print(f"  Создано новых признаков: {created_count}")
        print(f"  Обновлено существующих: {updated_count}")
        print(f"  Добавлено синонимов: {aliases_count}")


def cmd_export(args) -> None:
    """Сборка и выгрузка датасета признаков в CSV / Parquet по конфигурационному файлу с фильтрацией."""
    is_json = getattr(args, "json", False) or "--json" in sys.argv
    configure_cli_logging(is_json)
    config_file = getattr(args, "config", None) or getattr(args, "config_pos", None)
    if not config_file and Path("configs/features.yaml").exists():
        config_file = "configs/features.yaml"

    try:
        result = export_dataset(
            config=config_file,
            output=getattr(args, "output", None),
            format=getattr(args, "format", None),
            from_date=getattr(args, "from_date", None),
            to_date=getattr(args, "to_date", None),
            filter_diagnosis=getattr(args, "filter_diagnosis", None),
            filter_doctor=getattr(args, "filter_doctor", None),
            filter_institution=getattr(args, "filter_institution", None),
        )

        if is_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"Датасет успешно собран и экспортирован:")
            print(f"  Файл: {result['file']}")
            print(f"  Формат: {result['format']}")
            print(f"  Всего анкет/пациентов: {result['total_documents']}")
            print(f"  Признаков: {result['total_features']}")
            print(f"  Колонок в файле: {len(result['columns'])}")
            print(f"  Размер: {result['file_size_bytes']} байт")
            if result.get("filters"):
                print(f"  Примененные фильтры: {result['filters']}")
            print(f"  Отсутствующие признаки заполнены как NaN.")
    except Exception as e:
        sys.stderr.write(f"Ошибка при экспорте датасета: {e}\n")
        sys.exit(1)


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


def cmd_web(args) -> None:
    """Запуск веб-интерфейса для просмотра анкет и онтологии признаков."""
    from src.web.app import run_server

    run_server(host=args.host, port=args.port, reload=args.reload)


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

    # 1. Команда: process (пакетная обработка директории с анкетами, требуется --dir)
    p_proc = subparsers.add_parser(
        "process",
        parents=[parent_parser],
        help="Пакетная обработка директории с протоколами/анкетами (требуется --dir)",
    )
    p_proc.add_argument(
        "--dir",
        "-d",
        type=str,
        default=None,
        help="Путь к директории с файлами анкет (обязательный параметр: отправляется директория, не файл)",
    )
    p_proc.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help=argparse.SUPPRESS,
    )
    p_proc.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
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
        default="_output",
        help="Папка для сохранения нормализованных JSON файлов",
    )
    p_proc.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять скомпилированные JSON файлы на диск",
    )
    p_proc.add_argument(
        "--resume",
        action="store_true",
        help="Инкрементальный режим: пропускать файлы, уже сохраненные в базе данных",
    )
    p_proc.add_argument(
        "--skip-existing",
        action="store_true",
        help="Синоним для --resume: пропускать уже обработанные файлы",
    )
    p_proc.add_argument(
        "--anonymize",
        action="store_true",
        help="Деидентификация персональных данных (ФИО, паспорта, СНИЛС, полисы, контакты) по 152-ФЗ и HIPAA",
    )
    p_proc.add_argument(
        "--json",
        action="store_true",
        help="Вывод чистого JSON в stdout",
    )
    p_proc.set_defaults(func=cmd_process)

    # 2. Команда: ontology (схема, дерево, курация, экспорт и импорт схемы)
    p_onto = subparsers.add_parser(
        "ontology",
        parents=[parent_parser],
        help="Управление онтологией, схема, курация и выгрузка",
    )
    p_onto.add_argument("--json", action="store_true", help="Вывод чистого JSON в stdout")
    p_onto.set_defaults(func=cmd_ontology)

    onto_sub = p_onto.add_subparsers(dest="onto_command", required=False)

    # 2.1 ontology tree
    p_tree = onto_sub.add_parser("tree", help="Выгрузка дерева признаков и статистики онтологии")
    p_tree.add_argument("--json", action="store_true", help="Вывод чистого JSON в stdout")
    p_tree.set_defaults(func=cmd_ontology)

    # 2.2 ontology rename <node> <new_name>
    p_ren = onto_sub.add_parser("rename", help="Переименование отображаемого имени узла онтологии")
    p_ren.add_argument("node", type=str, help="ID узла или full_path (например: liver.right_lobe)")
    p_ren.add_argument("new_name", type=str, help="Новое отображаемое имя узла")
    p_ren.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    p_ren.set_defaults(func=cmd_ontology_rename)

    # 2.3 ontology merge --source <src> --target <dst>
    p_mrg = onto_sub.add_parser("merge", help="Слияние двух узлов онтологии (--source вливается в --target)")
    p_mrg.add_argument("--source", "-s", required=True, type=str, help="Исходный узел (ID или full_path)")
    p_mrg.add_argument("--target", "-t", required=True, type=str, help="Целевой узел (ID или full_path)")
    p_mrg.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    p_mrg.set_defaults(func=cmd_ontology_merge)

    # 2.4 ontology delete <node> [--force]
    p_del = onto_sub.add_parser("delete", help="Удаление узла онтологии")
    p_del.add_argument("node", type=str, help="ID узла или full_path")
    p_del.add_argument("--force", action="store_true", help="Принудительное удаление вместе со значениями анкет")
    p_del.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    p_del.set_defaults(func=cmd_ontology_delete)

    # 2.5 ontology dump-schema -o <file.json>
    p_dump = onto_sub.add_parser("dump-schema", help="Экспорт схемы онтологии в файл JSON")
    p_dump.add_argument("--output", "-o", type=str, default=None, help="Путь к файлу JSON")
    p_dump.add_argument("--json", action="store_true", help="Вывод в stdout в формате JSON")
    p_dump.set_defaults(func=cmd_ontology_dump_schema)

    # 2.6 ontology load-schema -i <file.json> [--overwrite]
    p_load = onto_sub.add_parser("load-schema", help="Импорт схемы онтологии из файла JSON")
    p_load.add_argument("--input", "-i", required=True, type=str, help="Путь к файлу схемы JSON")
    p_load.add_argument("--overwrite", action="store_true", help="Перезаписывать существующие узлы")
    p_load.add_argument("--json", action="store_true", help="Вывод статуса в формате JSON")
    p_load.set_defaults(func=cmd_ontology_load_schema)

    # 3. Команда: export
    p_exp = subparsers.add_parser(
        "export",
        parents=[parent_parser],
        help="Экспорт датасета признаков в CSV / Parquet по YAML-конфигурации",
    )
    p_exp.add_argument(
        "config_pos",
        nargs="?",
        default=None,
        metavar="CONFIG",
        help="Путь к YAML-файлу конфигурации признаков (например, configs/features.yaml)",
    )
    p_exp.add_argument(
        "--config",
        "-c",
        dest="config",
        type=str,
        default=None,
        help="Путь к YAML-файлу конфигурации признаков",
    )
    p_exp.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Путь к результирующему файлу (.csv или .parquet)",
    )
    p_exp.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["csv", "parquet"],
        default=None,
        help="Формат выгрузки (csv или parquet)",
    )
    p_exp.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Фильтрация: начальная дата осмотра (ГГГГ-ММ-ДД)",
    )
    p_exp.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="Фильтрация: конечная дата осмотра (ГГГГ-ММ-ДД)",
    )
    p_exp.add_argument(
        "--filter-diagnosis",
        type=str,
        default=None,
        help="Фильтрация: подстрока клинического диагноза/заключения",
    )
    p_exp.add_argument(
        "--filter-doctor",
        type=str,
        default=None,
        help="Фильтрация: подстрока ФИО врача",
    )
    p_exp.add_argument(
        "--filter-institution",
        type=str,
        default=None,
        help="Фильтрация: подстрока медицинского учреждения",
    )
    p_exp.add_argument(
        "--json",
        action="store_true",
        help="Вывод отчета об экспорте в формате JSON",
    )
    p_exp.set_defaults(func=cmd_export)

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

    # 7. Команда: web (запуск веб-интерфейса)
    p_web = subparsers.add_parser("web", parents=[parent_parser], help="Запуск простого веб-интерфейса онтологии и анкет")
    p_web.add_argument("--host", type=str, default="127.0.0.1", help="Хост веб-сервера (по умолчанию: 127.0.0.1)")
    p_web.add_argument("--port", type=int, default=8000, help="Порт веб-сервера (по умолчанию: 8000)")
    p_web.add_argument("--reload", action="store_true", help="Автоматический перезапуск при изменении кода")
    p_web.set_defaults(func=cmd_web)

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
