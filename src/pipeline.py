"""Главный модуль пайплайна динамического структурирования медицинских протоколов.

Оркестрирует:
1. Чтение входных данных одного анализа (текст протокола или файл вида '1 copy.txt').
2. Извлечение сущностей через LLM + детерминированный парсинг (ExtractedDocument).
3. Регистрацию исходного протокола (Document).
4. Сопоставление с онтологией и развитие схемы (MatchingEngine -> FeatureNode, FeatureAlias).
5. Запись типизированных фактов в разреженную матрицу (DocumentValue).
6. Компиляцию и кэширование итогового иерархического JSON (DocumentJsonCache),
   в точности соответствующего формату 2 - копия.json.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from src.config import logger, settings
from src.db import get_db_session, get_default_profile
from src.embeddings import BaseEmbeddingProvider, get_embedding_provider
from src.extractor import AdministrativeInfo, EntityExtractor, ExtractedDocument, ExtractedEntity
from src.matcher import MatchDecision, MatchingEngine
from src.models import (
    Document,
    DocumentJsonCache,
    DocumentProfile,
    DocumentValue,
    FeatureAlias,
    FeatureNode,
    Parameter,
    ParameterSynonym,
    PatientRecord,
)


def set_nested_dict_value(target: Dict[str, Any], path_parts: List[str], value: Any) -> None:
    """Установка значения по цепочке вложенных ключей в словаре."""
    cur = target
    for part in path_parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[path_parts[-1]] = value


def build_compiled_json(
    administrative: AdministrativeInfo,
    findings: List[Tuple[str, str, Any]],
    conclusions: List[str],
) -> Dict[str, Any]:
    """Сборка эталонного иерархического документа (соответствует формату 2 - копия.json).

    Args:
        administrative: Административная шапка документа.
        findings: Список троек (full_path, data_type, value).
        conclusions: Список строк заключений/диагнозов.
    """
    compiled: Dict[str, Any] = {
        "institution": administrative.institution,
        "patient": administrative.patient,
        "age": administrative.age,
        "date_of_birth": administrative.date_of_birth,
        "address": administrative.address,
        "consultation_date": administrative.consultation_date,
        "examination": administrative.examination,
    }

    # Вложенные блоки органов и анатомических структур
    for full_path, data_type, value in findings:
        if full_path in ("conclusion", "administrative"):
            continue
        parts = full_path.split(".")
        if data_type == "integer":
            try:
                val = int(value)
            except (ValueError, TypeError):
                val = value
        elif data_type == "float":
            try:
                val = float(value)
            except (ValueError, TypeError):
                val = value
        else:
            val = value

        set_nested_dict_value(compiled, parts, val)

    # Заключение, врач и специальность в конце документа
    compiled["conclusion"] = conclusions
    compiled["doctor"] = administrative.doctor
    compiled["specialization"] = administrative.specialization

    return compiled


class QuestionnaireProcessor:
    """Сервис обработки медицинских протоколов (листов консультативного осмотра) и развития онтологии."""

    def __init__(
        self,
        extractor: Optional[EntityExtractor] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        profile_code: str = "us_abdomen",
    ):
        self.extractor = extractor or EntityExtractor()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.profile_code = profile_code
        logger.info(f"QuestionnaireProcessor успешно инициализирован (профиль: {self.profile_code}).")

    def process_file(
        self,
        file_path: str,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        session: Optional[Session] = None,
        save_json: bool = True,
    ) -> Dict[str, Any]:
        """Чтение текстового файла протокола (например, '1 copy.txt') и запуск обработки."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Файл протокола не найден: {file_path}")

        # Чтение файла с поддержкой utf-8 и fallback на cp1251
        text_content = ""
        for encoding in ("utf-8", "cp1251", "utf-8-sig"):
            try:
                with open(p, "r", encoding=encoding) as f:
                    text_content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if not text_content:
            raise ValueError(f"Не удалось прочитать содержимое файла {file_path} (пустой или неподдерживаемая кодировка).")

        logger.info(f"Загружен файл протокола: {p.name} ({len(text_content)} симв.)")
        return self.process_questionnaire(
            text=text_content,
            patient_id=patient_id,
            patient_name=patient_name,
            session=session,
            save_json=save_json,
        )

    def process_document(
        self,
        text_or_path: str,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        session: Optional[Session] = None,
        save_json: bool = False,
    ) -> Dict[str, Any]:
        """Универсальная точка входа для одного анализа: принимает либо текст протокола, либо путь к файлу."""
        if os.path.exists(text_or_path) and os.path.isfile(text_or_path):
            return self.process_file(
                file_path=text_or_path,
                patient_id=patient_id,
                patient_name=patient_name,
                session=session,
                save_json=save_json,
            )
        return self.process_questionnaire(
            text=text_or_path,
            patient_id=patient_id,
            patient_name=patient_name,
            session=session,
            save_json=save_json,
        )

    def process_questionnaire(
        self,
        text: str,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        session: Optional[Session] = None,
        save_json: bool = False,
    ) -> Dict[str, Any]:
        """Обработка одного протокола исследования (формат входных данных 'Лист консультативного осмотра'):

        1. Извлечение административной шапки (пациент, возраст, дата рождения, дата осмотра).
        2. Извлечение иерархических признаков органов и диагнозов.
        3. Регистрация документа (Document).
        4. Семантическое сопоставление с деревом признаков (MatchingEngine).
        5. Динамическое расширение схемы (Schema Evolution) при новых признаках.
        6. Сохранение типизированных фактов в DocumentValue.
        7. Компиляция и кэширование итогового JSON (DocumentJsonCache) по образцу 2 - копия.json.
        """
        logger.info(f"--- Начало обработки протокола (пациент: {patient_id or 'автоопределение'}) ---")

        # 1. Извлечение структурированного документа через LLM + детерминированный парсинг
        extracted_doc: ExtractedDocument = self.extractor.extract_document(
            text=text,
            patient_id=patient_id or "неизвестно",
        )

        # 2. Сохранение в рамках сессии
        if session is not None:
            return self._process_extracted_doc_in_session(
                session=session,
                text=text,
                extracted_doc=extracted_doc,
                patient_id=patient_id,
                patient_name=patient_name,
                save_json=save_json,
            )
        else:
            with get_db_session() as sess:
                return self._process_extracted_doc_in_session(
                    session=sess,
                    text=text,
                    extracted_doc=extracted_doc,
                    patient_id=patient_id,
                    patient_name=patient_name,
                    save_json=save_json,
                )

    def _parse_consultation_date(self, raw_date: Optional[str]) -> datetime:
        """Безопасный парсинг даты осмотра в datetime."""
        if not raw_date:
            return datetime.now()
        clean = raw_date.strip()
        for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d.%m.%Y", "%d. %m. %Y %H:%M"):
            try:
                return datetime.strptime(clean, fmt)
            except ValueError:
                continue
        return datetime.now()

    def _process_extracted_doc_in_session(
        self,
        session: Session,
        text: str,
        extracted_doc: ExtractedDocument,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        save_json: bool = False,
    ) -> Dict[str, Any]:
        """Внутренняя логика сохранения документа, фактов и компиляции JSON внутри активной сессии БД."""
        matcher = MatchingEngine(
            session=session,
            embedding_provider=self.embedding_provider,
            profile_code=self.profile_code,
        )

        admin = extracted_doc.administrative

        # Определение финальных идентификаторов пациента
        effective_patient_id = (
            patient_id
            or (admin.patient if admin.patient != "---" else None)
            or f"P-{int(datetime.now().timestamp())}"
        )
        effective_patient_name = (
            patient_name
            or (admin.patient if admin.patient != "---" else None)
        )
        consult_dt = self._parse_consultation_date(admin.consultation_date)

        # 1. Регистрация исходного документа в таблице documents
        document = Document(
            profile_id=matcher.profile.id,
            patient_id=str(effective_patient_id),
            patient_name=effective_patient_name,
            consultation_date=consult_dt,
            raw_text=text,
            administrative=admin.model_dump(),
        )
        session.add(document)
        session.flush()

        new_features_count = 0
        matched_count = 0
        processed_details = []
        compiled_findings: List[Tuple[str, str, Any]] = []

        # 2. Обработка и сохранение каждого извлеченного признака
        doc_values_by_feature: Dict[int, DocumentValue] = {}
        for entity in extracted_doc.findings:
            decision: MatchDecision = matcher.match_parameter(
                raw_key=entity.raw_key,
                raw_value=entity.raw_value,
                context=entity.context,
                path_hint=entity.path_hint,
                category=entity.category,
                unit=entity.unit,
                data_type=entity.data_type,
            )

            if decision.action in ("NEW", "NEW_FEATURE", "NEW_BRANCH"):
                new_features_count += 1
            else:
                matched_count += 1

            if decision.feature_id:
                # Предотвращение дубликатов (document_id, feature_id) как в текущей сессии, так и в БД
                existing_val = doc_values_by_feature.get(decision.feature_id)
                if not existing_val:
                    existing_val = (
                        session.query(DocumentValue)
                        .filter(
                            DocumentValue.document_id == document.id,
                            DocumentValue.feature_id == decision.feature_id,
                        )
                        .first()
                    )

                if not existing_val:
                    val_record = DocumentValue(
                        document_id=document.id,
                        feature_id=decision.feature_id,
                        value_numeric=decision.value_numeric,
                        value_text=decision.value_text,
                        value_json=decision.value_json,
                        raw_context=entity.context,
                        confidence=entity.confidence,
                        meta_data={
                            "raw_key": entity.raw_key,
                            "path_hint": entity.path_hint,
                            "action": decision.action,
                            "reason": decision.reason,
                            "similarity": round(decision.similarity, 4),
                        },
                    )
                    session.add(val_record)
                    session.flush()
                    doc_values_by_feature[decision.feature_id] = val_record
                else:
                    # Если у существующей записи нет значения, а у новой появилось — дополняем
                    if existing_val.value_numeric is None and decision.value_numeric is not None:
                        existing_val.value_numeric = decision.value_numeric
                    if existing_val.value_text is None and decision.value_text is not None:
                        existing_val.value_text = decision.value_text
                    if existing_val.value_json is None and decision.value_json is not None:
                        existing_val.value_json = decision.value_json
                    if entity.context and entity.context not in (existing_val.raw_context or ""):
                        existing_val.raw_context = (f"{existing_val.raw_context or ''}; {entity.context}").strip("; ")

            val_for_json = (
                decision.value_numeric
                if decision.value_numeric is not None
                else (decision.value_json if decision.value_json is not None else decision.value_text)
            )
            compiled_findings.append((decision.full_path, decision.data_type, val_for_json))

            processed_details.append(
                {
                    "raw_key": entity.raw_key,
                    "raw_value": entity.raw_value,
                    "action": decision.action,
                    "feature_id": decision.feature_id,
                    "full_path": decision.full_path,
                    "data_type": decision.data_type,
                    "similarity": round(decision.similarity, 4),
                    "reason": decision.reason,
                }
            )

        # 3. Сохранение заключений в DocumentValue
        if extracted_doc.conclusion:
            conclusion_node = matcher.ensure_feature_path(
                full_path="conclusion",
                data_type="array",
                display_name="Клиническое заключение",
                status="approved",
            )
            existing_conclusion = doc_values_by_feature.get(conclusion_node.id) or (
                session.query(DocumentValue)
                .filter(
                    DocumentValue.document_id == document.id,
                    DocumentValue.feature_id == conclusion_node.id,
                )
                .first()
            )
            if not existing_conclusion:
                conclusion_val = DocumentValue(
                    document_id=document.id,
                    feature_id=conclusion_node.id,
                    value_numeric=None,
                    value_text=None,
                    value_json=extracted_doc.conclusion,
                    raw_context="; ".join(extracted_doc.conclusion),
                    confidence=1.0,
                    meta_data={"action": "MATCH", "reason": "Клинические диагнозы и синдромы протокола"},
                )
                session.add(conclusion_val)
                session.flush()
                doc_values_by_feature[conclusion_node.id] = conclusion_val
            else:
                existing_conclusion.value_json = extracted_doc.conclusion
                existing_conclusion.raw_context = "; ".join(extracted_doc.conclusion)

        # 4. Компиляция итогового валидного JSON по эталону
        compiled_json = build_compiled_json(
            administrative=admin,
            findings=compiled_findings,
            conclusions=extracted_doc.conclusion,
        )

        # 5. Сохранение в кэш document_json_cache
        cache_entry = DocumentJsonCache(
            document_id=document.id,
            compiled_json=compiled_json,
        )
        session.add(cache_entry)

        # 6. Опциональное сохранение в локальный файл
        saved_file_path: Optional[str] = None
        if save_json:
            output_dir = Path("analysis_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(r"[^\w\-]+", "_", str(effective_patient_id))
            saved_file_path = str(output_dir / f"{safe_id}_normalized.json")
            with open(saved_file_path, "w", encoding="utf-8") as f:
                json.dump(compiled_json, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON-артефакт сохранен в файл: {saved_file_path}")

        logger.info(
            f"[УСПЕХ] Протокол {effective_patient_id} обработан: "
            f"создано {new_features_count} новых узлов схемы, "
            f"сопоставлено {matched_count} признаков (всего признаков: {len(extracted_doc.findings)})."
        )

        return {
            "document_id": document.id,
            "patient_id": effective_patient_id,
            "patient_name": effective_patient_name or "",
            "total_extracted": len(extracted_doc.findings),
            "new_parameters": new_features_count,
            "new_features": new_features_count,
            "synonyms_linked": matched_count,
            "matched_features": matched_count,
            "compiled_json": compiled_json,
            "saved_file": saved_file_path,
            "results": processed_details,
        }

    def process_excel_file(
        self,
        file_path: str = "input.xlsm",
        sheet_name: str = "Лист1",
        limit: Optional[int] = None,
        save_json: bool = False,
    ) -> List[Dict[str, Any]]:
        """TODO: Групповой анализ (batch processing) будет реализован на следующем этапе.

        В данный момент пайплайн сфокусирован на детерминированном структурировании
        одного клинического листа осмотра.
        """
        logger.info(
            f"TODO: Групповой анализ из файла {file_path} временно отключен. "
            f"Используйте process_document / process_file для обработки индивидуальных протоколов."
        )
        return []

    def get_ontology_summary(self) -> Dict[str, Any]:
        """Получение сводки состояния дерева онтологии признаков."""
        with get_db_session() as session:
            profile = get_default_profile(session)

            total_nodes = (
                session.query(FeatureNode)
                .filter(FeatureNode.profile_id == profile.id)
                .count()
            )
            leaf_nodes = (
                session.query(FeatureNode)
                .filter(
                    FeatureNode.profile_id == profile.id,
                    FeatureNode.is_leaf == True,
                )
                .count()
            )
            branch_nodes = total_nodes - leaf_nodes
            alias_count = (
                session.query(FeatureAlias)
                .join(FeatureNode, FeatureNode.id == FeatureAlias.feature_id)
                .filter(FeatureNode.profile_id == profile.id)
                .count()
            )
            doc_count = (
                session.query(Document)
                .filter(Document.profile_id == profile.id)
                .count()
            )
            val_count = (
                session.query(DocumentValue)
                .join(Document, Document.id == DocumentValue.document_id)
                .filter(Document.profile_id == profile.id)
                .count()
            )

            nodes = (
                session.query(FeatureNode)
                .filter(FeatureNode.profile_id == profile.id)
                .order_by(FeatureNode.full_path)
                .all()
            )

            tree_list = []
            for n in nodes:
                aliases = [a.alias_text for a in n.aliases]
                tree_list.append(
                    {
                        "id": n.id,
                        "parent_id": n.parent_id,
                        "key_name": n.key_name,
                        "full_path": n.full_path,
                        "display_name": n.display_name,
                        "data_type": n.data_type,
                        "unit": n.unit,
                        "is_leaf": n.is_leaf,
                        "status": n.status,
                        "aliases_count": len(aliases),
                        "aliases": aliases,
                        "records_count": len(n.values),
                    }
                )

            return {
                "profile_code": profile.code,
                "profile_name": profile.name,
                "total_features": total_nodes,
                "leaf_features": leaf_nodes,
                "branch_features": branch_nodes,
                "total_aliases": alias_count,
                "total_documents": doc_count,
                "total_values": val_count,
                "features": tree_list,
            }
