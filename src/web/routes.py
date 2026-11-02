"""API маршруты и контроллеры для веб-интерфейса медицинской онтологии."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func
from src.config import logger
from src.db import get_db_session
from src.models import (
    Document,
    DocumentJsonCache,
    DocumentProfile,
    DocumentValue,
    FeatureAlias,
    FeatureNode,
)

router = APIRouter(prefix="/api")

FALLBACK_JSON_DIR = Path("_analysis_output")


def _get_nested_value(data: Any, path: str) -> Any:
    """Извлекает значение по точечному пути (например 'vitals.temperature') из словаря."""
    if not isinstance(data, dict):
        return None
    parts = path.split(".")
    curr = data
    for p in parts:
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
        else:
            return None
    return curr


def _load_fallback_questionnaires() -> List[Dict[str, Any]]:
    """Резервная загрузка анкет из локальных JSON-файлов в папке _analysis_output."""
    items = []
    if not FALLBACK_JSON_DIR.exists():
        return items

    for p in sorted(FALLBACK_JSON_DIR.glob("*_normalized.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Извлекаем patient_id из имени файла (например, "1_normalized.json" -> "1")
            stem_id = p.stem.replace("_normalized", "")
            items.append(
                {
                    "id": stem_id,
                    "patient_id": stem_id,
                    "patient_name": data.get("patient") or f"Пациент {stem_id}",
                    "consultation_date": data.get("consultation_date"),
                    "institution": data.get("institution"),
                    "doctor": data.get("doctor"),
                    "specialization": data.get("specialization"),
                    "examination": data.get("examination"),
                    "values_count": len(data),
                    "has_cached_json": True,
                    "source": "local_file",
                }
            )
        except Exception as e:
            logger.warning(f"Ошибка чтения локального JSON {p}: {e}")
    return items


def _load_fallback_questionnaire_json(doc_id: str) -> Optional[Dict[str, Any]]:
    """Резервная загрузка JSON конкретной анкеты из локального файла."""
    for candidate in [
        FALLBACK_JSON_DIR / f"{doc_id}_normalized.json",
        FALLBACK_JSON_DIR / f"{doc_id}.json",
    ]:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки резервного файла {candidate}: {e}")
    return None


# ============================================================================
# 1. СПИСОК АНКЕТ (QUESTIONNAIRES LIST)
# ============================================================================

@router.get("/documents")
@router.get("/questionnaires")
def get_questionnaires() -> List[Dict[str, Any]]:
    """Получить список всех анкет (документов) в системе с метаданными и количеством признаков."""
    try:
        with get_db_session() as session:
            docs = session.query(Document).order_by(Document.id).all()
            if not docs:
                return _load_fallback_questionnaires()

            # Подсчет количества извлеченных признаков на каждую анкету
            counts_query = (
                session.query(DocumentValue.document_id, func.count(DocumentValue.id))
                .group_by(DocumentValue.document_id)
                .all()
            )
            counts_map = {doc_id: cnt for doc_id, cnt in counts_query}

            results = []
            for d in docs:
                admin = d.administrative or {}
                # Извлекаем краткие сводные данные
                diag = None
                if d.json_cache and isinstance(d.json_cache.compiled_json, dict):
                    concl = d.json_cache.compiled_json.get("conclusion")
                    if isinstance(concl, list) and concl:
                        diag = concl[0]
                    elif isinstance(concl, str):
                        diag = concl

                results.append(
                    {
                        "id": d.id,
                        "patient_id": d.patient_id,
                        "patient_name": d.patient_name or admin.get("patient") or f"Пациент #{d.patient_id}",
                        "consultation_date": d.consultation_date.isoformat() if d.consultation_date else admin.get("consultation_date"),
                        "age": admin.get("age"),
                        "institution": admin.get("institution"),
                        "doctor": admin.get("doctor"),
                        "specialization": admin.get("specialization"),
                        "examination": admin.get("examination"),
                        "diagnosis_preview": diag or admin.get("diagnosis"),
                        "values_count": counts_map.get(d.id, len(d.values) if d.values else 0),
                        "has_cached_json": d.json_cache is not None,
                        "created_at": d.created_at.isoformat() if d.created_at else None,
                        "source": "database",
                    }
                )
            return results
    except Exception as e:
        logger.error(f"Ошибка получения списка анкет из БД: {e}")
        fallback = _load_fallback_questionnaires()
        if fallback:
            return fallback
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")


# ============================================================================
# 2. ПРОСМОТР АНКЕТЫ ПО ID (JSON VIEW)
# ============================================================================

@router.get("/documents/{doc_id}")
@router.get("/questionnaires/{doc_id}")
def get_questionnaire(doc_id: str) -> Dict[str, Any]:
    """Получить анкету и ее полный нормализованный JSON по ID."""
    try:
        # Пробуем найти в БД
        with get_db_session() as session:
            doc = None
            if doc_id.isdigit():
                doc = session.query(Document).filter(Document.id == int(doc_id)).first()
            if not doc:
                doc = session.query(Document).filter(Document.patient_id == doc_id).first()

            if doc:
                compiled = None
                if doc.json_cache and doc.json_cache.compiled_json:
                    compiled = doc.json_cache.compiled_json
                else:
                    # Попробуем резервный локальный файл
                    compiled = _load_fallback_questionnaire_json(doc.patient_id)

                admin = doc.administrative or {}
                return {
                    "id": doc.id,
                    "patient_id": doc.patient_id,
                    "patient_name": doc.patient_name or admin.get("patient") or f"Пациент #{doc.patient_id}",
                    "consultation_date": doc.consultation_date.isoformat() if doc.consultation_date else admin.get("consultation_date"),
                    "administrative": admin,
                    "raw_text": doc.raw_text,
                    "json": compiled or admin,
                    "values_count": len(doc.values) if doc.values else 0,
                    "source": "database",
                }
    except Exception as e:
        logger.warning(f"Ошибка чтения анкеты {doc_id} из БД: {e}")

    # Резервная загрузка из файлов
    file_json = _load_fallback_questionnaire_json(doc_id)
    if file_json:
        return {
            "id": doc_id,
            "patient_id": doc_id,
            "patient_name": file_json.get("patient", f"Пациент #{doc_id}"),
            "consultation_date": file_json.get("consultation_date"),
            "administrative": {
                "institution": file_json.get("institution"),
                "doctor": file_json.get("doctor"),
                "specialization": file_json.get("specialization"),
                "age": file_json.get("age"),
            },
            "raw_text": None,
            "json": file_json,
            "values_count": len(file_json),
            "source": "local_file",
        }

    raise HTTPException(status_code=404, detail=f"Анкета с ID '{doc_id}' не найдена.")


@router.get("/documents/{doc_id}/raw-json")
@router.get("/questionnaires/{doc_id}/raw-json")
def get_questionnaire_raw_json(doc_id: str):
    """Возвращает чистый JSON анкеты без обертки для прямого скачивания/просмотра."""
    data = get_questionnaire(doc_id)
    return JSONResponse(
        content=data.get("json", {}),
        headers={"Content-Disposition": f'inline; filename="questionnaire_{doc_id}.json"'},
    )


# ============================================================================
# 3. ОНТОЛОГИЯ ПРИЗНАКОВ В ВИДЕ ГРАФА (GRAPH VISUALIZATION)
# ============================================================================

# Цветовая палитра для типов узлов графа
NODE_COLORS = {
    "root": {"background": "#4f46e5", "border": "#3730a3", "highlight": "#6366f1"},
    "category": {"background": "#0284c7", "border": "#0369a1", "highlight": "#38bdf8"},
    "subcategory": {"background": "#0d9488", "border": "#0f766e", "highlight": "#2dd4bf"},
    "float": {"background": "#10b981", "border": "#059669", "highlight": "#34d399"},
    "integer": {"background": "#14b8a6", "border": "#0d9488", "highlight": "#2dd4bf"},
    "string": {"background": "#f59e0b", "border": "#d97706", "highlight": "#fbbf24"},
    "object": {"background": "#6366f1", "border": "#4f46e5", "highlight": "#818cf8"},
    "array": {"background": "#8b5cf6", "border": "#7c3aed", "highlight": "#a78bfa"},
    "default": {"background": "#64748b", "border": "#475569", "highlight": "#94a3b8"},
}


@router.get("/ontology/graph")
def get_ontology_graph(
    include_root: bool = Query(True, description="Включить центральный корневой узел онтологии"),
    leaf_only: bool = Query(False, description="Возвращать только листовые признаки"),
) -> Dict[str, Any]:
    """Возвращает узлы (nodes) и связи (edges) для интерактивной визуализации графа (Vis.js)."""
    try:
        with get_db_session() as session:
            nodes_query = session.query(FeatureNode).order_by(FeatureNode.id).all()
            if not nodes_query:
                return _build_fallback_graph()

            # Подсчет количества заполненных значений в анкетах для каждого признака
            counts_query = (
                session.query(DocumentValue.feature_id, func.count(DocumentValue.id))
                .group_by(DocumentValue.feature_id)
                .all()
            )
            records_count_map = {fid: cnt for fid, cnt in counts_query}
            total_docs = session.query(Document).count()

            nodes = []
            edges = []

            # 1. Центральный корневой узел
            root_id = "root_0"
            if include_root:
                nodes.append(
                    {
                        "id": root_id,
                        "label": "Медицинская онтология\n(Клинический протокол)",
                        "full_path": "root",
                        "display_name": "Медицинская онтология",
                        "data_type": "root",
                        "is_leaf": False,
                        "records_count": total_docs,
                        "shape": "box",
                        "margin": 12,
                        "font": {"color": "#ffffff", "size": 15, "bold": True},
                        "color": NODE_COLORS["root"],
                        "shadow": {"enabled": True, "size": 10, "color": "rgba(0,0,0,0.3)"},
                        "level": 0,
                        "title": f"Корневой профиль онтологии ({total_docs} анкет в базе)",
                    }
                )

            # Вычисление уровней иерархии (level) для древовидного макета
            node_map = {n.id: n for n in nodes_query}

            def get_depth(node: FeatureNode, depth: int = 1) -> int:
                if node.parent_id and node.parent_id in node_map:
                    return get_depth(node_map[node.parent_id], depth + 1)
                return depth

            for n in nodes_query:
                if leaf_only and not n.is_leaf:
                    continue

                recs = records_count_map.get(n.id, 0)
                depth = get_depth(n)

                # Определение типа и стиля узла
                if not n.is_leaf:
                    node_type = "category" if depth <= 2 else "subcategory"
                    shape = "box"
                    font = {"color": "#ffffff", "size": 13, "bold": True}
                    size = 28
                else:
                    node_type = n.data_type if n.data_type in NODE_COLORS else "default"
                    shape = "dot"
                    font = {"color": "#1e293b", "size": 11}
                    size = 18 + min(recs * 2, 16)  # Размер зависит от количества заполненных анкет

                color = NODE_COLORS.get(node_type, NODE_COLORS["default"])

                # Всплывающая подсказка (HTML)
                tooltip = (
                    f"<div style='font-family: sans-serif; min-width: 180px;'>"
                    f"<b style='font-size: 13px;'>{n.display_name or n.key_name}</b><br/>"
                    f"<span style='color: #64748b; font-size: 11px;'>Путь:</span> <code>{n.full_path}</code><br/>"
                    f"<span style='color: #64748b; font-size: 11px;'>Тип:</span> <b>{n.data_type}</b>"
                    f"{f' ({n.unit})' if n.unit else ''}<br/>"
                    f"<span style='color: #64748b; font-size: 11px;'>Заполнено:</span> <b>{recs} из {total_docs} анкет</b><br/>"
                    f"<div style='margin-top: 5px; color: #2563eb; font-size: 11px;'>👉 Кликните для просмотра значений по всем анкетам</div>"
                    f"</div>"
                )

                nodes.append(
                    {
                        "id": n.id,
                        "label": n.display_name if n.display_name else n.key_name,
                        "full_path": n.full_path,
                        "key_name": n.key_name,
                        "display_name": n.display_name,
                        "data_type": n.data_type,
                        "unit": n.unit,
                        "is_leaf": n.is_leaf,
                        "records_count": recs,
                        "total_documents": total_docs,
                        "shape": shape,
                        "size": size,
                        "font": font,
                        "color": color,
                        "level": depth,
                        "title": tooltip,
                        "shadow": {"enabled": True, "size": 6, "color": "rgba(0,0,0,0.15)"},
                    }
                )

                # Связи (Edges)
                if n.parent_id and n.parent_id in node_map:
                    edges.append(
                        {
                            "from": n.parent_id,
                            "to": n.id,
                            "arrows": "to",
                            "color": {"color": "#94a3b8", "highlight": "#2563eb", "opacity": 0.7},
                            "width": 1.5,
                        }
                    )
                elif include_root and n.parent_id is None:
                    # Связь от центрального корневого узла
                    edges.append(
                        {
                            "from": root_id,
                            "to": n.id,
                            "arrows": "to",
                            "color": {"color": "#cbd5e1", "highlight": "#4f46e5", "opacity": 0.8},
                            "width": 2,
                        }
                    )

            return {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "total_nodes": len(nodes_query),
                    "leaf_nodes": sum(1 for n in nodes_query if n.is_leaf),
                    "branch_nodes": sum(1 for n in nodes_query if not n.is_leaf),
                    "total_documents": total_docs,
                },
            }
    except Exception as e:
        logger.error(f"Ошибка построения графа онтологии из БД: {e}")
        return _build_fallback_graph()


def _build_fallback_graph() -> Dict[str, Any]:
    """Резервное построение графа признаков на основе локальных JSON файлов."""
    fallback_docs = _load_fallback_questionnaires()
    feature_paths = set()
    for d in fallback_docs:
        json_data = _load_fallback_questionnaire_json(d["id"]) or {}

        def _traverse(obj: Any, prefix: str = ""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    path = f"{prefix}.{k}" if prefix else k
                    feature_paths.add(path)
                    _traverse(v, path)

        _traverse(json_data)

    nodes = []
    edges = []
    root_id = "root_0"
    nodes.append(
        {
            "id": root_id,
            "label": "Медицинская онтология (Файлы)",
            "full_path": "root",
            "display_name": "Медицинская онтология",
            "data_type": "root",
            "is_leaf": False,
            "records_count": len(fallback_docs),
            "shape": "box",
            "color": NODE_COLORS["root"],
            "level": 0,
        }
    )

    id_counter = 1
    path_to_id = {}
    for p in sorted(feature_paths):
        path_to_id[p] = id_counter
        id_counter += 1

    for path, nid in path_to_id.items():
        parts = path.split(".")
        name = parts[-1]
        parent_path = ".".join(parts[:-1]) if len(parts) > 1 else None

        nodes.append(
            {
                "id": nid,
                "label": name,
                "full_path": path,
                "key_name": name,
                "display_name": name,
                "data_type": "string",
                "is_leaf": True,
                "records_count": len(fallback_docs),
                "total_documents": len(fallback_docs),
                "shape": "dot",
                "color": NODE_COLORS["string"],
                "level": len(parts),
            }
        )
        if parent_path and parent_path in path_to_id:
            edges.append({"from": path_to_id[parent_path], "to": nid, "arrows": "to"})
        else:
            edges.append({"from": root_id, "to": nid, "arrows": "to"})

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "leaf_nodes": len(nodes) - 1,
            "branch_nodes": 0,
            "total_documents": len(fallback_docs),
        },
    }


# ============================================================================
# 4. МИНИ-ТАБЛИЦА ЗНАЧЕНИЙ ПРИЗНАКА У ВСЕХ АНКЕТ И МЕТАДАННЫЕ
# ============================================================================

@router.get("/ontology/nodes/{node_id}/values")
def get_node_values_across_questionnaires(node_id: str) -> Dict[str, Any]:
    """Возвращает мини-таблицу со значением выбранного узла онтологии у ВСЕХ анкет,

    включая пары 'признак - метаданные' (document_id, confidence, action, similarity, raw_context и т.д.).
    """
    try:
        with get_db_session() as session:
            # 1. Поиск узла онтологии
            node = None
            if node_id.isdigit():
                node = session.query(FeatureNode).filter(FeatureNode.id == int(node_id)).first()
            if not node:
                node = session.query(FeatureNode).filter(FeatureNode.full_path == node_id).first()

            if not node:
                raise HTTPException(status_code=404, detail=f"Узел онтологии '{node_id}' не найден.")

            # 2. Получение всех анкет (документов) в системе
            docs = session.query(Document).order_by(Document.id).all()
            total_docs = len(docs)

            # 3. Получение всех сохраненных значений (DocumentValue) для данного узла
            values = (
                session.query(DocumentValue)
                .filter(DocumentValue.feature_id == node.id)
                .all()
            )
            val_by_doc_id = {v.document_id: v for v in values}

            # 4. Синонимы и псевдонимы признака
            aliases = [a.alias_text for a in node.aliases] if node.aliases else []

            # 5. Сбор информации по каждой анкете
            doc_rows = []
            filled_count = 0
            numeric_values: List[float] = []

            for doc in docs:
                admin = doc.administrative or {}
                doc_val = val_by_doc_id.get(doc.id)

                value_display = None
                value_raw = None
                confidence = None
                meta_data: Dict[str, Any] = {}
                raw_context = None
                is_present = False

                if doc_val is not None:
                    is_present = True
                    filled_count += 1
                    confidence = round(doc_val.confidence, 3) if doc_val.confidence is not None else 1.0
                    raw_context = doc_val.raw_context
                    meta_data = dict(doc_val.meta_data or {})

                    # Извлечение типизированного значения
                    if doc_val.value_numeric is not None:
                        value_raw = doc_val.value_numeric
                        value_display = f"{doc_val.value_numeric}{f' {node.unit}' if node.unit else ''}"
                        numeric_values.append(doc_val.value_numeric)
                    elif doc_val.value_text is not None:
                        value_raw = doc_val.value_text
                        value_display = doc_val.value_text
                    elif doc_val.value_json is not None:
                        value_raw = doc_val.value_json
                        if isinstance(doc_val.value_json, list):
                            value_display = ", ".join(str(x) for x in doc_val.value_json)
                        elif isinstance(doc_val.value_json, dict):
                            value_display = json.dumps(doc_val.value_json, ensure_ascii=False)
                        else:
                            value_display = str(doc_val.value_json)
                    else:
                        value_display = "—"
                else:
                    # Резервная проверка в кэше скомпилированного JSON
                    if doc.json_cache and isinstance(doc.json_cache.compiled_json, dict):
                        cached_val = _get_nested_value(doc.json_cache.compiled_json, node.full_path)
                        if cached_val is not None:
                            is_present = True
                            filled_count += 1
                            value_raw = cached_val
                            if isinstance(cached_val, (int, float)):
                                numeric_values.append(float(cached_val))
                                value_display = f"{cached_val}{f' {node.unit}' if node.unit else ''}"
                            elif isinstance(cached_val, list):
                                value_display = ", ".join(str(x) for x in cached_val)
                            elif isinstance(cached_val, dict):
                                value_display = json.dumps(cached_val, ensure_ascii=False)
                            else:
                                value_display = str(cached_val)
                            meta_data["source"] = "json_cache"

                # Дополняем стандартными метаданными для таблицы
                standard_metadata = {
                    "document_id": doc.id,
                    "patient_id": doc.patient_id,
                    "feature_id": node.id,
                    "feature_path": node.full_path,
                }
                if confidence is not None:
                    standard_metadata["confidence"] = confidence
                # Объединяем со специфическими метаданными извлечения
                full_metadata = {**standard_metadata, **meta_data}

                doc_rows.append(
                    {
                        "document_id": doc.id,
                        "patient_id": doc.patient_id,
                        "patient_name": doc.patient_name or admin.get("patient") or f"Пациент #{doc.patient_id}",
                        "consultation_date": doc.consultation_date.isoformat() if doc.consultation_date else admin.get("consultation_date"),
                        "doctor": admin.get("doctor", "—"),
                        "institution": admin.get("institution", "—"),
                        "is_present": is_present,
                        "value": value_display if is_present else None,
                        "value_raw": value_raw,
                        "confidence": confidence,
                        "raw_context": raw_context,
                        "metadata": full_metadata,
                    }
                )

            # Вычисление базовой статистики по признаку
            fill_pct = round((filled_count / total_docs * 100), 1) if total_docs > 0 else 0
            stats = {
                "total_documents": total_docs,
                "filled_documents": filled_count,
                "missing_documents": total_docs - filled_count,
                "fill_percentage": fill_pct,
                "numeric_min": min(numeric_values) if numeric_values else None,
                "numeric_max": max(numeric_values) if numeric_values else None,
                "numeric_avg": round(sum(numeric_values) / len(numeric_values), 2) if numeric_values else None,
            }

            # Если узел является веткой (не листом), возвращаем список его дочерних узлов
            children_info = []
            if not node.is_leaf:
                children = (
                    session.query(FeatureNode)
                    .filter(FeatureNode.parent_id == node.id)
                    .order_by(FeatureNode.key_name)
                    .all()
                )
                children_info = [
                    {
                        "id": c.id,
                        "key_name": c.key_name,
                        "full_path": c.full_path,
                        "display_name": c.display_name,
                        "data_type": c.data_type,
                        "unit": c.unit,
                        "is_leaf": c.is_leaf,
                    }
                    for c in children
                ]

            return {
                "node": {
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "key_name": node.key_name,
                    "full_path": node.full_path,
                    "display_name": node.display_name,
                    "description": node.description,
                    "data_type": node.data_type,
                    "unit": node.unit,
                    "is_leaf": node.is_leaf,
                    "status": node.status,
                    "aliases": aliases,
                    "metadata": node.meta_data or {},
                },
                "children": children_info,
                "stats": stats,
                "documents": doc_rows,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения значений узла {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
