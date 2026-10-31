"""Модуль сопоставления клинических признаков (Matching Engine) с деревом онтологии.

Поддерживает:
1. Поиск по материализованному пути (full_path) и точным синонимам (feature_aliases).
2. Векторный Top-K поиск через HNSW/pgvector косинусного сходства.
3. LLM-арбитраж серой зоны (0.70 .. 0.90) для разрешения неоднозначности.
4. Динамическое расширение схемы (Schema Evolution): создание промежуточных веток и листовых узлов.
5. Нормализацию типов данных (числа, двухмерные размеры, строки, массивы).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import requests
from sqlalchemy import text
from sqlalchemy.orm import Session
from src.config import logger, settings
from src.embeddings import BaseEmbeddingProvider, get_embedding_provider
from src.models import DocumentProfile, FeatureAlias, FeatureNode


@dataclass
class CandidateMatch:
    """Кандидат из схемы онтологии, найденный при векторном или текстовом поиске."""

    feature_id: int
    full_path: str
    matched_text: str
    match_source: Literal["feature", "alias", "canonical", "synonym"]
    similarity: float

    # Свойства для обратной совместимости с legacy-кодом
    @property
    def parameter_id(self) -> int:
        return self.feature_id

    @property
    def canonical_name(self) -> str:
        return self.full_path


@dataclass
class MatchDecision:
    """Итоговое решение алгоритма сопоставления признака."""

    action: Literal["MATCH", "SYNONYM", "NEW", "CHILD_OF", "NEW_FEATURE", "NEW_BRANCH"]
    feature_id: Optional[int]
    full_path: str
    similarity: float
    reason: str
    data_type: str = "string"
    unit: Optional[str] = None
    value_numeric: Optional[float] = None
    value_text: Optional[str] = None
    value_json: Optional[Any] = None
    parent_id: Optional[int] = None
    candidates: List[CandidateMatch] = field(default_factory=list)

    # Свойства для обратной совместимости
    @property
    def parameter_id(self) -> Optional[int]:
        return self.feature_id

    @property
    def canonical_name(self) -> str:
        return self.full_path


# Базовый словарь соответствий частых анатомических терминов для нормализации путей
ANATOMICAL_KEY_MAPPING = {
    "печень": "liver",
    "правая_доля": "right_lobe",
    "правая доля": "right_lobe",
    "левая_доля": "left_lobe",
    "левая доля": "left_lobe",
    "паренхима": "parenchyma",
    "сосудистый_рисунок": "vascular_tissue",
    "желчные_протоки": "bile_ducts",
    "желчный_пузырь": "gallbladder",
    "желчный пузырь": "gallbladder",
    "стенка": "wall",
    "толщина_стенки": "wall",
    "размер": "size",
    "форма": "form",
    "поджелудочная_железа": "pancreas",
    "поджелудочная железа": "pancreas",
    "головка": "head",
    "тело": "body",
    "хвост": "tail",
    "структура": "structure",
    "селезенка": "spleen",
    "вена": "vein",
    "почка_правая": "right_kidney",
    "правая_почка": "right_kidney",
    "правая почка": "right_kidney",
    "почка_левая": "left_kidney",
    "левая_почка": "left_kidney",
    "левая почка": "left_kidney",
    "чашечка": "calyx",
    "лоханка": "pelvis",
    "свободная_жидкость": "free_liquid_in_abdomen",
    "лимфоузлы": "paraaortic_lymph_nodes",
    "заключение": "conclusion",
}


def normalize_value_types(
    value: Any,
    data_type: str,
) -> Tuple[Optional[float], Optional[str], Optional[Any]]:
    """Нормализация сырого значения под целевые типизированные колонки DocumentValue.

    Возвращает кортеж: (value_numeric, value_text, value_json)
    """
    if value is None:
        return None, None, None

    # Если значение — массив или сложный словарь (например, conclusion)
    if isinstance(value, (list, dict)):
        return None, None, value

    if data_type in ("integer", "float"):
        if isinstance(value, (int, float)):
            return float(value), None, None
        # Извлечение первого числа из строки
        val_str = str(value).replace(",", ".")
        match = re.search(r"[-+]?\d*\.?\d+", val_str)
        if match:
            try:
                num = float(match.group(0))
                return num, None, None
            except ValueError:
                pass
        return None, str(value).strip(), None

    if data_type == "dimension_2d":
        # Двухмерные габариты хранятся как каноническая строка ("65  х  28 мм")
        return None, str(value).strip(), None

    # По умолчанию — текстовая строка
    return None, str(value).strip(), None


class MatchingEngine:
    """Интеллектуальный механизм сопоставления параметров с деревом онтологии."""

    GENERIC_KEYS = {
        "size",
        "wall",
        "wall_thickness",
        "structure",
        "form",
        "pathology",
        "pain",
        "palpation",
        "head",
        "body",
        "tail",
        "lesions",
        "findings",
        "state",
        "status",
        "type",
        "function",
        "symptoms",
        "examination",
        "diameter",
        "length",
        "width",
        "volume",
        "condition",
        "presence",
        "right_lobe",
        "left_lobe",
        "parenchyma",
        "vascular_tissue",
        "bile_ducts",
        "calyx",
        "pelvis",
        "contour",
        "echogenicity",
        "capsule",
    }

    ARBITRATOR_SYSTEM_PROMPT = (
        "Ты — ведущий медицинский эксперт-онтолог. Твоя задача — сопоставить извлеченный из протокола параметр "
        "с существующим деревом онтологии признаков.\n"
        "Тебе дан исходный термин, предложенный путь, значение, контекст и список ближайших узлов онтологии.\n\n"
        "ПРАВИЛА АРБИТРАЖА:\n"
        "1. MATCH: если признак по клиническому смыслу точно соответствует существующему узлу "
        "(например: 'ЧСС' соответствует 'vitals.pulse', или 'толщина стенки ЖП' соответствует 'liver.gallbladder.wall').\n"
        "   Укажи action='MATCH', 'matched_feature_id' (ID узла) и 'full_path'.\n"
        "2. NEW_FEATURE: если признак является другим самостоятельным показателем или новой структурой "
        "(например: 'артериальное давление' НЕ является пульсом -> 'vitals.blood_pressure', или 'полип' -> 'liver.gallbladder.polyp').\n"
        "   Укажи action='NEW_FEATURE', 'full_path' (например: 'vitals.blood_pressure') и 'data_type'.\n"
        "3. NEW_BRANCH: если описан принципиально новый орган или функциональная система.\n\n"
        "Отвечай СТРОГО в формате JSON без markdown:\n"
        "{\n"
        '  "action": "MATCH",\n'
        '  "matched_feature_id": 14,\n'
        '  "full_path": "liver.gallbladder.wall",\n'
        '  "reason": "Толщина стенки желчного пузыря является синонимом wall"\n'
        "}\n"
        "ИЛИ:\n"
        "{\n"
        '  "action": "NEW_FEATURE",\n'
        '  "full_path": "vitals.blood_pressure",\n'
        '  "data_type": "string",\n'
        '  "reason": "Артериальное давление является отдельным показателем, отличным от пульса"\n'
        "}"
    )

    def __init__(
        self,
        session: Session,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        profile_code: str = "us_abdomen",
    ):
        self.session = session
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL

        # Получение или создание профиля документа
        self.profile = (
            self.session.query(DocumentProfile)
            .filter(DocumentProfile.code == profile_code)
            .first()
        )
        if not self.profile:
            self.profile = DocumentProfile(
                code=profile_code,
                name="Ультразвуковое исследование органов брюшной полости",
                description="Протокол УЗИ ОБП",
            )
            self.session.add(self.profile)
            self.session.flush()

    def _canonicalize_path(self, path_hint: Optional[str], raw_key: str, category: Optional[str]) -> str:
        """Построение стандартизированного пути (snake_case) с учетом анатомических соответствий."""
        if path_hint and "." in path_hint:
            raw_parts = path_hint.split(".")
        elif category and category != raw_key:
            raw_parts = [category, raw_key]
        else:
            raw_parts = [raw_key]

        norm_parts = []
        for part in raw_parts:
            clean_part = part.strip().lower()
            translated = ANATOMICAL_KEY_MAPPING.get(clean_part, clean_part)
            translated = re.sub(r"[^\w]+", "_", translated).strip("_")
            if translated:
                norm_parts.append(translated)

        return ".".join(norm_parts) if norm_parts else raw_key.lower().strip()

    def ensure_feature_path(
        self,
        full_path: str,
        data_type: str = "string",
        unit: Optional[str] = None,
        display_name: Optional[str] = None,
        status: str = "approved",
    ) -> FeatureNode:
        """Гарантирует существование полного пути в дереве feature_nodes (создает родителей при отсутствии)."""
        clean_path = full_path.strip().lower()

        # 1. Проверяем в текущей сессии среди еще не закоммиченных объектов
        for obj in self.session.new:
            if isinstance(obj, FeatureNode) and obj.profile_id == self.profile.id and obj.full_path == clean_path:
                return obj

        # 2. Проверяем в базе
        existing = (
            self.session.query(FeatureNode)
            .filter(
                FeatureNode.profile_id == self.profile.id,
                FeatureNode.full_path == clean_path,
            )
            .first()
        )
        if existing:
            return existing

        # 3. Рекурсивно создаем цепочку родителей
        parts = clean_path.split(".")
        parent_id: Optional[int] = None
        current_subpath = ""

        for i, part in enumerate(parts):
            current_subpath = f"{current_subpath}.{part}" if current_subpath else part
            is_target_leaf = i == len(parts) - 1

            # Ищем узел на данном уровне
            node = (
                self.session.query(FeatureNode)
                .filter(
                    FeatureNode.profile_id == self.profile.id,
                    FeatureNode.full_path == current_subpath,
                )
                .first()
            )

            if not node:
                # Проверяем среди новых в сессии
                for obj in self.session.new:
                    if isinstance(obj, FeatureNode) and obj.profile_id == self.profile.id and obj.full_path == current_subpath:
                        node = obj
                        break

            if not node:
                node_data_type = data_type if is_target_leaf else "object"
                node_unit = unit if is_target_leaf else None
                node_leaf = is_target_leaf
                node_display = display_name if is_target_leaf and display_name else part.replace("_", " ").capitalize()
                node_status = status if is_target_leaf else "approved"

                emb = self.embedding_provider.embed_text(f"{node_display} {current_subpath}")

                node = FeatureNode(
                    profile_id=self.profile.id,
                    parent_id=parent_id,
                    key_name=part,
                    full_path=current_subpath,
                    display_name=node_display,
                    data_type=node_data_type,
                    unit=node_unit,
                    is_leaf=node_leaf,
                    embedding=emb,
                    status=node_status,
                )
                self.session.add(node)
                self.session.flush()
                logger.info(f"Создан новый узел онтологии: ID {node.id} ('{current_subpath}')")

            parent_id = node.id

        return node

    def find_top_candidates(self, query_text: str, top_k: int = 3) -> List[CandidateMatch]:
        """Поиск кандидатов в feature_nodes и feature_aliases через HNSW/косинусное сходство."""
        clean_text = query_text.strip().lower()
        if not clean_text:
            return []

        # Прямое точное текстовое совпадение с full_path
        exact_node = (
            self.session.query(FeatureNode)
            .filter(
                FeatureNode.profile_id == self.profile.id,
                FeatureNode.full_path == clean_text,
            )
            .first()
        )
        if exact_node:
            return [
                CandidateMatch(
                    feature_id=exact_node.id,
                    full_path=exact_node.full_path,
                    matched_text=exact_node.full_path,
                    match_source="feature",
                    similarity=1.0,
                )
            ]

        # Прямое точное совпадение с alias
        exact_alias = (
            self.session.query(FeatureAlias)
            .join(FeatureNode, FeatureNode.id == FeatureAlias.feature_id)
            .filter(
                FeatureNode.profile_id == self.profile.id,
                FeatureAlias.alias_text.ilike(clean_text),
            )
            .first()
        )
        if exact_alias and exact_alias.feature:
            return [
                CandidateMatch(
                    feature_id=exact_alias.feature_id,
                    full_path=exact_alias.feature.full_path,
                    matched_text=exact_alias.alias_text,
                    match_source="alias",
                    similarity=1.0,
                )
            ]

        # Векторный поиск ближайших узлов
        try:
            vector = self.embedding_provider.embed_text(clean_text)
            vector_str = f"[{','.join(str(x) for x in vector)}]"

            sql = text(
                """
                WITH combined AS (
                    SELECT 
                        fn.id AS feature_id,
                        fn.full_path AS full_path,
                        fn.display_name AS matched_text,
                        'feature' AS match_source,
                        CAST(1.0 - (fn.embedding <=> CAST(:vec AS vector)) AS FLOAT) AS similarity
                    FROM feature_nodes fn
                    WHERE fn.profile_id = :profile_id 
                      AND fn.embedding IS NOT NULL
                      AND fn.is_leaf = true

                    UNION ALL

                    SELECT 
                        fa.feature_id AS feature_id,
                        fn.full_path AS full_path,
                        fa.alias_text AS matched_text,
                        'alias' AS match_source,
                        CAST(1.0 - (fa.embedding <=> CAST(:vec AS vector)) AS FLOAT) AS similarity
                    FROM feature_aliases fa
                    JOIN feature_nodes fn ON fn.id = fa.feature_id
                    WHERE fn.profile_id = :profile_id 
                      AND fa.embedding IS NOT NULL
                )
                SELECT feature_id, full_path, matched_text, match_source, similarity
                FROM combined
                ORDER BY similarity DESC
                LIMIT :top_k;
                """
            )
            rows = self.session.execute(
                sql,
                {"profile_id": self.profile.id, "vec": vector_str, "top_k": top_k},
            ).fetchall()

            return [
                CandidateMatch(
                    feature_id=row.feature_id,
                    full_path=row.full_path,
                    matched_text=row.matched_text,
                    match_source=row.match_source,
                    similarity=float(row.similarity),
                )
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"Ошибка векторного поиска в PostgreSQL: {e}. Используем текстовое сопоставление.")
            return []

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Вызов LLM-арбитра."""
        if self.provider == "ollama":
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
            payload = {
                "model": self.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_thread": 10,
                    "num_predict": 250,
                },
            }
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "{}")
        else:
            import litellm

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            if settings.OPENAI_API_KEY:
                kwargs["api_key"] = settings.OPENAI_API_KEY
            if settings.OPENAI_BASE_URL:
                kwargs["base_url"] = settings.OPENAI_BASE_URL

            res = litellm.completion(**kwargs)
            return res.choices[0].message.content

    def _arbitrate_with_llm(
        self,
        raw_key: str,
        path_hint: str,
        raw_value: Any,
        context: str,
        candidates: List[CandidateMatch],
    ) -> MatchDecision:
        """Разрешение серой зоны через LLM."""
        candidates_str = "\n".join(
            [
                f"- ID {c.feature_id}: путь '{c.full_path}', сходство: {c.similarity:.4f} (через '{c.matched_text}')"
                for c in candidates
            ]
        )
        prompt = (
            f"Входящий параметр: '{raw_key}' (подсказка пути: '{path_hint}')\n"
            f"Значение: '{raw_value}'\n"
            f"Контекст: '{context}'\n\n"
            f"Кандидаты из онтологии:\n{candidates_str}\n\n"
            f"Определи: MATCH (это синоним одного из кандидатов) или NEW_FEATURE (новый параметр схемы)."
        )

        llm_decision: Optional[Dict[str, Any]] = None
        try:
            raw_resp = self._call_llm(self.ARBITRATOR_SYSTEM_PROMPT, prompt)
            llm_decision = json.loads(raw_resp)
        except Exception as e:
            logger.warning(f"Ошибка вызова или парсинга LLM-арбитража: {e}")

        if llm_decision and isinstance(llm_decision, dict):
            action = str(llm_decision.get("action", "")).upper()
            matched_id = llm_decision.get("matched_feature_id")

            if action in ("MATCH", "SYNONYM") and matched_id:
                node = self.session.get(FeatureNode, int(matched_id))
                if node:
                    self._ensure_alias(node.id, raw_key)
                    return MatchDecision(
                        action="MATCH",
                        feature_id=node.id,
                        full_path=node.full_path,
                        data_type=node.data_type,
                        unit=node.unit,
                        similarity=candidates[0].similarity if candidates else 0.85,
                        reason=f"LLM-арбитраж: {llm_decision.get('reason', 'Признан синонимом')}",
                        candidates=candidates,
                    )
            elif action in ("NEW", "NEW_FEATURE", "NEW_BRANCH"):
                target_new_path = llm_decision.get("full_path")
                if not target_new_path and llm_decision.get("parent_path") and llm_decision.get("key_name"):
                    target_new_path = f"{llm_decision['parent_path']}.{llm_decision['key_name']}"
                if not target_new_path:
                    target_new_path = path_hint

                target_data_type = llm_decision.get("data_type") or "string"
                new_node = self.ensure_feature_path(target_new_path, data_type=target_data_type, status="candidate")
                if raw_key.lower().strip() != new_node.key_name.lower().strip():
                    self._ensure_alias(new_node.id, raw_key)
                return MatchDecision(
                    action="NEW_FEATURE",
                    feature_id=new_node.id,
                    full_path=new_node.full_path,
                    data_type=new_node.data_type,
                    unit=new_node.unit,
                    similarity=candidates[0].similarity if candidates else 0.0,
                    reason=f"LLM-арбитраж: {llm_decision.get('reason', 'Зарегистрирован как новый признак схемы')}",
                    candidates=candidates,
                )

        # Fallback при невозможности арбитража (LLM не ответила или вернула некорректный ответ)
        if candidates and candidates[0].similarity >= settings.SIMILARITY_THRESHOLD_HIGH:
            c = candidates[0]
            node = self.session.get(FeatureNode, c.feature_id)
            if node:
                return MatchDecision(
                    action="MATCH",
                    feature_id=node.id,
                    full_path=node.full_path,
                    data_type=node.data_type,
                    unit=node.unit,
                    similarity=c.similarity,
                    reason=f"Fallback арбитража: привязка к '{c.full_path}' ({c.similarity:.4f})",
                    candidates=candidates,
                )

        # Создаем новый узел по предложенному пути
        new_node = self.ensure_feature_path(path_hint, status="candidate")
        return MatchDecision(
            action="NEW_FEATURE",
            feature_id=new_node.id,
            full_path=new_node.full_path,
            data_type=new_node.data_type,
            unit=new_node.unit,
            similarity=candidates[0].similarity if candidates else 0.0,
            reason="Арбитраж (fallback): зарегистрирован как новый признак схемы",
            candidates=candidates,
        )

    def _ensure_alias(self, feature_id: int, alias_text: str) -> None:
        """Добавление формулировки в feature_aliases при отсутствии."""
        clean = alias_text.strip().lower()
        if not clean or len(clean) < 3 or clean in self.GENERIC_KEYS:
            return

        exists = (
            self.session.query(FeatureAlias)
            .join(FeatureNode, FeatureNode.id == FeatureAlias.feature_id)
            .filter(
                FeatureNode.profile_id == self.profile.id,
                FeatureAlias.alias_text.ilike(clean),
            )
            .first()
        )
        if not exists:
            emb = self.embedding_provider.embed_text(clean)
            alias = FeatureAlias(feature_id=feature_id, alias_text=clean, embedding=emb)
            self.session.add(alias)
            self.session.flush()

    def match_parameter(
        self,
        raw_key: str,
        raw_value: Any,
        context: str = "",
        path_hint: Optional[str] = None,
        category: Optional[str] = None,
        unit: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> MatchDecision:
        """Главный метод сопоставления извлеченного признака с онтологией."""
        # 1. Формируем канонический путь
        target_path = self._canonicalize_path(path_hint=path_hint, raw_key=raw_key, category=category)

        # 2. Проверяем точное совпадение по full_path
        exact_node = (
            self.session.query(FeatureNode)
            .filter(
                FeatureNode.profile_id == self.profile.id,
                FeatureNode.full_path == target_path,
            )
            .first()
        )
        if exact_node:
            num_val, text_val, json_val = normalize_value_types(raw_value, exact_node.data_type)
            if raw_key.lower().strip() != exact_node.key_name.lower().strip():
                self._ensure_alias(exact_node.id, raw_key)
            return MatchDecision(
                action="MATCH",
                feature_id=exact_node.id,
                full_path=exact_node.full_path,
                data_type=exact_node.data_type,
                unit=exact_node.unit,
                value_numeric=num_val,
                value_text=text_val,
                value_json=json_val,
                similarity=1.0,
                reason="Точное совпадение по пути онтологии",
                candidates=[],
            )

        # 3. Поиск ближайших кандидатов
        search_query = f"{target_path} {raw_key}"
        candidates = self.find_top_candidates(search_query, top_k=3)

        if candidates:
            best = candidates[0]
            # Высокая уверенность (>= 0.90)
            if best.similarity >= settings.SIMILARITY_THRESHOLD_HIGH:
                node = self.session.get(FeatureNode, best.feature_id)
                if node:
                    num_val, text_val, json_val = normalize_value_types(raw_value, node.data_type)
                    if raw_key.lower().strip() != node.key_name.lower().strip():
                        self._ensure_alias(node.id, raw_key)
                    return MatchDecision(
                        action="MATCH",
                        feature_id=node.id,
                        full_path=node.full_path,
                        data_type=node.data_type,
                        unit=node.unit,
                        value_numeric=num_val,
                        value_text=text_val,
                        value_json=json_val,
                        similarity=best.similarity,
                        reason=f"Высокое семантическое сходство ({best.similarity:.4f}) с '{best.matched_text}'",
                        candidates=candidates,
                    )

            # Серая зона (0.70 .. 0.90)
            if best.similarity >= settings.SIMILARITY_THRESHOLD_LOW:
                decision = self._arbitrate_with_llm(
                    raw_key=raw_key,
                    path_hint=target_path,
                    raw_value=raw_value,
                    context=context,
                    candidates=candidates,
                )
                num_val, text_val, json_val = normalize_value_types(raw_value, decision.data_type)
                decision.value_numeric = num_val
                decision.value_text = text_val
                decision.value_json = json_val
                return decision

        # 4. Низкое сходство или принципиально новый признак -> Динамическое расширение схемы
        determined_type = data_type or ("integer" if isinstance(raw_value, int) else "string")
        new_node = self.ensure_feature_path(
            full_path=target_path,
            data_type=determined_type,
            unit=unit,
            display_name=raw_key.replace("_", " ").capitalize(),
            status="candidate",
        )
        if raw_key.lower().strip() != new_node.key_name.lower().strip():
            self._ensure_alias(new_node.id, raw_key)

        num_val, text_val, json_val = normalize_value_types(raw_value, new_node.data_type)
        return MatchDecision(
            action="NEW_FEATURE",
            feature_id=new_node.id,
            full_path=new_node.full_path,
            data_type=new_node.data_type,
            unit=new_node.unit,
            value_numeric=num_val,
            value_text=text_val,
            value_json=json_val,
            similarity=candidates[0].similarity if candidates else 0.0,
            reason="Динамическое расширение онтологии: создан новый узел схемы",
            candidates=candidates,
        )
