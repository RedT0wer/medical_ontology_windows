"""Модуль алгоритма сопоставления (Matching Engine) с гибридным поиском и LLM-арбитражем."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import requests
from sqlalchemy import text
from sqlalchemy.orm import Session
from src.config import logger, settings
from src.embeddings import BaseEmbeddingProvider, get_embedding_provider
from src.models import Parameter, ParameterSynonym


@dataclass
class CandidateMatch:
    """Кандидат из онтологии, найденный при векторном поиске."""

    parameter_id: int
    canonical_name: str
    matched_text: str
    match_source: Literal["canonical", "synonym"]
    similarity: float


@dataclass
class MatchDecision:
    """Итоговое решение алгоритма сопоставления."""

    action: Literal["SYNONYM", "NEW", "CHILD_OF", "NEW_PARENT_AND_CHILD"]
    parameter_id: Optional[int]
    canonical_name: str
    similarity: float
    reason: str
    parent_id: Optional[int] = None
    new_parent_name: Optional[str] = None
    candidates: List[CandidateMatch] = field(default_factory=list)


# Базовый словарь клинических нормализаций для ускорения формулирования канонических имен
CLINICAL_NORMALIZATIONS = {
    "ад": "Артериальное давление",
    "артериальное давление": "Артериальное давление",
    "давление": "Артериальное давление",
    "кровяное давление": "Артериальное давление",
    "чсс": "Частота сердечных сокращений",
    "частота сердечных сокращений": "Частота сердечных сокращений",
    "пульс": "Частота сердечных сокращений",
    "чдд": "Частота дыхательных движений",
    "частота дыхания": "Частота дыхательных движений",
    "температура": "Температура тела",
    "температура тела": "Температура тела",
    "spo2": "Сатурация кислорода SpO2",
    "сатурация": "Сатурация кислорода SpO2",
    "сатурация кислорода": "Сатурация кислорода SpO2",
    "цефалгия": "Головная боль",
    "головная боль": "Головная боль",
    "гемоглобин": "Уровень гемоглобина крови",
    "рвота": "Рвота",
    "слабость": "Общая слабость",
    "головокружение": "Головокружение",
}


class MatchingEngine:
    """Механизм сопоставления параметров (Top-3 поиск + пороговые правила + LLM-арбитр с иерархией)."""

    ARBITRATOR_SYSTEM_PROMPT = (
        "Ты — ведущий медицинский эксперт-онтолог. Твоя задача — арбитраж клинического термина из анкеты.\n"
        "Тебе дан исходный параметр ('raw_key') со значением и контекстом, а также список кандидатов из онтологии.\n\n"
        "ПРАВИЛА КЛИНИЧЕСКОГО АРБИТРАЖА (ИЕРАРХИЯ, РОДИТЕЛЬСКИЕ КЛАССЫ И СИНОНИМЫ):\n"
        "1. SYNONYM (полный синоним): если сырой параметр означает абсолютно то же самое, "
        "что существующий кандидат (например: 'АД' = 'Артериальное давление', 'пульс' = 'Частота сердечных сокращений', "
        "'цефалгия' = 'Головная боль'). Укажи action='SYNONYM' и 'matched_parameter_id'.\n"
        "2. CHILD_OF (подкласс существующего родителя): если термин является уточнением/подклассом (например, 'Боли в эпигастрии'), "
        "и подходящий общий родитель ('Боли') ЕСТЬ среди кандидатов. "
        "Укажи action='CHILD_OF', 'parent_parameter_id' (ID родителя), и 'canonical_name' (название подкласса).\n"
        "3. NEW_PARENT_AND_CHILD (создать новый родительский класс и подкласс): если термин является анатомическим/клиническим уточнением "
        "(например: 'тяжесть в эпигастрии', 'боли в колене', 'чувство тяжести в правом подреберье'), "
        "но общего родительского класса ('Тяжесть', 'Боли') в онтологии еще НЕТ (среди кандидатов только чужие симптомы вроде 'Отрыжка' или другие частные подклассы). "
        "НЕ привязывай к неподходящим кандидатам! Укажи action='NEW_PARENT_AND_CHILD', 'new_parent_name' (например: 'Тяжесть') и 'canonical_name' (например: 'Тяжесть в эпигастрии').\n"
        "4. NEW: если параметр принципиально новый, самостоятельный и не требует родителя (например, 'Тошнота', 'Одышка'). "
        "Укажи action='NEW' и сформулируй 'canonical_name'.\n\n"
        "Отвечай СТРОГО в формате JSON без markdown:\n"
        "{\n"
        '  "action": "NEW_PARENT_AND_CHILD",\n'
        '  "new_parent_name": "Тяжесть",\n'
        '  "parent_parameter_id": null,\n'
        '  "matched_parameter_id": null,\n'
        '  "canonical_name": "Тяжесть в эпигастрии",\n'
        '  "reason": "Выделение нового родительского симптома \'Тяжесть\' и анатомического подкласса \'Тяжесть в эпигастрии\'"\n'
        "}"
    )

    def __init__(self, session: Session, embedding_provider: Optional[BaseEmbeddingProvider] = None):
        self.session = session
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL

    def find_top_candidates(self, raw_key: str, top_k: int = 3) -> List[CandidateMatch]:
        """Поиск Top-K ближайших записей в таблицах parameters и parameter_synonyms."""
        raw_key_clean = raw_key.strip()
        if not raw_key_clean:
            return []

        # 1. Прямая проверка точного совпадения по тексту (без учета регистра)
        exact_param = (
            self.session.query(Parameter)
            .filter(Parameter.canonical_name.ilike(raw_key_clean))
            .first()
        )
        if exact_param:
            return [
                CandidateMatch(
                    parameter_id=exact_param.id,
                    canonical_name=exact_param.canonical_name,
                    matched_text=exact_param.canonical_name,
                    match_source="canonical",
                    similarity=1.0,
                )
            ]

        exact_syn = (
            self.session.query(ParameterSynonym)
            .filter(ParameterSynonym.synonym_text.ilike(raw_key_clean))
            .first()
        )
        if exact_syn:
            canonical_name = exact_syn.parameter.canonical_name if exact_syn.parameter else raw_key_clean
            return [
                CandidateMatch(
                    parameter_id=exact_syn.parameter_id,
                    canonical_name=canonical_name,
                    matched_text=exact_syn.synonym_text,
                    match_source="synonym",
                    similarity=1.0,
                )
            ]

        # 2. Вычисление эмбеддинга
        vector = self.embedding_provider.embed_text(raw_key_clean)
        vector_str = f"[{','.join(str(x) for x in vector)}]"

        # 3. Векторный поиск Top-K через HNSW с косинусным сходством
        query = text(
            """
            WITH combined AS (
                SELECT 
                    p.id AS parameter_id,
                    p.canonical_name AS canonical_name,
                    p.canonical_name AS matched_text,
                    'canonical' AS match_source,
                    CAST(1.0 - (p.embedding <=> CAST(:vec AS vector)) AS FLOAT) AS similarity
                FROM parameters p
                WHERE p.embedding IS NOT NULL

                UNION ALL

                SELECT 
                    s.parameter_id AS parameter_id,
                    p.canonical_name AS canonical_name,
                    s.synonym_text AS matched_text,
                    'synonym' AS match_source,
                    CAST(1.0 - (s.embedding <=> CAST(:vec AS vector)) AS FLOAT) AS similarity
                FROM parameter_synonyms s
                JOIN parameters p ON p.id = s.parameter_id
                WHERE s.embedding IS NOT NULL
            )
            SELECT parameter_id, canonical_name, matched_text, match_source, similarity
            FROM combined
            ORDER BY similarity DESC
            LIMIT :top_k;
            """
        )

        rows = self.session.execute(query, {"vec": vector_str, "top_k": top_k}).fetchall()

        candidates = []
        for row in rows:
            candidates.append(
                CandidateMatch(
                    parameter_id=row.parameter_id,
                    canonical_name=row.canonical_name,
                    matched_text=row.matched_text,
                    match_source=row.match_source,
                    similarity=float(row.similarity),
                )
            )

        return candidates

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Вызов LLM с многопоточностью и валидацией JSON."""
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
            resp = requests.post(url, json=payload, timeout=300)
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

    def _formulate_canonical(self, raw_key: str, raw_value: str, context: str) -> Tuple[str, str, Dict[str, Any]]:
        """Быстрое и надежное формулирование канонического названия для нового параметра."""
        clean_key = raw_key.strip().lower()
        # Проверяем словарь частых клинических нормализаций
        for key_pattern, canonical_val in CLINICAL_NORMALIZATIONS.items():
            if key_pattern in clean_key:
                return (
                    canonical_val,
                    f"Клинический показатель: {canonical_val}",
                    {"category": "Клинические показатели"},
                )

        # Стандартная очистка названия
        cleaned = re.sub(r"\s+", " ", raw_key.strip()).capitalize()
        return cleaned, f"Клинический показатель: {cleaned}", {"category": "Клинические показатели"}

    def _arbitrate_with_llm(
        self,
        raw_key: str,
        raw_value: str,
        context: str,
        candidates: List[CandidateMatch],
    ) -> MatchDecision:
        """Вызов LLM-арбитра для разрешения неоднозначности в серой зоне (0.70 .. 0.90)."""
        candidates_info = "\n".join(
            [
                f"- ID {c.parameter_id}: '{c.canonical_name}' (найден через {c.match_source} '{c.matched_text}', сходство: {c.similarity:.4f})"
                for c in candidates
            ]
        )
        user_prompt = (
            f"Входящий сырой параметр: '{raw_key}'\n"
            f"Значение: '{raw_value}'\n"
            f"Контекст: '{context}'\n\n"
            f"Кандидаты из онтологии:\n{candidates_info}\n\n"
            f"Определи действие: SYNONYM (полный синоним существующего ID), "
            f"CHILD_OF (клинический подкласс/уточнение существующего ID) или NEW."
        )

        logger.info(f"Запуск LLM-арбитра для параметра '{raw_key}' (серая зона)...")
        try:
            raw_response = self._call_llm(self.ARBITRATOR_SYSTEM_PROMPT, user_prompt)
            data = json.loads(raw_response)
            action = data.get("action", "NEW").upper()
            matched_id = data.get("matched_parameter_id")
            parent_id = data.get("parent_parameter_id")
            new_parent_name = data.get("new_parent_name")
            if action == "CHILD_OF" and not parent_id and matched_id:
                parent_id = matched_id
            reason = data.get("reason", "Решение LLM-арбитра")
            canonical_name = data.get("canonical_name")

            # Интерактивное модальное окно подтверждения экспертом (Human-in-the-loop)
            user_verdict = {}
            if settings.HUMAN_IN_THE_LOOP:
                from src.arbitration_modal import render_cli_arbitration_modal

                if action == "SYNONYM" and matched_id:
                    p = self.session.get(Parameter, int(matched_id))
                    default_c_name = canonical_name or (p.canonical_name if p else raw_key.capitalize())
                elif action in ("CHILD_OF", "NEW_PARENT_AND_CHILD"):
                    default_c_name = canonical_name or raw_key.capitalize()
                else:
                    default_c_name = canonical_name or raw_key.capitalize()

                user_verdict = render_cli_arbitration_modal(
                    raw_key=raw_key,
                    raw_value=raw_value,
                    context=context,
                    candidates=candidates,
                    llm_action=action,
                    llm_target_id=int(matched_id) if matched_id else None,
                    llm_parent_id=int(parent_id) if parent_id else None,
                    llm_new_parent_name=new_parent_name,
                    llm_canonical_name=default_c_name,
                    llm_reason=reason,
                )
                action = user_verdict["action"]
                matched_id = user_verdict.get("parameter_id")
                parent_id = user_verdict.get("parent_id") or parent_id
                new_parent_name = user_verdict.get("new_parent_name") or new_parent_name
                canonical_name = user_verdict["canonical_name"]
                reason = user_verdict["reason"]

            if action == "SYNONYM" and matched_id:
                param = self.session.get(Parameter, int(matched_id))
                c_name = param.canonical_name if param else (canonical_name or raw_key)
                logger.info(f"[АРБИТРАЖ] Признан СИНОНИМОМ ID {matched_id} ('{c_name}'). Обоснование: {reason}")
                return MatchDecision(
                    action="SYNONYM",
                    parameter_id=int(matched_id),
                    canonical_name=c_name,
                    similarity=candidates[0].similarity if candidates else 0.8,
                    reason=f"Арбитраж: {reason}",
                    candidates=candidates,
                )
            elif action == "NEW_PARENT_AND_CHILD":
                parent_name = new_parent_name or "Общий симптом"
                c_name = canonical_name or self._formulate_canonical(raw_key, raw_value, context)[0]
                logger.info(
                    f"[АРБИТРАЖ] Решение: Создать НОВЫЙ родительский класс '{parent_name}' "
                    f"и подкласс '{c_name}'. Причина: {reason}"
                )
                return MatchDecision(
                    action="NEW_PARENT_AND_CHILD",
                    parameter_id=None,
                    parent_id=None,
                    new_parent_name=parent_name,
                    canonical_name=c_name,
                    similarity=candidates[0].similarity if candidates else 0.8,
                    reason=f"Арбитраж (NEW_PARENT_AND_CHILD): {reason}",
                    candidates=candidates,
                )
            elif action == "CHILD_OF":
                target_p_id = int(parent_id) if parent_id else (int(matched_id) if matched_id else (candidates[0].parameter_id if candidates else None))
                parent_param = self.session.get(Parameter, target_p_id) if target_p_id else None
                p_name = parent_param.canonical_name if parent_param else "общий параметр"
                c_name = canonical_name or self._formulate_canonical(raw_key, raw_value, context)[0]
                logger.info(
                    f"[АРБИТРАЖ] Признан ПОДКЛАССОМ (CHILD_OF) для ID {target_p_id} ('{p_name}'). "
                    f"Название подкласса: '{c_name}'. Обоснование: {reason}"
                )
                return MatchDecision(
                    action="CHILD_OF",
                    parameter_id=None,
                    parent_id=target_p_id,
                    canonical_name=c_name,
                    similarity=candidates[0].similarity if candidates else 0.8,
                    reason=f"Арбитраж (CHILD_OF): {reason}",
                    candidates=candidates,
                )
            elif action == "SKIP":
                logger.info(f"[АРБИТРАЖ] Параметр '{raw_key}' пропущен экспертом.")
                return MatchDecision(
                    action="NEW",
                    parameter_id=None,
                    canonical_name=raw_key,
                    similarity=0.0,
                    reason="Пропущено экспертом в модальном окне",
                    candidates=candidates,
                )
            else:
                c_name = canonical_name or self._formulate_canonical(raw_key, raw_value, context)[0]
                logger.info(f"[АРБИТРАЖ] Признан НОВЫМ параметром ('{c_name}'). Обоснование: {reason}")
                return MatchDecision(
                    action="NEW",
                    parameter_id=None,
                    canonical_name=c_name,
                    similarity=candidates[0].similarity if candidates else 0.0,
                    reason=f"Арбитраж (NEW): {reason}",
                    candidates=candidates,
                )

        except Exception as e:
            logger.error(f"Ошибка LLM-арбитража: {e}. Применяем клинический fallback.")
            if candidates and candidates[0].similarity >= 0.78:
                return MatchDecision(
                    action="SYNONYM",
                    parameter_id=candidates[0].parameter_id,
                    canonical_name=candidates[0].canonical_name,
                    similarity=candidates[0].similarity,
                    reason=f"Клинический fallback (сходство {candidates[0].similarity:.4f} с '{candidates[0].matched_text}')",
                    candidates=candidates,
                )
            else:
                c_name, _, _ = self._formulate_canonical(raw_key, raw_value, context)
                return MatchDecision(
                    action="NEW",
                    parameter_id=None,
                    canonical_name=c_name,
                    similarity=candidates[0].similarity if candidates else 0.0,
                    reason="Клинический fallback: признан новым параметром",
                    candidates=candidates,
                )

    def match_parameter(self, raw_key: str, raw_value: str, context: str) -> MatchDecision:
        """Главный метод сопоставления сырого параметра с базой онтологии."""
        clean_key = raw_key.strip()
        candidates = self.find_top_candidates(clean_key, top_k=3)

        # 1. Если база онтологии пуста
        if not candidates:
            canonical, _, _ = self._formulate_canonical(clean_key, raw_value, context)
            logger.info(f"Онтология пуста. Параметр '{clean_key}' зафиксирован как канонический: '{canonical}'")
            return MatchDecision(
                action="NEW",
                parameter_id=None,
                canonical_name=canonical,
                similarity=0.0,
                reason="Первичная фиксация параметра (онтология пуста)",
                candidates=[],
            )

        best_match = candidates[0]
        similarity = best_match.similarity

        # 2. Высокая уверенность (similarity >= 0.90)
        if similarity >= settings.SIMILARITY_THRESHOLD_HIGH:
            logger.info(
                f"[СХОДСТВО ВЫСОКОЕ] '{clean_key}' -> ID {best_match.parameter_id} ('{best_match.canonical_name}') "
                f"через {best_match.match_source} '{best_match.matched_text}' (сходство {similarity:.4f} >= {settings.SIMILARITY_THRESHOLD_HIGH})"
            )
            return MatchDecision(
                action="SYNONYM",
                parameter_id=best_match.parameter_id,
                canonical_name=best_match.canonical_name,
                similarity=similarity,
                reason=f"Высокое косинусное сходство ({similarity:.4f}) с '{best_match.matched_text}'",
                candidates=candidates,
            )

        # 3. Серая зона (0.70 <= similarity < 0.90)
        if similarity >= settings.SIMILARITY_THRESHOLD_LOW:
            logger.info(
                f"[СЕРАЯ ЗОНА] '{clean_key}' имеет сходство {similarity:.4f} с '{best_match.matched_text}'. "
                f"Запуск LLM-арбитра..."
            )
            return self._arbitrate_with_llm(clean_key, raw_value, context, candidates)

        # 4. Низкое сходство (similarity < 0.70)
        logger.info(
            f"[НОВОЕ] '{clean_key}' имеет низкое сходство ({similarity:.4f} < {settings.SIMILARITY_THRESHOLD_LOW}) "
            f"с ближайшим '{best_match.matched_text}'. Создание нового канонического параметра..."
        )
        canonical, _, _ = self._formulate_canonical(clean_key, raw_value, context)
        return MatchDecision(
            action="NEW",
            parameter_id=None,
            canonical_name=canonical,
            similarity=similarity,
            reason=f"Низкое сходство ({similarity:.4f} < {settings.SIMILARITY_THRESHOLD_LOW}) с '{best_match.matched_text}'",
            candidates=candidates,
        )
