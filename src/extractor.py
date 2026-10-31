"""Модуль извлечения клинических параметров из медицинских анкет с помощью LLM (Qwen)."""

import json
from typing import Any, Dict, List, Optional
import requests
from pydantic import BaseModel, Field
from src.config import logger, settings


class ExtractedEntity(BaseModel):
    """Модель извлеченного клинического параметра."""

    raw_key: str = Field(..., description="Исходная формулировка параметра или симптома из текста")
    raw_value: str = Field(..., description="Значение параметра или характер проявления")
    context: str = Field(default="", description="Контекстное предложение или фрагмент из текста")
    category: Optional[str] = Field(default=None, description="Клиническая категория (например, Гемодинамика, Жалобы)")
    unit: Optional[str] = Field(default=None, description="Единица измерения (например, мм рт. ст., уд/мин)")
    confidence: float = Field(default=1.0, description="Степень уверенности извлечения")


class EntityExtractor:
    """Сервис извлечения клинических параметров из неструктурированного текста."""

    SYSTEM_PROMPT = (
        "Ты — медицинский клинический эксперт. Твоя задача — извлечь из текста медицинской анкеты "
        "все клинические показатели, симптомы, жалобы и витальные функции (АД, ЧСС/пульс, ЧДД, температура, SpO2).\n"
        "Правила:\n"
        "1. Для каждого показателя укажи: 'raw_key' (название), 'raw_value' (значение), 'context' (предложение), "
        "'category' (категория), 'unit' (единица измерения или null).\n"
        "2. Учитывай российские медицинские термины и сокращения (АД, ЧСС, пульс, ЧДД, сатурация, цефалгия и т.д.).\n"
        "3. Верни результат СТРОГО в виде валидного JSON без markdown и пояснений:\n"
        "{\n"
        '  "parameters": [\n'
        '    {"raw_key": "АД", "raw_value": "120/80 мм рт. ст.", "context": "...", "category": "Гемодинамика", "unit": "мм рт. ст."}\n'
        "  ]\n"
        "}"
    )

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        logger.info(f"Инициализация EntityExtractor (провайдер: {self.provider}, модель: {self.model})")

    def _call_ollama(self, prompt: str) -> str:
        """Прямой вызов Ollama API с многопоточностью и гарантией JSON формата."""
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "system": self.SYSTEM_PROMPT,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_thread": 10,
                "num_predict": 1024,
            },
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        raw_text = response.json().get("response", "{}")

        # Защита от незавершенного JSON (если строка была оборвана по лимиту)
        raw_text_clean = raw_text.strip()
        if not raw_text_clean.endswith("}"):
            # Пытаемся корректно закрыть незавершенный массив параметров
            if '"parameters":' in raw_text_clean and "]" not in raw_text_clean.split('"parameters":')[-1]:
                raw_text_clean += "]}"
            elif not raw_text_clean.endswith("}"):
                raw_text_clean += "}"
        return raw_text_clean


    def _call_litellm_or_openai(self, prompt: str) -> str:
        """Вызов через LiteLLM или OpenAI API."""
        import litellm

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
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

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

    def extract_from_text(self, text: str, patient_id: str = "неизвестно") -> List[ExtractedEntity]:
        """Извлечение клинических сущностей из произвольного русскоязычного текста анкеты."""
        if not text or not text.strip():
            logger.warning(f"Получен пустой текст анкеты для пациента {patient_id}.")
            return []

        logger.info(f"Извлечение параметров из текста анкеты (пациент {patient_id}, длина: {len(text)} симв.)...")
        prompt = f"Текст медицинской анкеты пациента:\n\"\"\"\n{text.strip()}\n\"\"\""

        try:
            if self.provider == "ollama":
                raw_response = self._call_ollama(prompt)
            else:
                raw_response = self._call_litellm_or_openai(prompt)

            items = []
            try:
                data = json.loads(raw_response)
                items = data.get("parameters", [])
            except json.JSONDecodeError:
                # Устойчивый fallback через регулярные выражения для извлечения валидных блоков
                import re

                pattern = re.compile(
                    r'\{[^{}]*"raw_key"\s*:\s*"([^"]+)"[^{}]*"raw_value"\s*:\s*"([^"]+)"[^{}]*\}',
                    re.DOTALL,
                )
                for m in pattern.finditer(raw_response):
                    items.append({"raw_key": m.group(1), "raw_value": m.group(2)})

            entities = []
            for item in items:
                raw_key = item.get("raw_key", "").strip()
                raw_value = str(item.get("raw_value", "")).strip()
                if not raw_key or not raw_value:
                    continue

                entity = ExtractedEntity(
                    raw_key=raw_key,
                    raw_value=raw_value,
                    context=item.get("context", "") or text[:120],
                    category=item.get("category"),
                    unit=item.get("unit"),
                    confidence=float(item.get("confidence", 1.0)),
                )
                entities.append(entity)

            logger.info(f"Извлечено клинических параметров: {len(entities)} (пациент {patient_id}).")
            return entities

        except Exception as e:
            logger.error(f"Ошибка при извлечении параметров через LLM: {e}")
            return []

