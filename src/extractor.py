"""Модуль извлечения структурированных клинических данных и административной информации из медицинских протоколов с помощью LLM (Qwen).

Преобразует неструктурированный текст протокола (например, УЗИ ОБП)
в иерархический ExtractedDocument (шапка + дерево признаков + заключение).
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
import requests
from pydantic import BaseModel, Field
from src.config import logger, settings


class AdministrativeInfo(BaseModel):
    """Административная информация и метаданные протокола исследования."""

    institution: str = Field(default="---", description="Название медицинского учреждения")
    patient: str = Field(default="---", description="ФИО пациента")
    age: Optional[int] = Field(default=None, description="Возраст пациента (число)")
    date_of_birth: Optional[str] = Field(default=None, description="Дата рождения (YYYY-MM-DD)")
    address: str = Field(default="---", description="Адрес пациента")
    consultation_date: Optional[str] = Field(default=None, description="Дата и время осмотра (ISO 8601)")
    examination: str = Field(
        default="ультразвуковое_исследование_органов_брюшной_полости",
        description="Название исследования (snake_case)",
    )
    doctor: str = Field(default="---", description="ФИО врача")
    specialization: str = Field(
        default="ультразвуковая_диагностика",
        description="Специальность врача",
    )


class ExtractedEntity(BaseModel):
    """Модель атомарного извлеченного клинического признака/измерения."""

    raw_key: str = Field(..., description="Исходное название признака или узла (например, right_lobe, wall)")
    raw_value: Any = Field(..., description="Значение параметра (число, строка, массив)")
    context: str = Field(default="", description="Контекстное предложение или цитата из текста")
    path_hint: Optional[str] = Field(
        default=None,
        description="Иерархический путь в целевом JSON (например, liver.right_lobe, liver.gallbladder.wall)",
    )
    category: Optional[str] = Field(default=None, description="Анатомическая категория или орган (например, liver, pancreas)")
    unit: Optional[str] = Field(default=None, description="Единица измерения (например, мм, см, лет)")
    data_type: Optional[str] = Field(
        default="string",
        description="Тип данных (integer, float, string, dimension_2d, array)",
    )
    confidence: float = Field(default=1.0, description="Степень уверенности извлечения")


class ExtractedDocument(BaseModel):
    """Полный результат структурирования медицинского документа."""

    administrative: AdministrativeInfo = Field(default_factory=AdministrativeInfo)
    findings: List[ExtractedEntity] = Field(default_factory=list)
    conclusion: List[str] = Field(default_factory=list)
    raw_json_draft: Dict[str, Any] = Field(default_factory=dict)


class EntityExtractor:
    """Сервис извлечения иерархических клинических параметров и метаданных из текста протокола."""

    ADMIN_KEYS = {
        "institution",
        "patient",
        "age",
        "date_of_birth",
        "address",
        "consultation_date",
        "examination",
        "doctor",
        "specialization",
    }

    SYSTEM_PROMPT = (
        "Ты — высококвалифицированный медицинский эксперт и специалист по структурированию данных протоколов осмотра (УЗИ, КТ, выписки).\n"
        "Твоя задача — извлечь из текста протокола всю информацию и вернуть СТРОГО валидный JSON в следующем эталонном формате:\n\n"
        "{\n"
        '  "institution": "Название клиники или ---",\n'
        '  "patient": "ФИО пациента или ---",\n'
        '  "age": 94,\n'
        '  "date_of_birth": "1927-10-15",\n'
        '  "address": "---",\n'
        '  "consultation_date": "2022-04-22T16:24",\n'
        '  "examination": "ультразвуковое_исследование_органов_брюшной_полости",\n'
        '  "liver": {\n'
        '    "right_lobe": 152,\n'
        '    "left_lobe": 67,\n'
        '    "parenchyma": "диффузно_неоднородная, повышенной_эхогенности",\n'
        '    "vascular_tissue": "обеднена",\n'
        '    "bile_ducts": "не_расширены",\n'
        '    "gallbladder": {\n'
        '      "size": "65  х  28 мм",\n'
        '      "wall": 3,\n'
        '      "form": "правильная",\n'
        '      "intrahepatic_choledochal_system": "гиперэхогенные_образования_до_14 мм"\n'
        '    },\n'
        '    "pancreas": {\n'
        '      "head": 25,\n'
        '      "body": 14,\n'
        '      "tail": 24,\n'
        '      "structure": "однородная, эхогенность_повышена, контуры_нечеткие"\n'
        '    },\n'
        '    "spleen": {\n'
        '      "size": "94  х  34 мм",\n'
        '      "structure": "однородная",\n'
        '      "vein": "не_расширена"\n'
        '    },\n'
        '    "free_liquid_in_abdomen": "не_визуализируется",\n'
        '    "paraaortic_lymph_nodes": "не_изменены"\n'
        '  },\n'
        '  "right_kidney": {\n'
        '    "calyx": 20,\n'
        '    "pelvis": 42,\n'
        '    "parenchyma": 8\n'
        '  },\n'
        '  "conclusion": [\n'
        '    "жкб",\n'
        '    "диффузные_изменения_в_печени",\n'
        '    "гидронефроз_справа"\n'
        '  ],\n'
        '  "doctor": "---",\n'
        '  "specialization": "ультразвуковая_диагностика"\n'
        "}\n\n"
        "Правила:\n"
        "1. Ключи органов и структур пиши строго в snake_case латиницей (liver, gallbladder, pancreas, spleen, right_kidney, left_kidney и т.д.).\n"
        "2. Числовые измерения (толщина, доли, диаметры) записывай числами БЕЗ единиц (например: right_lobe: 152, wall: 3).\n"
        "3. Габаритные размеры двухмерные пиши строкой: \"65  х  28 мм\".\n"
        "4. Качественные характеристики пиши в нижнем регистре через snake_case или запятую (\"диффузно_неоднородная, повышенной_эхогенности\").\n"
        "5. Отсутствие признака/норму отражай лаконично (\"не_расширены\", \"не_визуализируется\").\n"
        "6. В \"conclusion\" включи массив всех диагнозов и синдромов в нижнем регистре.\n"
        "7. Верни СТРОГО JSON без markdown-разметки и пояснений."
    )

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        logger.info(f"Инициализация EntityExtractor (провайдер: {self.provider}, модель: {self.model})")

    def _clean_json_response(self, raw_text: str) -> Dict[str, Any]:
        """Очистка от markdown-блоков, обрезки и парсинг JSON."""
        cleaned = raw_text.strip()
        # Удаление markdown блоков ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        # Восстановление незавершенных скобок
        if not cleaned.endswith("}"):
            open_braces = cleaned.count("{") - cleaned.count("}")
            if open_braces > 0:
                cleaned += "}" * open_braces

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # Если LLM обернула ответ в {"data": ...} или {"result": ...}
                for wrap_key in ("data", "result", "protocol", "response"):
                    if wrap_key in data and isinstance(data[wrap_key], dict) and len(data) == 1:
                        data = data[wrap_key]
                return data
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Ошибка прямого JSONDecodeError: {e}. Применяем эвристический парсинг.")
            # Попытка найти JSON блок между первой { и последней }
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except Exception:
                    pass
            return {}

    def _call_ollama(self, prompt: str) -> str:
        """Прямой вызов Ollama API."""
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
                "num_predict": 2048,
            },
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "{}")

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

    def _find_context_snippet(self, text: str, search_terms: List[str]) -> str:
        """Поиск контекстного предложения в исходном тексте по совпадению терминов."""
        sentences = re.split(r"[.\n;]+", text)
        for term in search_terms:
            if not term or len(str(term)) < 2:
                continue
            str_term = str(term).lower()
            for s in sentences:
                if str_term in s.lower():
                    snippet = s.strip()
                    if snippet:
                        return snippet
        return text[:120].strip()

    def _infer_type_and_unit(self, key_name: str, value: Any) -> Tuple[str, Optional[str]]:
        """Определение типа данных и единицы измерения по значению и имени ключа."""
        if isinstance(value, bool):
            return "boolean", None
        if isinstance(value, int):
            unit = "мм" if any(k in key_name for k in ("lobe", "wall", "head", "body", "tail", "size", "calyx", "pelvis", "parenchyma")) else None
            return "integer", unit
        if isinstance(value, float):
            return "float", "мм"
        if isinstance(value, list):
            return "array", None
        if isinstance(value, str):
            val_str = value.strip()
            # Проверка двухмерного габарита вида "65 х 28 мм"
            if re.search(r"\d+\s*[xхXХ×]\s*\d+", val_str):
                return "dimension_2d", "мм"
            # Проверка чистого числа в строке
            if re.match(r"^-?\d+$", val_str):
                return "integer", "мм"
            if re.match(r"^-?\d+\.\d+$", val_str):
                return "float", "мм"
            return "string", None
        return "object", None

    def _flatten_clinical_tree(
        self,
        tree: Dict[str, Any],
        raw_text: str,
        prefix: str = "",
    ) -> List[ExtractedEntity]:
        """Рекурсивное разворачивание дерева органов и измерений в список атомарных признаков."""
        items: List[ExtractedEntity] = []

        for key, value in tree.items():
            if key in self.ADMIN_KEYS or key == "conclusion":
                continue

            full_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Вложенный объект/структура (например, gallbladder внутри liver)
                items.extend(self._flatten_clinical_tree(value, raw_text, prefix=full_path))
            else:
                data_type, unit = self._infer_type_and_unit(key, value)
                context = self._find_context_snippet(raw_text, [str(value), key.replace("_", " ")])

                # Нормализация числовых значений
                norm_value = value
                if data_type == "integer":
                    try:
                        norm_value = int(value)
                    except (ValueError, TypeError):
                        pass
                elif data_type == "float":
                    try:
                        norm_value = float(value)
                    except (ValueError, TypeError):
                        pass

                entity = ExtractedEntity(
                    raw_key=key,
                    raw_value=norm_value,
                    context=context,
                    path_hint=full_path,
                    category=prefix.split(".")[0] if prefix else key,
                    unit=unit,
                    data_type=data_type,
                    confidence=1.0,
                )
                items.append(entity)

        return items

    def _parse_administrative(self, data: Dict[str, Any], patient_id: str) -> AdministrativeInfo:
        """Извлечение и валидация административной шапки."""
        raw_age = data.get("age")
        norm_age: Optional[int] = None
        if raw_age is not None:
            if isinstance(raw_age, int):
                norm_age = raw_age
            elif isinstance(raw_age, str):
                match = re.search(r"\d+", raw_age)
                if match:
                    norm_age = int(match.group(0))

        patient_name = data.get("patient") or patient_id or "---"

        return AdministrativeInfo(
            institution=str(data.get("institution") or "---"),
            patient=str(patient_name),
            age=norm_age,
            date_of_birth=data.get("date_of_birth"),
            address=str(data.get("address") or "---"),
            consultation_date=data.get("consultation_date"),
            examination=str(data.get("examination") or "ультразвуковое_исследование_органов_брюшной_полости"),
            doctor=str(data.get("doctor") or "---"),
            specialization=str(data.get("specialization") or "ультразвуковая_диагностика"),
        )

    def extract_document(self, text: str, patient_id: str = "неизвестно") -> ExtractedDocument:
        """Главный метод извлечения: возвращает полный иерархический ExtractedDocument."""
        if not text or not text.strip():
            logger.warning(f"Получен пустой текст анкеты для пациента {patient_id}.")
            return ExtractedDocument()

        logger.info(f"Извлечение иерархических признаков через LLM (пациент {patient_id}, длина: {len(text)} симв.)...")
        prompt = f"Текст медицинского протокола исследования:\n\"\"\"\n{text.strip()}\n\"\"\""

        try:
            if self.provider == "ollama":
                raw_response = self._call_ollama(prompt)
            else:
                raw_response = self._call_litellm_or_openai(prompt)

            data = self._clean_json_response(raw_response)

            # Если LLM вернула старый плоский формат {"parameters": [...]}
            if "parameters" in data and isinstance(data["parameters"], list) and len(data) == 1:
                entities = []
                for p in data["parameters"]:
                    entities.append(
                        ExtractedEntity(
                            raw_key=p.get("raw_key", ""),
                            raw_value=p.get("raw_value", ""),
                            context=p.get("context", ""),
                            category=p.get("category"),
                            unit=p.get("unit"),
                            confidence=float(p.get("confidence", 1.0)),
                        )
                    )
                return ExtractedDocument(
                    findings=entities,
                    raw_json_draft=data,
                )

            # Извлечение административной шапки
            admin_info = self._parse_administrative(data, patient_id=patient_id)

            # Извлечение заключения
            raw_conclusion = data.get("conclusion", [])
            conclusion_list: List[str] = []
            if isinstance(raw_conclusion, list):
                conclusion_list = [str(c).strip() for c in raw_conclusion if str(c).strip()]
            elif isinstance(raw_conclusion, str):
                conclusion_list = [c.strip() for c in re.split(r"[,;\n]+", raw_conclusion) if c.strip()]

            # Разворачивание клинического дерева
            findings = self._flatten_clinical_tree(data, raw_text=text)

            logger.info(
                f"Успешно извлечено: {len(findings)} клинических признаков, "
                f"{len(conclusion_list)} заключений (пациент {patient_id})."
            )

            return ExtractedDocument(
                administrative=admin_info,
                findings=findings,
                conclusion=conclusion_list,
                raw_json_draft=data,
            )

        except Exception as e:
            logger.error(f"Ошибка при извлечении параметров через LLM: {e}")
            return ExtractedDocument()

    def extract_from_text(self, text: str, patient_id: str = "неизвестно") -> List[ExtractedEntity]:
        """Обратная совместимость с кодом, ожидающим List[ExtractedEntity]."""
        doc = self.extract_document(text, patient_id=patient_id)
        return doc.findings
