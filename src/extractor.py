"""Модуль извлечения структурированных клинических данных и метаданных из медицинских протоколов с помощью LLM (Qwen).

Преобразует неструктурированный текст протокола (лист консультативного осмотра, протокол УЗИ, ЭКГ, выписка)
в иерархический ExtractedDocument на основе общеклинической методологии извлечения.
Извлечение всех сущностей и паспортной части выполняется строго через языковую модель.
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
    age: Optional[int] = Field(default=None, description="Возраст пациента (целое число)")
    date_of_birth: Optional[str] = Field(default=None, description="Дата рождения в формате ISO (YYYY-MM-DD)")
    address: str = Field(default="---", description="Адрес пациента")
    consultation_date: Optional[str] = Field(default=None, description="Дата и время осмотра в формате ISO (YYYY-MM-DDTHH:MM)")
    examination: str = Field(
        default="лист_консультативного_осмотра",
        description="Канонический тип исследования/осмотра в snake_case",
    )
    doctor: str = Field(default="---", description="ФИО врача")
    specialization: str = Field(
        default="терапия",
        description="Специальность врача в snake_case",
    )


class ExtractedEntity(BaseModel):
    """Модель атомарного извлеченного клинического признака/измерения."""

    raw_key: str = Field(..., description="Исходное название признака или узла (например, right_lobe, wall, heart_rate)")
    raw_value: Any = Field(..., description="Значение параметра (число, строка, массив)")
    context: str = Field(default="", description="Контекстное предложение или цитата из текста")
    path_hint: Optional[str] = Field(
        default=None,
        description="Иерархический путь в целевом JSON (например, liver.right_lobe, liver.gallbladder.wall, vitals.heart_rate)",
    )
    category: Optional[str] = Field(default=None, description="Анатомическая категория или орган (например, liver, pancreas, vitals)")
    unit: Optional[str] = Field(default=None, description="Единица измерения (например, мм, см, уд/мин, мм рт. ст.)")
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
    """Сервис структурирования произвольных медицинских текстов на базе методологического LLM-извлечения."""

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
        "Ты — ведущий медицинский эксперт-онтолог и специалист по извлечению данных из клинических документов "
        "(листов консультативного осмотра, протоколов инструментальных исследований, выписок, карт стационарного больного).\n\n"
        "Твоя задача — проанализировать неструктурированный медицинский текст на русском языке и преобразовать его "
        "в строго структурированный валидный JSON-документ единого формата.\n\n"
        "### МЕТОДОЛОГИЯ И ПРИНЦИПЫ ИЗВЛЕЧЕНИЯ:\n\n"
        "1. АДМИНИСТРАТИВНЫЙ БЛОК (Паспортные данные и метаданные осмотра):\n"
        "Извлеки из текста метаданные (если поле отсутствует или замаскировано '---', сохрани '---' для строк или null для чисел):\n"
        "- \"institution\": наименование медицинской организации (строка или '---')\n"
        "- \"patient\": ФИО пациента (строка или '---')\n"
        "- \"age\": возраст пациента строго целым числом (например: 94 или null)\n"
        "- \"date_of_birth\": дата рождения строго в формате ISO 'YYYY-MM-DD' (например: '1927-10-15' или null)\n"
        "- \"address\": адрес пациента (строка или '---')\n"
        "- \"consultation_date\": дата и время осмотра/исследования в формате ISO 'YYYY-MM-DDTHH:MM' (например: '2022-04-22T15:11' или null)\n"
        "- \"examination\": тип исследования или осмотра на латинице в snake_case (например: 'лист_консультативного_осмотра', 'ультразвуковое_исследование_органов_брюшной_полости', 'электрокардиографическое_исследование', 'первичный_осмотр_хирурга')\n"
        "- \"doctor\": ФИО врача (строка или '---')\n"
        "- \"specialization\": специальность врача на латинице в snake_case (например: 'ультразвуковая_диагностика', 'хирургия', 'функциональная_диагностика', 'терапия')\n\n"
        "2. ИЕРАРХИЧЕСКОЕ ДЕРЕВО КЛИНИЧЕСКИХ ПРИЗНАКОВ (Анатомические системы, органы и измерения):\n"
        "Все описанные в тексте показатели, симптомы и инструментальные находки разложи по иерархическим блокам:\n"
        "- Ключи верхнего уровня: названия анатомических систем, органов или функциональных блоков строго на латинице в snake_case "
        "(например: liver, gallbladder, pancreas, spleen, kidneys, right_kidney, left_kidney, cardiovascular_system, respiratory_system, digestive_system, vitals, ecg, endoscopy и т.д.).\n"
        "- Анатомическая вложенность: если структура является анатомической частью органа "
        "(например: желчный пузырь внутри печени/билиарной системы, чашечки/лоханка внутри почки, доли внутри печени/легких, отделы внутри сердца), "
        "структурируй их как вложенные словари: 'орган' -> 'подструктура' -> 'параметр'.\n"
        "- Имена параметров (листьев дерева): лаконичные клинические термины в snake_case "
        "(например: right_lobe, wall, size, heart_rate, blood_pressure, parenchyma, structure, vessels, calyx, pelvis).\n\n"
        "3. ПРАВИЛА ТИПИЗАЦИИ ЗНАЧЕНИЙ (Критически важно для формирования признакового датасета):\n"
        "- Числовые измерения (размеры, толщина, пульс, ЧДД, фракция выброса, лабораторные параметры, температура): "
        "извлекай СТРОГО ЧИСЛАМИ (int или float) БЕЗ указания единиц измерения (например: 152, 67, 3, 100, 36.6, 0.06).\n"
        "- Многомерные габариты (двухмерные или трехмерные размеры органов): "
        "сохраняй как каноническую строку (например: '65 х 28 мм', '9 х 8 х 7 см').\n"
        "- Качественные характеристики (эхогенность, структура, форма, характер дыхания, ритмичность): "
        "нормализуй в нижний регистр на русском языке через запятую или snake_case "
        "(например: 'диффузно_неоднородная, повышенной_эхогенности', 'приглушенные, ритмичные', 'везикулярное, хрипов_нет').\n"
        "- Отрицательные находки и отсутствие патологии: фиксируй лаконичной формулировкой (например: 'не_расширены', 'не_пальпируется', 'без_онкопатологии', 'хрипов_нет').\n\n"
        "4. КЛИНИЧЕСКОЕ ЗАКЛЮЧЕНИЕ И ДИАГНОЗЫ ('conclusion'):\n"
        "Массив строк со всеми установленными основными диагнозами, синдромами, осложнениями и сопутствующими патологиями в нижнем регистре на русском языке:\n"
        "[\"множественные_язвы_дпк\", \"состоявшееся_желудочное_кровотечение\", \"ибс\", \"желчекаменная_болезнь\", \"диффузные_изменения_в_печени\"]\n\n"
        "5. ФОРМАТ ВЫВОДА:\n"
        "Верни СТРОГО валидный JSON-объект без пояснительного текста, преамбул и markdown-разметки."
    )

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        logger.info(f"Инициализация EntityExtractor (провайдер: {self.provider}, модель: {self.model})")

    def _clean_json_response(self, raw_text: str) -> Dict[str, Any]:
        """Очистка от markdown-тегов и устойчивый парсинг JSON."""
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        if not cleaned.endswith("}"):
            open_braces = cleaned.count("{") - cleaned.count("}")
            if open_braces > 0:
                cleaned += "}" * open_braces

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                for wrap_key in ("data", "result", "protocol", "response", "document"):
                    if wrap_key in data and isinstance(data[wrap_key], dict) and len(data) == 1:
                        data = data[wrap_key]
                return data
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Ошибка парсинга JSON: {e}. Попытка извлечения среза скобок.")
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
                "num_predict": 2500,
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
        """Поиск контекстного предложения в исходном тексте для трассировки (lineage)."""
        sentences = [s.strip() for s in text.replace("\r\n", "\n").split("\n") if s.strip()]
        for term in search_terms:
            if not term or len(str(term)) < 2:
                continue
            str_term = str(term).lower()
            for s in sentences:
                if str_term in s.lower():
                    return s[:200]
        return text[:150].strip()

    def _infer_type_and_unit(self, key_name: str, value: Any) -> Tuple[str, Optional[str]]:
        """Определение типа данных и единицы измерения по структуре значения."""
        if isinstance(value, bool):
            return "boolean", None
        if isinstance(value, int):
            unit = "мм" if any(k in key_name for k in ("lobe", "wall", "head", "body", "tail", "size", "calyx", "pelvis", "parenchyma")) else None
            if "rate" in key_name or "pulse" in key_name or "chss" in key_name:
                unit = "уд/мин"
            return "integer", unit
        if isinstance(value, float):
            return "float", "мм" if "temp" not in key_name else "°C"
        if isinstance(value, list):
            return "array", None
        if isinstance(value, str):
            val_str = value.strip()
            # Проверка двухмерного/трехмерного габарита ("65 х 28 мм", "9*8*7 см")
            if re.search(r"\d+\s*[*xхXХ×]\s*\d+", val_str):
                return "dimension_2d", "мм"
            # Если число пришло в виде строки
            if val_str.isdigit():
                return "integer", None
            return "string", None
        return "object", None

    def _flatten_clinical_tree(
        self,
        tree: Dict[str, Any],
        raw_text: str,
        prefix: str = "",
    ) -> List[ExtractedEntity]:
        """Рекурсивное разворачивание извлеченного дерева органов и измерений в список атомарных признаков."""
        items: List[ExtractedEntity] = []

        for key, value in tree.items():
            if key in self.ADMIN_KEYS or key == "conclusion":
                continue

            full_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                items.extend(self._flatten_clinical_tree(value, raw_text, prefix=full_path))
            else:
                data_type, unit = self._infer_type_and_unit(key, value)
                context = self._find_context_snippet(raw_text, [str(value), key.replace("_", " ")])

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
        """Валидация извлеченной моделью административной шапки."""
        raw_age = data.get("age")
        norm_age: Optional[int] = None
        if raw_age is not None:
            try:
                norm_age = int(raw_age)
            except (ValueError, TypeError):
                pass

        raw_patient = data.get("patient")
        patient_name = (
            raw_patient
            if raw_patient and raw_patient != "---"
            else (patient_id if patient_id != "неизвестно" else "---")
        )

        return AdministrativeInfo(
            institution=str(data.get("institution") or "---"),
            patient=str(patient_name),
            age=norm_age,
            date_of_birth=data.get("date_of_birth"),
            address=str(data.get("address") or "---"),
            consultation_date=data.get("consultation_date"),
            examination=str(data.get("examination") or "лист_консультативного_осмотра"),
            doctor=str(data.get("doctor") or "---"),
            specialization=str(data.get("specialization") or "терапия"),
        )

    def extract_document(self, text: str, patient_id: str = "неизвестно") -> ExtractedDocument:
        """Главный метод извлечения: передает протокол в LLM и парсит возвращенную структуру."""
        if not text or not text.strip():
            logger.warning(f"Получен пустой текст протокола для пациента {patient_id}.")
            return ExtractedDocument()

        logger.info(f"Извлечение признаков через LLM (пациент {patient_id}, длина: {len(text)} симв.)...")
        prompt = f"Текст медицинского протокола осмотра / исследования:\n\"\"\"\n{text.strip()}\n\"\"\""

        try:
            if self.provider == "ollama":
                raw_response = self._call_ollama(prompt)
            else:
                raw_response = self._call_litellm_or_openai(prompt)

            data = self._clean_json_response(raw_response)

            # Валидация административной шапки из ответа модели
            admin_info = self._parse_administrative(data, patient_id=patient_id)

            # Извлечение заключения из ответа модели
            raw_conclusion = data.get("conclusion", [])
            conclusion_list: List[str] = []
            if isinstance(raw_conclusion, list):
                conclusion_list = [str(c).strip() for c in raw_conclusion if str(c).strip()]
            elif isinstance(raw_conclusion, str) and raw_conclusion.strip():
                conclusion_list = [raw_conclusion.strip()]

            # Разворачивание клинического дерева органов и систем
            findings = self._flatten_clinical_tree(data, raw_text=text)

            logger.info(
                f"Успешно извлечено через LLM: {len(findings)} клинических признаков, "
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
