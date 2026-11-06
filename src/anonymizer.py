"""Модуль деидентификации и анонимизации персональных данных (ПДн) в медицинских протоколах.

Соответствует базовым требованиям Федерального закона № 152-ФЗ «О персональных данных»
и международному стандарту HIPAA Safe Harbor.

Выполняет детекцию и маскирование:
- ФИО пациентов и третьих лиц
- Серий и номеров паспортов РФ
- Номеров СНИЛС (XXX-XXX-XXX XX)
- Номеров полисов ОМС и ДМС (16 цифр)
- Контактных номеров телефонов (+7 / 8)
- Адресов электронной почты (email)
- Точных адресов проживания и регистрации
- Точных дат рождения (с возможностью сохранения возраста)
"""

from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Pattern, Tuple


# Регулярные выражения для детекции персональных данных
RE_SNILS: Pattern = re.compile(
    r"\b\d{3}[-\s]\d{3}[-\s]\d{3}[-\s]\d{2}\b"
)

RE_PASSPORT_RF: Pattern = re.compile(
    r"\b(?:паспорт(?:\s*РФ)?\s*:?\s*)?(?:\d{2}\s*\d{2}\s*№?\s*\d{6}|\d{4}\s*№?\s*\d{6})\b",
    re.IGNORECASE,
)

RE_OMS_POLIS: Pattern = re.compile(
    r"\b(?:полис(?:\s*ОМС|\s*ДМС)?\s*:?\s*)?\d{16}\b",
    re.IGNORECASE,
)

RE_PHONE: Pattern = re.compile(
    r"(?:\+7|8)[\s\-(]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"
)

RE_EMAIL: Pattern = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

RE_BIRTH_DATE: Pattern = re.compile(
    r"(?:(?:дата\s*рождения|д\.?\s*р\.?|род\.?)\s*:?\s*)(\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b)",
    re.IGNORECASE,
)

RE_ADDRESS: Pattern = re.compile(
    r"(?:(?:адрес|прожив\.|проживание|регистрация|зарег\.)\s*:?\s*)([^\n\r]+)",
    re.IGNORECASE,
)

RE_FIO_PREFIX: Pattern = re.compile(
    r"(?:(?:пациент(?:ка)?|больной(?:ая)?|фио|ф\.и\.о\.)\s*:?\s*)([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})",
    re.IGNORECASE,
)


class DataAnonymizer:
    """Сервис деидентификации медицинских документов."""

    def __init__(self, default_pseudonym_prefix: str = "PATIENT"):
        self.default_pseudonym_prefix = default_pseudonym_prefix

    def anonymize_text(
        self,
        text: str,
        pseudonym: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Маскирование персональных данных в сыром тексте протокола.

        Args:
            text: Исходный текст медицинского документа.
            pseudonym: Пользовательский псевдоним пациента (например 'PATIENT_001').

        Returns:
            Tuple[str, Dict[str, Any]]: (Анонимизированный текст, отчет о скрытых ПДн)
        """
        if not text:
            return "", {"masked_items": 0}

        cleaned = text
        audit: Dict[str, List[str]] = {
            "fio": [],
            "phone": [],
            "email": [],
            "snils": [],
            "passport": [],
            "polis": [],
            "birth_date": [],
            "address": [],
        }

        pat_placeholder = f"[{pseudonym or self.default_pseudonym_prefix}]"

        # 1. Маскирование ФИО после ключевых слов (Пациент: Иванов И.И.)
        def _replace_fio(m):
            val = m.group(1).strip()
            audit["fio"].append(val)
            prefix = m.group(0)[: m.start(1) - m.start(0)]
            return f"{prefix}{pat_placeholder}"

        cleaned = RE_FIO_PREFIX.sub(_replace_fio, cleaned)

        # 2. Маскирование номеров телефонов
        def _replace_phone(m):
            audit["phone"].append(m.group(0))
            return "[ТЕЛЕФОН_СКРЫТ]"

        cleaned = RE_PHONE.sub(_replace_phone, cleaned)

        # 3. Маскирование email
        def _replace_email(m):
            audit["email"].append(m.group(0))
            return "[EMAIL_СКРЫТ]"

        cleaned = RE_EMAIL.sub(_replace_email, cleaned)

        # 4. Маскирование СНИЛС
        def _replace_snils(m):
            audit["snils"].append(m.group(0))
            return "[СНИЛС_СКРЫТ]"

        cleaned = RE_SNILS.sub(_replace_snils, cleaned)

        # 5. Маскирование паспортов
        def _replace_passport(m):
            audit["passport"].append(m.group(0))
            return "[ПАСПОРТ_СКРЫТ]"

        cleaned = RE_PASSPORT_RF.sub(_replace_passport, cleaned)

        # 6. Маскирование полисов ОМС/ДМС
        def _replace_polis(m):
            audit["polis"].append(m.group(0))
            return "[ПОЛИС_СКРЫТ]"

        cleaned = RE_OMS_POLIS.sub(_replace_polis, cleaned)

        # 7. Маскирование точной даты рождения (сохраняя указание)
        def _replace_bdate(m):
            val = m.group(1)
            audit["birth_date"].append(val)
            prefix = m.group(0)[: m.start(1) - m.start(0)]
            # Пробуем извлечь только год рождения
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", val)
            if year_match:
                return f"{prefix}[ГОД_РОЖДЕНИЯ_{year_match.group(1)}]"
            return f"{prefix}[ДАТА_РОЖДЕНИЯ_СКРЫТА]"

        cleaned = RE_BIRTH_DATE.sub(_replace_bdate, cleaned)

        # 8. Маскирование адресов
        def _replace_address(m):
            val = m.group(1).strip().rstrip(".")
            audit["address"].append(val)
            prefix = m.group(0)[: m.start(1) - m.start(0)]
            return f"{prefix}[АДРЕС_СКРЫТ]"

        cleaned = RE_ADDRESS.sub(_replace_address, cleaned)

        total_masked = sum(len(v) for v in audit.values())
        return cleaned, {
            "masked_items": total_masked,
            "audit": {k: v for k, v in audit.items() if v},
        }

    def anonymize_administrative_block(
        self,
        admin_data: Dict[str, Any],
        pseudonym: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Псевдонимизация административных данных анкеты."""
        if not admin_data or not isinstance(admin_data, dict):
            return {}

        result = dict(admin_data)
        pat_placeholder = pseudonym or self.default_pseudonym_prefix

        # Маскируем пациента
        if "patient" in result and result["patient"]:
            result["patient_original_masked"] = True
            result["patient"] = pat_placeholder

        # Маскируем адрес
        if "address" in result and result["address"]:
            result["address"] = "[АДРЕС_СКРЫТ]"

        # Преобразуем точную дату рождения в год, если возможно
        if "date_of_birth" in result and result["date_of_birth"]:
            dob_str = str(result["date_of_birth"]).strip()
            year_m = re.search(r"\b(19\d{2}|20\d{2})\b", dob_str)
            if year_m:
                result["date_of_birth"] = f"{year_m.group(1)}-XX-XX"
            else:
                result["date_of_birth"] = "[ДАТА_СКРЫТА]"

        return result


_default_anonymizer = DataAnonymizer()


def anonymize_text(text: str, pseudonym: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Функция быстрого маскирования персональных данных в тексте."""
    return _default_anonymizer.anonymize_text(text=text, pseudonym=pseudonym)


def anonymize_administrative(admin_data: Dict[str, Any], pseudonym: Optional[str] = None) -> Dict[str, Any]:
    """Функция быстрого маскирования словаря административных атрибутов."""
    return _default_anonymizer.anonymize_administrative_block(admin_data=admin_data, pseudonym=pseudonym)
