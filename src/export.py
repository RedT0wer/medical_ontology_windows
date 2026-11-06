"""Модуль экспорта датасета признаков медицинских анкет в CSV и Parquet.

Принимает на вход:
- config: путь к YAML/JSON файлу конфигурации (например, configs/features.yaml) со списком признаков
- output: путь к результирующему файлу (.csv или .parquet)
- format: формат выгрузки ('csv' или 'parquet')

Выгружает значения указанных признаков для всех анкет в базе данных.
Если у анкеты нет признака — ставит NaN.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml
from sqlalchemy.orm import Session
from src.db import get_db_session
from src.models import Document, DocumentValue


def load_feature_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Загрузка перечня признаков из конфигурационного YAML/JSON файла.

    Поддерживает:
    - словарь с ключом 'features' / 'columns'
    - список признаков верхнего уровня
    - словарь, ключи которого являются признаками
    - опциональные параметры 'output' и 'format'

    Returns:
        {
            "features": List[str],
            "output": Optional[str],
            "format": Optional[str],
        }
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    if content is None:
        raise ValueError(f"Файл конфигурации пуст: {path}")

    features: List[str] = []
    output_file: Optional[str] = None
    output_format: Optional[str] = None

    if isinstance(content, list):
        features = [str(item).strip() for item in content if item is not None and str(item).strip()]
    elif isinstance(content, dict):
        output_file = content.get("output")
        output_format = content.get("format")

        raw_features = None
        for key in ("features", "features_list", "target_features", "columns"):
            if key in content:
                raw_features = content[key]
                break

        if raw_features is not None:
            if isinstance(raw_features, list):
                features = [str(item).strip() for item in raw_features if item is not None and str(item).strip()]
            elif isinstance(raw_features, dict):
                features = [str(k).strip() for k in raw_features.keys() if str(k).strip()]
            elif isinstance(raw_features, str):
                features = [f.strip() for f in raw_features.split(",") if f.strip()]
        else:
            service_keys = {"output", "format"}
            features = [str(k).strip() for k in content.keys() if k not in service_keys and str(k).strip()]
    else:
        raise ValueError(f"Неподдерживаемая структура файла конфигурации в {path}: {type(content)}")

    if not features:
        raise ValueError(f"В конфигурационном файле {path} не найдено признаков для экспорта.")

    return {
        "features": features,
        "output": output_file,
        "format": output_format,
    }


def extract_document_values_map(doc: Document) -> Dict[str, Any]:
    """Формирует словарь значений признаков для конкретной анкеты."""
    val_map: Dict[str, Any] = {}

    # 1. Извлеченные значения из фактов DocumentValue
    for v in doc.values:
        if not v.feature:
            continue

        val: Any = None
        if v.value_numeric is not None:
            val = v.value_numeric
        elif v.value_text is not None:
            val = v.value_text
        elif v.value_json is not None:
            if isinstance(v.value_json, (int, float, bool, str)):
                val = v.value_json
            elif isinstance(v.value_json, (list, tuple)) and all(isinstance(x, str) for x in v.value_json):
                val = "; ".join(v.value_json)
            else:
                val = json.dumps(v.value_json, ensure_ascii=False)

        if val is not None:
            val_map[v.feature.full_path] = val
            if v.feature.key_name and v.feature.key_name not in val_map:
                val_map[v.feature.key_name] = val

    # 2. Административный блок
    if doc.administrative and isinstance(doc.administrative, dict):
        for k, v in doc.administrative.items():
            if v is not None and k not in val_map:
                val_map[k] = v
                val_map[f"administrative.{k}"] = v
                val_map[f"administrative_block.{k}"] = v

    # 3. Кэш скомпилированного дерева JSON
    if doc.json_cache and isinstance(doc.json_cache.compiled_json, dict):
        def _flatten_json(d: Dict[str, Any], prefix: str = "") -> None:
            for k, v in d.items():
                curr_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten_json(v, curr_key)
                elif v is not None:
                    if curr_key not in val_map:
                        val_map[curr_key] = v
                    if k not in val_map:
                        val_map[k] = v

        _flatten_json(doc.json_cache.compiled_json)

    # 4. Поля самой анкеты
    val_map["id"] = doc.id
    val_map["document_id"] = doc.id
    val_map["patient_id"] = doc.patient_id
    if doc.patient_name:
        val_map["patient_name"] = doc.patient_name
    if doc.consultation_date:
        val_map["consultation_date"] = doc.consultation_date.isoformat()

    return val_map


def is_null_value(val: Any) -> bool:
    """Проверка, является ли значение пустым или NaN."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "nan", "none", "null"):
        return True
    return False


def serialize_cell_value(val: Any) -> Any:
    """Приводит значение ячейки к базовому типу для безопасного сохранения в датасет."""
    if is_null_value(val):
        return np.nan
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    if isinstance(val, (list, tuple, set)):
        if len(val) == 0:
            return np.nan
        if all(isinstance(i, str) for i in val):
            return "; ".join(val)
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, dict):
        if len(val) == 0:
            return np.nan
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def clean_column_for_export(series: pd.Series, col_name: str = "") -> pd.Series:
    """Нормализует типы данных в колонке: числа -> float64/int64, текст -> str с сохранением NaN."""
    if col_name in ("patient_id", "patient"):
        return series.map(lambda x: str(x) if not is_null_value(x) else np.nan)

    if col_name in ("document_id", "id"):
        try:
            return series.astype("int64")
        except (ValueError, TypeError):
            pass

    serialized = [serialize_cell_value(v) for v in series]
    non_null = [v for v in serialized if not is_null_value(v)]

    if not non_null:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    try:
        float_vals = [float(v) for v in non_null]
        if len(non_null) == len(series) and all(v.is_integer() for v in float_vals):
            return pd.Series([int(v) for v in float_vals], index=series.index, dtype="int64")
        return pd.Series(
            [float(v) if not is_null_value(v) else np.nan for v in serialized],
            index=series.index,
            dtype="float64",
        )
    except (ValueError, TypeError):
        return pd.Series(
            [str(v) if not is_null_value(v) else np.nan for v in serialized],
            index=series.index,
            dtype="object",
        )


def parse_flexible_date(date_val: Union[str, datetime]) -> datetime:
    """Парсинг даты из строки или datetime с поддержкой различных форматов."""
    if isinstance(date_val, datetime):
        return date_val
    clean = str(date_val).strip()
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%d.%m.%Y",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(clean)
    except Exception:
        raise ValueError(f"Не удалось распознать формат даты: '{date_val}'")


def build_dataset_dataframe(
    session: Session,
    features: List[str],
    from_date: Optional[Union[str, datetime]] = None,
    to_date: Optional[Union[str, datetime]] = None,
    filter_diagnosis: Optional[str] = None,
    filter_doctor: Optional[str] = None,
    filter_institution: Optional[str] = None,
) -> pd.DataFrame:
    """Формирует DataFrame со значениями указанных признаков для анкет в базе данных,

    с опциональной фильтрацией по дате, диагнозу, врачу и медицинскому учреждению.
    Если у анкеты нет признака — ставит NaN.
    """
    documents = session.query(Document).order_by(Document.id).all()

    # Фильтрация по нижней границе даты консультации
    if from_date is not None:
        from_dt = parse_flexible_date(from_date)
        documents = [d for d in documents if d.consultation_date and d.consultation_date >= from_dt]

    # Фильтрация по верхней границе даты консультации
    if to_date is not None:
        to_dt = parse_flexible_date(to_date)
        if to_dt.hour == 0 and to_dt.minute == 0 and to_dt.second == 0:
            to_dt = to_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        documents = [d for d in documents if d.consultation_date and d.consultation_date <= to_dt]

    # Фильтрация по диагнозу / клиническому заключению
    if filter_diagnosis:
        diag_term = filter_diagnosis.strip().lower()
        matched_docs = []
        for d in documents:
            matched = False
            if d.json_cache and isinstance(d.json_cache.compiled_json, dict):
                conc = d.json_cache.compiled_json.get("conclusion")
                if conc and diag_term in str(conc).lower():
                    matched = True
            if not matched:
                for v in d.values:
                    if v.feature and v.feature.full_path == "conclusion":
                        if (v.value_text and diag_term in v.value_text.lower()) or (
                            v.value_json and diag_term in str(v.value_json).lower()
                        ):
                            matched = True
                            break
            if not matched and d.raw_text and diag_term in d.raw_text.lower():
                matched = True

            if matched:
                matched_docs.append(d)
        documents = matched_docs

    # Фильтрация по врачу
    if filter_doctor:
        doc_term = filter_doctor.strip().lower()
        matched_docs = []
        for d in documents:
            matched = False
            if d.administrative and isinstance(d.administrative, dict):
                doctor_val = str(d.administrative.get("doctor") or "").lower()
                if doc_term in doctor_val:
                    matched = True
            if not matched and d.json_cache and isinstance(d.json_cache.compiled_json, dict):
                doctor_val = str(d.json_cache.compiled_json.get("doctor") or "").lower()
                if doc_term in doctor_val:
                    matched = True
            if matched:
                matched_docs.append(d)
        documents = matched_docs

    # Фильтрация по медицинскому учреждению
    if filter_institution:
        inst_term = filter_institution.strip().lower()
        matched_docs = []
        for d in documents:
            matched = False
            if d.administrative and isinstance(d.administrative, dict):
                inst_val = str(d.administrative.get("institution") or "").lower()
                if inst_term in inst_val:
                    matched = True
            if not matched and d.json_cache and isinstance(d.json_cache.compiled_json, dict):
                inst_val = str(d.json_cache.compiled_json.get("institution") or "").lower()
                if inst_term in inst_val:
                    matched = True
            if matched:
                matched_docs.append(d)
        documents = matched_docs

    columns = list(features)

    if not documents:
        df = pd.DataFrame(columns=columns)
        for col in columns:
            df[col] = clean_column_for_export(df[col], col_name=col)
        return df

    rows: List[Dict[str, Any]] = []
    for doc in documents:
        row: Dict[str, Any] = {}
        val_map = extract_document_values_map(doc)

        for feat in features:
            val = val_map.get(feat, np.nan)
            if val is None or val == "" or (isinstance(val, str) and val.lower() == "nan"):
                val = np.nan
            row[feat] = val

        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    for col in df.columns:
        df[col] = clean_column_for_export(df[col], col_name=col)

    return df


def export_dataset(
    config: Optional[Union[str, Path]] = None,
    output: Optional[Union[str, Path]] = None,
    format: Optional[str] = None,
    from_date: Optional[Union[str, datetime]] = None,
    to_date: Optional[Union[str, datetime]] = None,
    filter_diagnosis: Optional[str] = None,
    filter_doctor: Optional[str] = None,
    filter_institution: Optional[str] = None,
) -> Dict[str, Any]:
    """Сборка и выгрузка датасета указанных признаков для всех анкет в базе данных

    с возможностью гибкой фильтрации.

    Args:
        config: Путь к YAML-файлу конфигурации признаков (по умолчанию configs/features.yaml).
        output: Путь к файлу датасета (.csv или .parquet).
        format: Формат выгрузки ('csv' или 'parquet').
        from_date: Нижняя граница даты консультации (включительно).
        to_date: Верхняя граница даты консультации (включительно).
        filter_diagnosis: Подстрока диагноза для фильтрации анкет.
        filter_doctor: Подстрока ФИО врача.
        filter_institution: Подстрока названия медучреждения.

    Returns:
        Словарь с отчетом об экспорте.
    """
    config_path = config or "configs/features.yaml"
    cfg = load_feature_config(config_path)
    target_features = cfg["features"]

    eff_output = str(output) if output else cfg.get("output")
    eff_format = (format or cfg.get("format") or "").lower().strip() or None

    if not eff_output:
        default_fmt = eff_format if eff_format in ("csv", "parquet") else "csv"
        eff_output = f"dataset.{default_fmt}"

    out_file = Path(eff_output)
    suffix = out_file.suffix.lower()

    if eff_format in ("csv", "parquet"):
        final_format = eff_format
    elif suffix == ".parquet":
        final_format = "parquet"
    else:
        final_format = "csv"

    filters_applied = {
        k: v for k, v in {
            "from_date": str(from_date) if from_date else None,
            "to_date": str(to_date) if to_date else None,
            "filter_diagnosis": filter_diagnosis,
            "filter_doctor": filter_doctor,
            "filter_institution": filter_institution,
        }.items() if v is not None
    }

    with get_db_session() as session:
        df = build_dataset_dataframe(
            session=session,
            features=target_features,
            from_date=from_date,
            to_date=to_date,
            filter_diagnosis=filter_diagnosis,
            filter_doctor=filter_doctor,
            filter_institution=filter_institution,
        )

        out_file.parent.mkdir(parents=True, exist_ok=True)

        if final_format == "parquet":
            df.to_parquet(out_file, index=False)
        else:
            df.to_csv(out_file, index=False, na_rep="NaN", encoding="utf-8-sig")

        file_size = out_file.stat().st_size

        return {
            "status": "ok",
            "file": str(out_file.resolve()),
            "format": final_format,
            "total_documents": len(df),
            "total_features": len(target_features),
            "columns": list(df.columns),
            "file_size_bytes": file_size,
            "filters": filters_applied,
        }


def main() -> None:
    """Точка входа при прямом запуске модуля export.py."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Экспорт признаков всех анкет из базы данных в CSV или Parquet по конфигурационному файлу."
    )
    parser.add_argument(
        "config_pos",
        nargs="?",
        default=None,
        metavar="CONFIG",
        help="Путь к YAML-файлу конфигурации признаков (например, configs/features.yaml)",
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        type=str,
        default=None,
        help="Путь к YAML-файлу конфигурации признаков",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Путь к результирующему файлу (.csv или .parquet)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["csv", "parquet"],
        default=None,
        help="Формат выгрузки (csv или parquet)",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Фильтрация: начальная дата осмотра (ГГГГ-ММ-ДД)",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="Фильтрация: конечная дата осмотра (ГГГГ-ММ-ДД)",
    )
    parser.add_argument(
        "--filter-diagnosis",
        type=str,
        default=None,
        help="Фильтрация: подстрока диагноза/заключения",
    )
    parser.add_argument(
        "--filter-doctor",
        type=str,
        default=None,
        help="Фильтрация: подстрока ФИО врача",
    )
    parser.add_argument(
        "--filter-institution",
        type=str,
        default=None,
        help="Фильтрация: подстрока медицинского учреждения",
    )

    args = parser.parse_args()
    config_file = args.config or args.config_pos

    try:
        result = export_dataset(
            config=config_file,
            output=args.output,
            format=args.format,
            from_date=args.from_date,
            to_date=args.to_date,
            filter_diagnosis=args.filter_diagnosis,
            filter_doctor=args.filter_doctor,
            filter_institution=args.filter_institution,
        )
        print(f"Экспорт успешно завершен:")
        print(f"  Файл: {result['file']}")
        print(f"  Формат: {result['format']}")
        print(f"  Всего анкет в базе: {result['total_documents']}")
        print(f"  Признаков: {result['total_features']}")
        print(f"  Всего колонок: {len(result['columns'])}")
        print(f"  Размер: {result['file_size_bytes']} байт")
        print(f"  Пропущенные признаки заполнены NaN.")
    except Exception as e:
        sys.stderr.write(f"Ошибка при экспорте датасета: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
