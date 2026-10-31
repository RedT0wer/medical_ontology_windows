"""Интерактивное модальное окно арбитража спорных терминов и иерархии онтологии (Human-in-the-loop)."""

import sys
from typing import Any, List, Optional
from src.config import logger


def render_cli_arbitration_modal(
    raw_key: str,
    raw_value: str,
    context: str,
    candidates: List[Any],
    llm_action: str,
    llm_target_id: Optional[int] = None,
    llm_parent_id: Optional[int] = None,
    llm_new_parent_name: Optional[str] = None,
    llm_canonical_name: str = "",
    llm_reason: str = "",
    timeout_sec: int = 15,
) -> dict:
    """Отображение интерактивного псевдографического модального окна в консоли для эксперта."""
    width = 82
    border_h = "═" * (width - 2)
    border_s = "─" * (width - 2)

    lines = []
    lines.append(f"╔{border_h}╗")
    title = " МОДАЛЬНОЕ ОКНО: КЛИНИЧЕСКИЙ АРБИТРАЖ (СЕРАЯ ЗОНА И ИЕРАРХИЯ) "
    lines.append(f"║{title.center(width - 2)}║")
    lines.append(f"╠{border_h}╣")

    lines.append(f"║ Сырой термин: '{raw_key}' (значение: '{raw_value}')".ljust(width - 1) + "║")
    ctx_preview = (context[:67] + "...") if len(context) > 67 else context
    lines.append(f"║ Контекст: \"{ctx_preview}\"".ljust(width - 1) + "║")
    lines.append(f"╟{border_s}╢")

    lines.append("║ Найденные кандидаты в онтологии (Top-3):".ljust(width - 1) + "║")
    for i, c in enumerate(candidates, 1):
        c_line = f"║   [{i}] ID {c.parameter_id}: '{c.canonical_name}' (сходство {c.similarity:.4f} через '{c.matched_text}')"
        lines.append(c_line.ljust(width - 1) + "║")

    lines.append(f"╟{border_s}╢")
    lines.append("║ Рекомендация LLM-арбитра (Qwen 2.5):".ljust(width - 1) + "║")
    if llm_action == "NEW_PARENT_AND_CHILD":
        lines.append(f"║   Действие: НОВЫЙ РОДИТЕЛЬ И ПОДКЛАСС -> Родитель: '{llm_new_parent_name}', Подкласс: '{llm_canonical_name}'".ljust(width - 1) + "║")
    elif llm_action == "CHILD_OF":
        lines.append(f"║   Действие: CHILD_OF (Подкласс) -> Родитель ID {llm_parent_id}, Подкласс: '{llm_canonical_name}'".ljust(width - 1) + "║")
    elif llm_action == "SYNONYM":
        lines.append(f"║   Действие: SYNONYM (Синоним) -> ID {llm_target_id} ('{llm_canonical_name}')".ljust(width - 1) + "║")
    else:
        lines.append(f"║   Действие: NEW (Новый независимый параметр) -> '{llm_canonical_name}'".ljust(width - 1) + "║")

    reason_prev = (llm_reason[:67] + "...") if len(llm_reason) > 67 else llm_reason
    lines.append(f"║   Обоснование: {reason_prev}".ljust(width - 1) + "║")

    lines.append(f"╠{border_h}╣")
    lines.append("║ Выберите решение эксперта-клинициста:".ljust(width - 1) + "║")
    lines.append("║   [1] Подтвердить вердикт LLM (Enter по умолчанию)".ljust(width - 1) + "║")
    lines.append("║   [2] Привязать как СИНОНИМ к существующему параметру (указать ID)".ljust(width - 1) + "║")
    lines.append("║   [3] Сделать ДОЧЕРНИМ подклассом СУЩЕСТВУЮЩЕГО параметра (указать ID)".ljust(width - 1) + "║")
    lines.append("║   [4] Создать НОВЫЙ РОДИТЕЛЬСКИЙ класс (например, 'Тяжесть') и привязать подкласс".ljust(width - 1) + "║")
    lines.append("║   [5] Создать независимый НОВЫЙ канонический параметр (без родителя)".ljust(width - 1) + "║")
    lines.append("║   [6] Пропустить этот показатель".ljust(width - 1) + "║")
    lines.append(f"╚{border_h}╝")

    modal_str = "\n".join(lines)
    print("\n" + modal_str)

    # Если передан флаг --json или терминал не интерактивен, не блокируем ввод
    if "--json" in sys.argv or not sys.stdin.isatty():
        logger.info(f"[МОДАЛЬНОЕ ОКНО] Автоматически принято решение LLM ({llm_action}).")
        return {
            "action": llm_action,
            "parameter_id": llm_target_id,
            "parent_id": llm_parent_id,
            "new_parent_name": llm_new_parent_name,
            "canonical_name": llm_canonical_name,
            "reason": f"Подтверждено экспертом: {llm_reason}",
        }

    try:
        choice = input("\n[Эксперт] Ваш выбор [1/2/3/4/5/6] (по умолчанию 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "2":
        try:
            param_id_str = input("  Введите целевой ID параметра из таблицы: ").strip()
            param_id = int(param_id_str)
            return {
                "action": "SYNONYM",
                "parameter_id": param_id,
                "parent_id": None,
                "new_parent_name": None,
                "canonical_name": llm_canonical_name,
                "reason": "Переопределено экспертом: привязан как синоним",
            }
        except ValueError:
            print("  Некорректный ID. Применено решение LLM.")
    elif choice == "3":
        try:
            p_id_str = input("  Введите ID существующего родительского параметра: ").strip()
            p_id = int(p_id_str)
            subclass_name = input("  Введите каноническое наименование подкласса (Enter для текущего): ").strip()
            final_name = subclass_name if subclass_name else llm_canonical_name
            return {
                "action": "CHILD_OF",
                "parameter_id": None,
                "parent_id": p_id,
                "new_parent_name": None,
                "canonical_name": final_name,
                "reason": f"Задано экспертом: подкласс родителя ID {p_id}",
            }
        except ValueError:
            print("  Некорректный ID родителя. Применено решение LLM.")
    elif choice == "4":
        p_name = input("  Введите наименование НОВОГО родительского класса (например: 'Тяжесть'): ").strip()
        if p_name:
            subclass_name = input(f"  Введите наименование подкласса (по умолчанию '{llm_canonical_name}'): ").strip()
            final_subclass = subclass_name if subclass_name else llm_canonical_name
            return {
                "action": "NEW_PARENT_AND_CHILD",
                "parameter_id": None,
                "parent_id": None,
                "new_parent_name": p_name,
                "canonical_name": final_subclass,
                "reason": f"Создан новый родительский класс '{p_name}' и подкласс '{final_subclass}'",
            }
    elif choice == "5":
        custom_name = input("  Введите каноническое наименование нового параметра: ").strip()
        if custom_name:
            return {
                "action": "NEW",
                "parameter_id": None,
                "parent_id": None,
                "new_parent_name": None,
                "canonical_name": custom_name,
                "reason": f"Сформулировано экспертом вручную: {custom_name}",
            }
    elif choice == "6":
        return {
            "action": "SKIP",
            "parameter_id": None,
            "parent_id": None,
            "new_parent_name": None,
            "canonical_name": "",
            "reason": "Пропущено экспертом",
        }

    # По умолчанию принимаем рекомендацию LLM
    return {
        "action": llm_action,
        "parameter_id": llm_target_id,
        "parent_id": llm_parent_id,
        "new_parent_name": llm_new_parent_name,
        "canonical_name": llm_canonical_name,
        "reason": f"Подтверждено экспертом: {llm_reason}",
    }
