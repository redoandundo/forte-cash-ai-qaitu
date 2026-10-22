"""Reviewed rule candidates. Detector scores are never sufficient facts."""
from __future__ import annotations

from copy import deepcopy

SOURCE_URL = "https://adilet.zan.kz/rus/docs/V1900019680"
DOCUMENT_NAME = "Правила кассовых операций, постановление Правления НБРК № 231 от 29.11.2019"
VERIFIED_AT = "2026-10-18"

CONTEXT_FACT = {
    "id": "full_context_visible",
    "label": "Просмотрена последовательность до и после; нужные участники и рабочая зона видны",
}
RULES = [
    {
        "id": "CASH-02", "label": "Приём и пересчёт в присутствии клиента",
        "section": "пункт 20, абзац 1",
        "summary": "Приём и пересчёт наличных выполняются при клиенте.",
        "limitation": "Отсутствие клиента в кадре не доказывает отсутствие рядом с кассой. Нужен полный обзор места обслуживания.",
        "facts": [
            {"id": "cash_acceptance_or_counting_visible", "label": "Видно, что кассир принимает или пересчитывает наличные клиента"},
            {"id": "client_absent_throughout_action", "label": "Всё место обслуживания видно; клиент отсутствует во время этого действия"},
            CONTEXT_FACT,
        ],
    },
    {
        "id": "CASH-03", "label": "Хранение ранее принятых наличных",
        "section": "пункт 20, абзац 2",
        "summary": "Ранее принятые наличные хранятся в специальном устройстве, закрывающемся на ключ.",
        "limitation": "Нельзя установить принадлежность денег по одному кадру; деньги текущего клиента допустимы на рабочем месте.",
        "facts": [
            {"id": "previous_transaction_finished", "label": "По последовательности установлено завершение обслуживания предыдущего клиента"},
            {"id": "earlier_cash_identified", "label": "Прослежены именно ранее принятые деньги, а не наличные текущего клиента"},
            {"id": "earlier_cash_outside_storage", "label": "Эти деньги оставлены вне специального закрывающегося устройства хранения"},
            CONTEXT_FACT,
        ],
    },
    {
        "id": "CASH-04", "label": "Уход до завершения обслуживания",
        "section": "пункт 22",
        "summary": "Кассир не должен уходить с рабочего места до завершения обслуживания клиента.",
        "limitation": "Пустое рабочее место или проход человека сами по себе не подтверждают нарушение; нужно установить незавершённое обслуживание.",
        "facts": [
            {"id": "cashier_leaves_workplace", "label": "Видно, что кассир покидает своё рабочее место"},
            {"id": "client_service_in_progress", "label": "По последовательности установлено: обслуживание клиента ещё не завершено"},
            CONTEXT_FACT,
        ],
    },
]


def rule_catalog() -> dict[str, object]:
    return {
        "version": "1", "verified_at": VERIFIED_AT, "document_name": DOCUMENT_NAME,
        "source_url": SOURCE_URL, "rules": deepcopy(RULES),
    }


def unmatched_rule() -> dict[str, object]:
    return {
        "status": "not_established", "label": "Нормативное основание не установлено",
        "document_name": None, "section": None, "source_url": None,
        "applicability": "Для сопоставления требуется подтвердить все наблюдаемые условия конкретного правила.",
    }


def match_reviewed_rule(rule_id: str, confirmed_facts: list[str]) -> dict[str, object]:
    rule = next((item for item in RULES if item["id"] == rule_id), None)
    if rule is None:
        raise ValueError("Unknown rule")
    required = {fact["id"] for fact in rule["facts"]}
    if len(confirmed_facts) != len(set(confirmed_facts)) or set(confirmed_facts) != required:
        raise ValueError("Confirm exactly all required observations; unknown or missing facts are not allowed")
    observations = "; ".join(fact["label"] for fact in rule["facts"])
    return {
        "status": "matched", "label": rule["label"], "document_name": DOCUMENT_NAME,
        "section": rule["section"], "source_url": SOURCE_URL,
        "applicability": (
            f"Сопоставлено проверяющим вручную. Подтверждённые наблюдения: {observations}. "
            f"Это кандидат для проверки признаков возможного нарушения. {rule['limitation']}"
        ),
    }
