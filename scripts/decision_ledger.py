"""Separate a flat cast from the kun hexagram in the decision record."""

import json
import os
from pathlib import Path

LEDGER_ENV = "YIZIJUE_LEDGER"


def ledger_reason(state: str | None, action: str | None, observe: bool, current_reason: str | None) -> str | None:
    if observe and state != "000000":
        return "entropy_observe"
    if state == "000000" and action == "DENY_AND_LEDGER" and current_reason in {None, "collapse_head"}:
        return "kun_deny_ledger"
    return current_reason


def decision_record(response: dict) -> dict:
    parsed = response.get("json") if isinstance(response.get("json"), dict) else {}
    action_body = parsed.get("action") if isinstance(parsed.get("action"), dict) else {}
    collapse = parsed.get("collapse") if isinstance(parsed.get("collapse"), dict) else {}
    state = action_body.get("yizijue_state")
    action = response.get("action")
    reason = ledger_reason(state, action, bool(collapse.get("observe")), action_body.get("reason"))
    record = {
        "yizijue_state": state,
        "action": action,
        "reason": reason,
    }
    moving = _moving_cast_for_state(state, parsed.get("moving_cast"))
    if moving is not None:
        record["moving_cast"] = moving
    symbolic = _symbolic_for_cast(state)
    if symbolic is not None:
        record["symbolic_transition"] = symbolic
    return record


def _moving_cast_for_state(state: object, moving: object) -> dict | None:
    from onecode.experimental.yizijue_ledger import accepted_moving_cast

    return accepted_moving_cast(state, moving)


def _symbolic_for_cast(state: object) -> dict | None:
    if not isinstance(state, str) or len(state) != 6 or any(bit not in "01" for bit in state):
        return None
    from onecode.experimental.yizijue_ledger import symbolic_transition

    return symbolic_transition(state)


def append_decision(response: dict, path: Path | None = None, *, record_workspace: bool = True) -> None:
    record = decision_record(response)
    entry = _ledger_entry(record)
    target = path
    if target is None:
        configured = os.environ.get(LEDGER_ENV)
        if configured:
            target = Path(configured)
    workspace = os.environ.get("YIZIJUE_WORKSPACE") if record_workspace else None
    service = os.environ.get(LEDGER_ENV)
    helper_mirrors = bool(workspace) and bool(service) and target is not None and Path(service) == target
    if target is not None and not helper_mirrors:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry if entry is not None else _rejected_decision(record), ensure_ascii=False) + "\n")
    if not workspace:
        return
    if entry is None:
        from onecode.experimental.yizijue_ledger import append_rejected_ledger_entry

        append_rejected_ledger_entry(Path(workspace), record)
        return
    from onecode.experimental.yizijue_ledger import append_yizijue_ledger

    try:
        append_yizijue_ledger(Path(workspace), record)
    except ValueError:
        return


def _rejected_decision(record: dict) -> dict:
    from onecode.experimental.yizijue_ledger import rejected_ledger_entry

    return rejected_ledger_entry(record)


def _ledger_entry(record: dict) -> dict | None:
    from onecode.experimental.yizijue_ledger import yizijue_ledger_entry

    try:
        return yizijue_ledger_entry(record)
    except ValueError:
        return None
