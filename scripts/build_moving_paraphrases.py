#!/usr/bin/env python3
"""Same Da Yan casts, written in more than one sentence shape.

The held-out file uses a phrasing that never appears in training.
"""

import json
import sys
from pathlib import Path

ONECODE_SRC = Path("/Volumes/MacSSD/项目开发/one code/src")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ONECODE_SRC))
sys.path.insert(0, str(REPO_ROOT))

from onecode.experimental.moving_cast import LINE_NAMES, LINE_VALUES, cast_from_lines
from onecode.kernel.hexagram import IchingKernel
from onecode.kernel.prompt_rules import classify_prompt

OUT_DIR = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/hexagram_moving_v2")
VALUE_ZH = {6: ("老阴", "动"), 7: ("少阳", "静"), 8: ("少阴", "静"), 9: ("老阳", "动")}


def values_for(index: int) -> list[int]:
    values = []
    remainder = index
    for _ in range(6):
        values.append(LINE_VALUES[remainder % 4])
        remainder //= 4
    return values


def clauses(values: list[int], pattern: str) -> list[str]:
    rows = []
    for name, value in zip(LINE_NAMES, values):
        label, motion = VALUE_ZH[value]
        if pattern == "tight":
            rows.append(f"{name}{label}，{motion}")
        elif pattern == "shi":
            rows.append(f"{name}是{label}，{motion}")
        elif pattern == "wei":
            rows.append(f"{name}为{label}，记作{motion}")
        else:
            rows.append(f"{name}属于{label}，此爻{motion}")
    return rows


def sentence(values: list[int], pattern: str) -> str:
    parts = clauses(values, pattern)
    if pattern == "tight":
        return "起卦六爻自下而上为" + "，".join(parts) + "。翻转后仍落在六十四卦里。"
    if pattern == "shi":
        return "从下往上看：" + "；".join(parts) + "。静爻保持原卦。"
    if pattern == "wei":
        return "。".join(reversed(parts)) + "。六爻都已写明。"
    return "本次占得：" + "、".join(parts) + "。之卦只由这些爻决定。"


def row(index: int, pattern: str) -> dict:
    values = values_for(index)
    cast = cast_from_lines(values)
    if cast["after"] != IchingKernel.mutate_lines(cast["before"], cast["moving"]):
        raise SystemExit("moving cast drifted from the kernel")
    text = sentence(values, pattern)
    if classify_prompt(text) is not None:
        raise SystemExit(f"keyword rule captured {pattern} {index}")
    return {
        "id": f"moving-{pattern}-{index:04d}",
        "text": text,
        "values": values,
        "before": format(cast["before"], "06b"),
        "after": format(cast["after"], "06b"),
        "moving": cast["moving"],
        "pattern": pattern,
    }


def main() -> int:
    train = []
    valid = []
    test = []
    seen = set()
    for index in range(4**6):
        slot = index % 8
        if slot == 0:
            produced = [row(index, "heldout")]
            test.extend(produced)
        elif slot == 1:
            produced = [row(index, "shi")]
            valid.extend(produced)
        else:
            produced = [row(index, pattern) for pattern in ("tight", "shi", "wei")]
            train.extend(produced)
        for item in produced:
            if item["text"] in seen:
                raise SystemExit("duplicate cast text")
            seen.add(item["text"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        path = OUT_DIR / f"{name}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
        print(name, len(rows), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
