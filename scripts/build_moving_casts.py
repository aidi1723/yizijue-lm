#!/usr/bin/env python3
"""Enumerate every six-line Da Yan cast and hold some out for evaluation."""

import json
import sys
from pathlib import Path

ONECODE_SRC = Path("/Volumes/MacSSD/项目开发/one code/src")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ONECODE_SRC))
sys.path.insert(0, str(REPO_ROOT))

from onecode.experimental.moving_cast import LINE_VALUES, cast_from_lines
from onecode.kernel.hexagram import IchingKernel
from onecode.kernel.prompt_rules import classify_prompt

OUT_DIR = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/hexagram_moving_v1")
LINE_NAMES = ("初爻", "二爻", "三爻", "四爻", "五爻", "上爻")
VALUE_ZH = {6: "老阴，动", 7: "少阳，静", 8: "少阴，静", 9: "老阳，动"}


def values_for(index: int) -> list[int]:
    values = []
    remainder = index
    for _ in range(6):
        values.append(LINE_VALUES[remainder % 4])
        remainder //= 4
    return values


def split_name(index: int) -> str:
    if index % 8 == 0:
        return "test"
    if index % 8 == 1:
        return "valid"
    return "train"


def sentence(values: list[int]) -> str:
    parts = [f"{name}{VALUE_ZH[value]}" for name, value in zip(LINE_NAMES, values)]
    return "起卦六爻自下而上为" + "，".join(parts) + "。动爻翻转后仍落在六十四卦里。"


def main() -> int:
    buckets = {"train": [], "valid": [], "test": []}
    seen = set()
    for index in range(4**6):
        values = values_for(index)
        cast = cast_from_lines(values)
        if cast["after"] != IchingKernel.mutate_lines(cast["before"], cast["moving"]):
            raise SystemExit("moving cast drifted from the kernel")
        text = sentence(values)
        if classify_prompt(text) is not None:
            raise SystemExit(f"keyword rule captured cast {index}")
        if text in seen:
            raise SystemExit("duplicate cast text")
        seen.add(text)
        name = split_name(index)
        buckets[name].append(
            {
                "id": f"moving-{index:04d}",
                "text": text,
                "values": values,
                "before": format(cast["before"], "06b"),
                "after": format(cast["after"], "06b"),
                "moving": cast["moving"],
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        path = OUT_DIR / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(name, len(rows))
    print("casts", 4**6, "unique", len(seen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
