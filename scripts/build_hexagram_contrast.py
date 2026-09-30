#!/usr/bin/env python3
"""Add train-only contrasts for hexagrams the coverage test confused.

Each sentence names the true inner and outer trigrams and the lines that differ
from the wrong hexagram. The held-out coverage test is not changed.
"""

import json
import sys
from pathlib import Path

ONECODE_SRC = Path("/Volumes/MacSSD/项目开发/one code/src")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ONECODE_SRC))
sys.path.insert(0, str(REPO_ROOT))

from onecode.kernel.hexagram import IchingKernel
from onecode.kernel.prompt_rules import classify_prompt
from scripts.build_hexagram_coverage import line_reading, profile, row

OUT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/hexagram_coverage_v1/contrast_train.jsonl")
LINE_NAMES = ("初爻", "二爻", "三爻", "四爻", "五爻", "上爻")
# Gold, predicted. Taken from the phase-2 coverage test at temperature 1.
PAIRS = (
    (0b000010, 0b010100),
    (0b000010, 0b010101),
    (0b001000, 0b010101),
    (0b001000, 0b000001),
    (0b001010, 0b010100),
    (0b101001, 0b101010),
    (0b101100, 0b101010),
    (0b110011, 0b100110),
    (0b000001, 0b001000),
    (0b000100, 0b010100),
    (0b001001, 0b100100),
    (0b001011, 0b100110),
    (0b001110, 0b110110),
    (0b010011, 0b110010),
    (0b010110, 0b101010),
    (0b011000, 0b110010),
    (0b011001, 0b011110),
    (0b101000, 0b101010),
    (0b101110, 0b101010),
    (0b111011, 0b110111),
)
CLOSINGS = (
    "请记下这一卦，而不是旁边那一卦。",
    "内外卦的位置不能对调。",
    "不同的爻决定这是哪一卦。",
    "这句只区分两个卦，不新增放行。",
    "账上写这一卦的内外卦。",
    "另一卦的爻位不要套过来。",
    "先把易混的卦拆开。",
    "拒绝放行，但卦名要写对。",
)


def trigram_pair(status_code: int) -> str:
    inner = IchingKernel.TRIGRAM_NAMES[status_code & 0b111]
    outer = IchingKernel.TRIGRAM_NAMES[(status_code >> 3) & 0b111]
    item = profile(status_code)
    return f"内卦{item['inner_name']}（{inner}），外卦{item['outer_name']}（{outer}）"


def differing_lines(left: int, right: int) -> str:
    parts = []
    for index, name in enumerate(LINE_NAMES):
        if ((left ^ right) >> index) & 1:
            left_bit = "阳" if (left >> index) & 1 else "阴"
            right_bit = "阳" if (right >> index) & 1 else "阴"
            parts.append(f"{name}这一卦是{left_bit}，另一卦是{right_bit}")
    if not parts:
        raise SystemExit(f"{left:06b} and {right:06b} do not differ")
    return "；".join(parts)


def sentence(gold: int, other: int, closing: str) -> str:
    item = profile(gold)
    return (
        f"{closing}这一卦是{trigram_pair(gold)}。"
        f"不要读成{trigram_pair(other)}。"
        f"不同之处：{differing_lines(gold, other)}。"
        f"这一卦六爻自下而上是{line_reading(item['binary'])}。"
        f"{item['reason']}，符号动作是{item['action']}。未另钉放行，只入账。"
    )


def main() -> int:
    rows = []
    seen = set()
    for gold, other in PAIRS:
        for index, closing in enumerate(CLOSINGS):
            text = sentence(gold, other, closing)
            if classify_prompt(text) is not None:
                raise SystemExit(f"keyword rule captured {gold:06b}: {text}")
            if text in seen:
                raise SystemExit("duplicate contrast")
            seen.add(text)
            rows.append(row(f"contrast-{gold:06b}-not-{other:06b}-{index:02d}", text, format(gold, "06b")))
    OUT_PATH.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
    print(len(rows), OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
