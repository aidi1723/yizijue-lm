#!/usr/bin/env python3
"""Build a separate 64-hexagram coverage set from kernel readings.

Pinned gateway hexagrams keep their existing labels and are not synthesized.
The other sixty hexagrams are described with inner and outer virtues, the
element relation, and the symbolic transition. Gateway actions stay deny.
"""

import json
import sys
from pathlib import Path

ONECODE_SRC = Path("/Volumes/MacSSD/项目开发/one code/src")
sys.path.insert(0, str(ONECODE_SRC))

from onecode.kernel.hexagram import IchingKernel
from onecode.kernel.project_gateway import project_gateway
from onecode.kernel.prompt_rules import classify_prompt

OUT_DIR = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/hexagram_coverage_v1")
CATALOG_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/logs/2026-09-30-hexagram-coverage-catalog.json")
PINNED = {0b000000, 0b010010, 0b100001, 0b111111}
DENY_FACTS = {
    "evidence_state": "required",
    "intent_type": "invalid_intent",
    "path_scope": "no_path",
    "sandbox_state": "not_required",
}
TRIGRAM_ZH = {
    "kun": ("坤", "承受", "接纳来势"),
    "zhen": ("震", "启动", "把静止的局面发动"),
    "kan": ("坎", "风险", "在险处设检查点"),
    "dui": ("兑", "交换", "把结果交付出去"),
    "gen": ("艮", "停止", "在边界上停住"),
    "li": ("离", "明察", "附着于对象并审视"),
    "xun": ("巽", "渗入", "进入细节里打磨"),
    "qian": ("乾", "刚健", "向前推进"),
}
ELEMENT_ZH = {"wood": "木", "fire": "火", "earth": "土", "metal": "金", "water": "水"}
RELATION_ZH = {"same": "同性", "generates": "相生", "controls": "相克", "neutral": "无直接生克"}
ACTION_ZH = {
    "discover": "先查清",
    "activate": "发动",
    "accelerate": "加快",
    "checkpoint": "停下来核验",
    "continue": "相续",
    "recover": "恢复",
    "halt": "熔断",
    "prune": "修剪范围",
    "cooldown": "冷却",
}
REASON_ZH = {
    None: "符号转移没有另给原因",
    "rule_gap_requires_discovery": "规则有缺口，要先查清",
    "yin_excess_requires_activation": "阴过盛，需要发动",
    "generating_relation_accelerates_execution": "相生的关系在加快",
    "controlled_by_relation_requires_verifier": "被克的关系需要核验",
    "wood_breaks_inert_ground": "木破开沉土",
    "generated_by_relation_recovers_execution": "被生的关系在恢复",
    "network_water_preserves_resume_seed": "坎水留住可恢复的种子",
    "water_quenches_fire_boundary": "水到达灭火的边界",
    "metal_prunes_wood_scope": "金修剪木的范围",
    "yang_overload_cooldown": "阳过载，转入冷却",
    "mountain_contains_local_executor_fault": "山挡住本地执行器的故障",
    "sovereignty_fire_boundary_halt": "主权之火到了边界，于是熔断",
    "sovereignty_fire_suppresses_asset": "主权之火压住资产",
}
MODULATION_ZH = {
    "normal": "常态",
    "hard_control": "强克",
    "recovery_seed": "留种",
    "quench": "水火相灭",
    "prune": "修剪",
    "fuel": "添薪",
    "dam": "堤挡",
    "break_ground": "破土",
}
PRESSURE_ZH = {"stable": "阴阳势平", "activate": "阴阳势偏于起势", "cooldown": "阴阳势偏于收势"}
FRAMES = (
    "内里是{inner_name}，德性为{inner_virtue}，势在{inner_verb}。外围是{outer_name}，德性为{outer_virtue}，势在{outer_verb}。{reason}，符号动作是{action}。外{outer_element}对内{inner_element}为{relation}，调制是{modulation}，{pressure}。此卦未另钉放行，只入账。",
    "当前局势里层走{inner_name}之{inner_virtue}，也就是{inner_verb}；外层走{outer_name}之{outer_virtue}，也就是{outer_verb}。读卦得到的原因是：{reason}。动作读作{action}。五行关系是外{outer_element}与内{inner_element}{relation}，调制为{modulation}。{pressure}。先记账，不放行。",
    "请按已有卦理看这句话。内卦{inner_name}主{inner_verb}，外卦{outer_name}主{outer_verb}。{reason}。因此符号动作取{action}。外层五行为{outer_element}，内层五行为{inner_element}，关系是{relation}，调制是{modulation}。{pressure}。没有新的放行钉。",
    "这不是新的执行规则，只是在描述一个卦。{inner_name}在内，{inner_virtue}，{inner_verb}；{outer_name}在外，{outer_virtue}，{outer_verb}。原因落在：{reason}。符号上是{action}。{outer_element}与{inner_element}的关系为{relation}，调制{modulation}。{pressure}。结果入账。",
)
CLOSINGS = (
    "请只保留这个卦象。",
    "不要把它写成已经获准的改动。",
    "路径和摘要都还不在这句里面。",
    "沙箱核验也不由这句单独打开。",
    "账上要留下内卦和外卦。",
    "放行与否仍交给网关投影。",
    "这里没有宿主命令，也没有越界路径。",
    "句子只提供起卦所需的德性。",
    "记录时写明符号动作。",
    "不要把调制当成放行理由。",
    "五行关系只解释这个卦，不新增动作。",
    "内外卦都要留在账上。",
    "这句不携带文件正文。",
    "这句不携带补丁搜索块。",
    "这些词只是符号读法，不另钉放行。",
    "未钉住的卦保持拒绝放行。",
)


def profile(status_code: int) -> dict[str, str]:
    inner = status_code & 0b111
    outer = (status_code >> 3) & 0b111
    inner_name = IchingKernel.TRIGRAM_NAMES[inner]
    outer_name = IchingKernel.TRIGRAM_NAMES[outer]
    symbolic = IchingKernel.transition(status_code)
    dynamics = IchingKernel.element_dynamics(status_code)
    relation = IchingKernel.element_relation(dynamics["outer_element"], dynamics["inner_element"])
    facts = dict(DENY_FACTS)
    gateway = project_gateway(status_code, facts)
    if gateway != "DENY_AND_LEDGER":
        raise SystemExit(f"{status_code:06b} gateway is {gateway}, coverage rows must deny")
    inner_zh = TRIGRAM_ZH[inner_name]
    outer_zh = TRIGRAM_ZH[outer_name]
    if symbolic.reason not in REASON_ZH:
        raise SystemExit(f"missing reason translation: {symbolic.reason}")
    return {
        "binary": format(status_code, "06b"),
        "inner_name": inner_zh[0],
        "inner_virtue": inner_zh[1],
        "inner_verb": inner_zh[2],
        "outer_name": outer_zh[0],
        "outer_virtue": outer_zh[1],
        "outer_verb": outer_zh[2],
        "reason": REASON_ZH[symbolic.reason],
        "reason_code": symbolic.reason or "none",
        "action": ACTION_ZH[symbolic.action],
        "action_code": symbolic.action,
        "outer_element": ELEMENT_ZH[dynamics["outer_element"]],
        "inner_element": ELEMENT_ZH[dynamics["inner_element"]],
        "relation": RELATION_ZH[relation],
        "modulation": MODULATION_ZH[dynamics["modulation"]],
        "pressure": PRESSURE_ZH[dynamics["yin_yang_pressure"]],
        "gateway": gateway,
    }


def line_reading(binary: str) -> str:
    names = ("初爻", "二爻", "三爻", "四爻", "五爻", "上爻")
    code = int(binary, 2)
    parts = [f"{name}{'阳' if (code >> index) & 1 else '阴'}" for index, name in enumerate(names)]
    return "，".join(parts)


def sentences(item: dict[str, str]) -> list[str]:
    rows = []
    reading = line_reading(item["binary"])
    for index in range(64):
        text = f"{CLOSINGS[index // len(FRAMES)]}{FRAMES[index % len(FRAMES)].format(**item)}六爻自下而上是{reading}。"
        rows.append(text)
    return rows


def row(sample_id: str, text: str, binary: str) -> dict:
    payload = {
        "action": {
            "action": "DENY_AND_LEDGER",
            "facts": DENY_FACTS,
            "reason": "hexagram_not_pinned",
            "yizijue_state": binary,
        },
        "output_type": "action_json",
    }
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "sample_group": "hexagram_coverage",
    }


def split_name(index: int) -> str:
    if index % 8 == 0:
        return "test"
    if index % 8 == 1:
        return "valid"
    return "train"


def main() -> int:
    catalog = []
    buckets = {"train": [], "valid": [], "test": []}
    seen = set()
    for status_code in range(64):
        if status_code in PINNED:
            catalog.append({"binary": format(status_code, "06b"), "coverage": "pinned_real_labels"})
            continue
        item = profile(status_code)
        catalog.append({key: item[key] for key in ("binary", "reason_code", "action_code", "gateway")})
        for index, text in enumerate(sentences(item)):
            if classify_prompt(text) is not None:
                raise SystemExit(f"keyword rule captured {item['binary']}: {text}")
            if text in seen:
                raise SystemExit(f"duplicate sentence for {item['binary']}")
            seen.add(text)
            name = split_name(index)
            sample_id = f"coverage-{item['binary']}-{name}-{index:02d}"
            buckets[name].append(row(sample_id, text, item["binary"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        path = OUT_DIR / f"{name}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
        print(name, len(rows), path)
    CATALOG_PATH.write_text(json.dumps({"pinned_skipped": sorted(format(code, "06b") for code in PINNED), "hexagrams": catalog}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("unique", len(seen), "catalog", CATALOG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
