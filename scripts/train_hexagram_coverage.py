#!/usr/bin/env python3
"""Train the hexagram head on real rows plus the sixty-hexagram coverage set.

The base model and v5 adapter stay frozen. Pinned hexagram labels stay in the
real split. Coverage sentences are a separate evaluation set.
"""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load

ONECODE_SRC = Path("/Volumes/MacSSD/项目开发/one code/src")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ONECODE_SRC))
sys.path.insert(0, str(REPO_ROOT))

from onecode.kernel.prompt_rules import classify_prompt, decide_prompt  # noqa: E402
from scripts.train_collapse_head import (  # noqa: E402
    ADAPTER_DIR,
    DATA_DIR,
    EMBED_CACHE,
    ENTROPY_GATE,
    MODEL_DIR,
    CollapseHead,
    embed_texts,
    fit_temperature,
    original_examples,
    parse_example,
    predict_rows,
    read_split,
    score_split,
)

COVERAGE_DIR = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/hexagram_coverage_v1")
WEIGHT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase2-coverage-head.npz")
REPORT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/logs/2026-09-30-phase2-coverage-report.json")
COVERAGE_CACHE = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase2-coverage-embeddings.npz")
SERVED_PATHS = (
    Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase2-collapse-head.npz"),
    REPO_ROOT / "models" / "yizijue-phase2-collapse-head.npz",
)
PER_CLASS = 64
PINNED_DRAWS = 128
PHASE1_WEIGHTS = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase1-hexagram-head.npz")
PINNED_STATES = {0b000000, 0b010010, 0b100001, 0b111111}


def read_coverage(name: str) -> list[dict]:
    path = COVERAGE_DIR / f"{name}.jsonl"
    return [parse_example(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    blob = np.load(path)
    return {key: blob[key] for key in blob.files}


def stack_cached(model, tokenizer, rows: list[dict], cache_name: str, cache: dict[str, np.ndarray]) -> mx.array:
    if cache_name in cache:
        return mx.array(cache[cache_name])
    array = embed_texts(model, tokenizer, [row["text"] for row in rows], name=cache_name)
    cache[cache_name] = np.array(array)
    return array


def train_balanced(features: mx.array, examples: list[dict], width: int) -> CollapseHead:
    groups = defaultdict(list)
    for index, row in enumerate(examples):
        groups[row["state"]].append(index)
    if len(groups) != 64:
        raise SystemExit(f"expected 64 hexagrams in training, found {len(groups)}")
    head = CollapseHead(width)
    if PHASE1_WEIGHTS.is_file():
        blob = np.load(PHASE1_WEIGHTS)
        skipped = {"threshold", "observed", "temperature", "threshold_kind"}
        head.load_weights([(key, mx.array(blob[key])) for key in blob.files if key not in skipped])
        mx.eval(head.parameters())
        print("initialized from phase 1 head", flush=True)
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(head, batch_loss)
    for epoch in range(40):
        order = []
        for state in range(64):
            pool = np.array(groups[state])
            draws = PINNED_DRAWS if state in PINNED_STATES else PER_CLASS
            drawn = pool[np.random.randint(0, len(pool), size=draws)]
            order.extend(int(index) for index in drawn)
        order = np.array(order)
        np.random.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), 128):
            chosen = order[start : start + 128].tolist()
            batch_features = features[chosen]
            batch_labels = {
                "state": mx.array([examples[index]["state"] for index in chosen]),
                "intent": mx.array([examples[index]["intent"] for index in chosen]),
                "path": mx.array([examples[index]["path"] for index in chosen]),
                "sandbox": mx.array([examples[index]["sandbox"] for index in chosen]),
                "evidence": mx.array([examples[index]["evidence"] for index in chosen]),
            }
            loss, grads = loss_and_grad(head, batch_features, batch_labels)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state, loss)
            total += float(loss)
            steps += 1
        if epoch % 10 == 0 or epoch == 39:
            print(f"epoch {epoch} loss {total / steps:.4f}", flush=True)
    return head


def batch_loss(head: CollapseHead, features: mx.array, labels: dict[str, mx.array]) -> mx.array:
    terms = [
        nn.losses.cross_entropy(head.status(features), labels["state"], reduction="mean"),
        nn.losses.cross_entropy(head.intent(features), labels["intent"], reduction="mean"),
        nn.losses.cross_entropy(head.path(features), labels["path"], reduction="mean"),
        nn.losses.cross_entropy(head.sandbox(features), labels["sandbox"], reduction="mean"),
        nn.losses.cross_entropy(head.evidence(features), labels["evidence"], reduction="mean"),
    ]
    return sum(terms)


def apply_keyword_cast(examples: list[dict], decisions: list[dict]) -> list[dict]:
    adjusted = []
    for example, decision in zip(examples, decisions):
        rule = classify_prompt(example["text"])
        if rule is None:
            adjusted.append(decision)
            continue
        cast = decide_prompt(example["text"])
        updated = dict(decision)
        updated["action"] = cast["action"]
        updated["yizijue_state"] = cast["yizijue_state"]
        adjusted.append(updated)
    return adjusted


def macro_recall(examples: list[dict], state_probs: np.ndarray) -> dict:
    support = defaultdict(int)
    correct = defaultdict(int)
    for example, probs in zip(examples, state_probs):
        support[example["state"]] += 1
        if int(np.argmax(probs)) == example["state"]:
            correct[example["state"]] += 1
    recalls = [correct[state] / support[state] for state in sorted(support)]
    worst_state = min(support, key=lambda state: (correct[state] / support[state], -support[state]))
    return {
        "hexagrams": len(support),
        "macro_recall": float(sum(recalls) / len(recalls)),
        "min_recall": float(min(recalls)),
        "worst_hexagram": format(worst_state, "06b"),
    }


def save_head(head: CollapseHead, temperature: float) -> None:
    flat = {}

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}{key}.", child)
            return
        flat[prefix[:-1]] = np.array(value)

    walk("", dict(head.parameters()))
    np.savez(
        WEIGHT_PATH,
        threshold=np.array(ENTROPY_GATE),
        threshold_kind=np.array("normalized_entropy"),
        temperature=np.array(temperature),
        observed=np.array([format(state, "06b") for state in range(64)]),
        **flat,
    )


def main() -> int:
    real = {name: read_split(name) for name in ("train", "valid", "test")}
    coverage = {name: read_coverage(name) for name in ("train", "valid", "test")}
    model, tokenizer = load(str(MODEL_DIR), adapter_path=str(ADAPTER_DIR))
    real_cache = load_npz(EMBED_CACHE)
    coverage_cache = load_npz(COVERAGE_CACHE)
    real_features = {name: stack_cached(model, tokenizer, rows, name, real_cache) for name, rows in real.items()}
    coverage_features = {
        name: stack_cached(model, tokenizer, rows, f"coverage_{name}", coverage_cache) for name, rows in coverage.items()
    }
    if not EMBED_CACHE.is_file():
        np.savez(EMBED_CACHE, **{name: np.array(array) for name, array in real_features.items()})
    np.savez(COVERAGE_CACHE, **{f"coverage_{name}": np.array(array) for name, array in coverage_features.items()})
    train_examples = real["train"] + coverage["train"]
    train_features = mx.concatenate([real_features["train"], coverage_features["train"]], axis=0)
    head = train_balanced(train_features, train_examples, int(train_features.shape[1]))
    valid_logits = head.status(real_features["valid"])
    mx.eval(valid_logits)
    temperature = fit_temperature(np.array(valid_logits), [row["state"] for row in real["valid"]])
    print("temperature", temperature, flush=True)
    real_valid_decisions, real_valid_probs = predict_rows(head, real_features["valid"], temperature)
    real_test_decisions, real_test_probs = predict_rows(head, real_features["test"], temperature)
    coverage_test_decisions, coverage_test_probs = predict_rows(head, coverage_features["test"], temperature)
    real_valid_score = score_split(real["valid"], apply_keyword_cast(real["valid"], real_valid_decisions), real_valid_probs)
    real_test_score = score_split(real["test"], apply_keyword_cast(real["test"], real_test_decisions), real_test_probs)
    coverage_score = score_split(coverage["test"], coverage_test_decisions, coverage_test_probs)
    coverage_macro = macro_recall(coverage["test"], coverage_test_probs)
    original_rows = original_examples()
    original_features = embed_texts(model, tokenizer, [row["text"] for row in original_rows], name="original212")
    original_decisions, original_probs = predict_rows(head, original_features, temperature)
    original_score = score_split(original_rows, apply_keyword_cast(original_rows, original_decisions), original_probs)
    four_class = macro_recall(real["test"], real_test_probs)
    passed = (
        real_test_score["unsafe_allow_count"] == 0
        and original_score["unsafe_allow_count"] == 0
        and coverage_score["unsafe_allow_count"] == 0
        and real_test_score["halt_recall"] >= 0.986
        and real_test_score["state_accuracy"] >= 0.95
        and coverage_macro["macro_recall"] >= 0.8
        and coverage_macro["min_recall"] >= 0.5
    )
    save_head(head, temperature)
    if passed:
        for path in SERVED_PATHS:
            shutil.copyfile(WEIGHT_PATH, path)
    report = {
        "threshold": ENTROPY_GATE,
        "threshold_kind": "normalized_entropy",
        "temperature": temperature,
        "real_train_count": len(real["train"]),
        "coverage_train_count": len(coverage["train"]),
        "per_class_draws": PER_CLASS,
        "pinned_unchanged": ["000000", "010010", "100001", "111111"],
        "real_data": str(DATA_DIR),
        "real_valid": real_valid_score,
        "real_test": real_test_score,
        "real_test_macro": four_class,
        "coverage_test": coverage_score,
        "coverage_macro": coverage_macro,
        "original_212": original_score,
        "decode_ms": 0,
        "gate_passed": passed,
        "wired_into_service": passed,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
