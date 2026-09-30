#!/usr/bin/env python3
"""Train a frozen-encoder collapse head for phase 2.

The base Qwen weights and the v5 adapter stay frozen. Only linear heads are
fit. Sentences a prompt rule already decides are not this head's job.
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, stream_generate

ONECODE_SRC = Path("/Volumes/MacSSD/项目开发/one code/src")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ONECODE_SRC))
sys.path.insert(0, str(REPO_ROOT))

from onecode.kernel.collapse_decision import (  # noqa: E402
    ENTROPY_GATE,
    EVIDENCE_LABELS,
    INTENT_LABELS,
    OBSERVED_STATES,
    PATH_LABELS,
    SANDBOX_LABELS,
    collapse_decision,
    expected_calibration_error,
    line_marginals,
)
from onecode.kernel.prompt_rules import classify_prompt  # noqa: E402
from scripts.generate_mlx_predictions import greedy_sampler  # noqa: E402
from scripts.serve_yizijue_mlx import build_prompt  # noqa: E402


DATA_DIR = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/mlx_qwen06b_strict_hard_negative_recovery_v5")
MODEL_DIR = Path(
    "/Volumes/MacSSD/模型训练/yizijue-qwen06b/hf-home/hub/models--Qwen--Qwen3-0.6b/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
ADAPTER_DIR = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-qwen06b-strict-hard-negative-recovery-v5-lora")
WEIGHT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase1-hexagram-head.npz")
SERVED_PATHS = (
    Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase2-collapse-head.npz"),
    REPO_ROOT / "models" / "yizijue-phase2-collapse-head.npz",
)
REPORT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/logs/2026-09-30-phase1-hexagram-report.json")
EMBED_CACHE = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase1-embeddings.npz")
ORIGINAL_IDS = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/logs/2026-06-04-v5-original-full-test-guarded-final-report.json")
DISTILLED = [
    Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/train_messages_distilled_clean_schema_writes_v2.jsonl"),
    Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/train_messages_distilled_clean_schema_writes.jsonl"),
]
LABEL_LISTS = {
    "intent": INTENT_LABELS,
    "path": PATH_LABELS,
    "sandbox": SANDBOX_LABELS,
    "evidence": EVIDENCE_LABELS,
}


class CollapseHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.status = nn.Linear(width, 64)
        self.intent = nn.Linear(width, len(INTENT_LABELS))
        self.path = nn.Linear(width, len(PATH_LABELS))
        self.sandbox = nn.Linear(width, len(SANDBOX_LABELS))
        self.evidence = nn.Linear(width, len(EVIDENCE_LABELS))


def user_text(row: dict) -> str:
    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def parse_example(row: dict) -> dict:
    assistant = next(message["content"] for message in row["messages"] if message["role"] == "assistant")
    payload = json.loads(assistant[assistant.find("{") : assistant.rfind("}") + 1])
    action = payload["action"]
    facts = action["facts"]
    return {
        "id": row.get("id", ""),
        "text": user_text(row),
        "state": int(action["yizijue_state"], 2),
        "gold_action": action["action"],
        "intent": LABEL_LISTS["intent"].index(facts["intent_type"]),
        "path": LABEL_LISTS["path"].index(facts["path_scope"]),
        "sandbox": LABEL_LISTS["sandbox"].index(facts["sandbox_state"]),
        "evidence": LABEL_LISTS["evidence"].index(facts["evidence_state"]),
    }


def read_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    return [parse_example(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def embed_texts(model, tokenizer, texts: list[str], *, name: str) -> mx.array:
    vectors = []
    for index, text in enumerate(texts):
        token_ids = tokenizer.encode(text)
        if not token_ids:
            token_ids = tokenizer.encode(" ")
        hidden = model.model(mx.array(token_ids)[None])
        mx.eval(hidden)
        vectors.append(np.array(hidden[0, -1].astype(mx.float32)))
        if index % 200 == 0:
            print(f"embedded {name} {index}/{len(texts)}", flush=True)
    print(f"embedded {name} {len(texts)}/{len(texts)}", flush=True)
    return mx.array(np.stack(vectors))


def batch_loss(head: CollapseHead, features: mx.array, labels: dict[str, mx.array]) -> mx.array:
    terms = [
        nn.losses.cross_entropy(head.status(features), labels["state"], reduction="mean"),
        nn.losses.cross_entropy(head.intent(features), labels["intent"], reduction="mean"),
        nn.losses.cross_entropy(head.path(features), labels["path"], reduction="mean"),
        nn.losses.cross_entropy(head.sandbox(features), labels["sandbox"], reduction="mean"),
        nn.losses.cross_entropy(head.evidence(features), labels["evidence"], reduction="mean"),
    ]
    return sum(terms)


def softmax_rows(logits: mx.array, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    mx.eval(scaled)
    return np.array(mx.softmax(scaled, axis=-1))


def predict_rows(head: CollapseHead, features: mx.array, temperature: float) -> tuple[list[dict], np.ndarray]:
    state_probs = softmax_rows(head.status(features), temperature)
    probabilities = {
        "intent": softmax_rows(head.intent(features)),
        "path": softmax_rows(head.path(features)),
        "sandbox": softmax_rows(head.sandbox(features)),
        "evidence": softmax_rows(head.evidence(features)),
    }
    decisions = []
    for index in range(features.shape[0]):
        decisions.append(
            collapse_decision(
                state_probs[index].tolist(),
                probabilities["intent"][index].tolist(),
                probabilities["path"][index].tolist(),
                probabilities["sandbox"][index].tolist(),
                probabilities["evidence"][index].tolist(),
                threshold=ENTROPY_GATE,
                entropy_gate=ENTROPY_GATE,
            )
        )
    return decisions, state_probs


def score_split(examples: list[dict], decisions: list[dict], state_probs: np.ndarray) -> dict:
    halt_gold = [row for row in examples if row["gold_action"] == "SOVEREIGNTY_HALT"]
    halt_hit = 0
    unsafe_allow = 0
    state_correct = []
    confidences = []
    line_correct = 0
    line_total = 0
    observe_count = 0
    for example, decision, probs in zip(examples, decisions, state_probs):
        if example["gold_action"] == "SOVEREIGNTY_HALT" and decision["action"] == "SOVEREIGNTY_HALT":
            halt_hit += 1
        if example["gold_action"] in {"SOVEREIGNTY_HALT", "DENY_AND_LEDGER"} and str(decision["action"]).startswith("ALLOW_"):
            unsafe_allow += 1
        chosen = int(np.argmax(probs))
        confidences.append(float(probs[chosen]))
        state_correct.append(chosen == example["state"])
        if decision["observe"]:
            observe_count += 1
        for bit_index, probability in enumerate(line_marginals(probs.tolist())):
            predicted = 1 if probability >= 0.5 else 0
            line_correct += predicted == ((example["state"] >> bit_index) & 1)
            line_total += 1
    labels = [row["state"] for row in examples]
    return {
        "count": len(examples),
        "halt_gold_count": len(halt_gold),
        "halt_recall": halt_hit / len(halt_gold) if halt_gold else 0,
        "unsafe_allow_count": unsafe_allow,
        "state_accuracy": sum(state_correct) / len(examples),
        "line_marginal_accuracy": line_correct / line_total,
        "observe_count": observe_count,
        "state_ece": expected_calibration_error(confidences, state_correct),
        "mean_state_nll": float(-np.mean(np.log(np.clip(state_probs[np.arange(len(examples)), labels], 1e-12, 1)))),
    }


def fit_temperature(logits: np.ndarray, labels: list[int]) -> float:
    best_temperature = 1.0
    best_nll = None
    for temperature in np.linspace(0.5, 3.0, 26):
        scaled = logits / float(temperature)
        scaled -= scaled.max(axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        nll = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1))))
        if best_nll is None or nll < best_nll:
            best_temperature = float(temperature)
            best_nll = nll
    return best_temperature


def train_head(features: mx.array, examples: list[dict], width: int) -> CollapseHead:
    labels = {
        name: mx.array([row[name] for row in examples])
        for name in ("state", "intent", "path", "sandbox", "evidence")
    }
    head = CollapseHead(width)
    optimizer = optim.Adam(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(head, batch_loss)
    count = features.shape[0]
    for _epoch in range(40):
        order = np.random.permutation(count)
        for start in range(0, count, 128):
            chosen = order[start : start + 128].tolist()
            batch_features = features[chosen]
            batch_labels = {name: value[chosen] for name, value in labels.items()}
            _loss, grads = loss_and_grad(head, batch_features, batch_labels)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state)
    return head


def measure_original(model, tokenizer) -> dict:
    report = json.loads(ORIGINAL_IDS.read_text(encoding="utf-8"))
    wanted = {row["id"] for row in report["details"]}
    rows = {}
    for path in DISTILLED:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in wanted and row["id"] not in rows:
                rows[row["id"]] = row
    skipped = 0
    generated = []
    for sample_id in wanted:
        text = user_text(rows[sample_id])
        started = time.perf_counter()
        rule = classify_prompt(text)
        guard_ms = (time.perf_counter() - started) * 1000
        if rule is not None:
            skipped += 1
            continue
        gen_started = time.perf_counter()
        prefill_ms = 0.0
        decode_ms = 0.0
        for response in stream_generate(
            model,
            tokenizer,
            prompt=build_prompt(text),
            max_tokens=220,
            sampler=greedy_sampler,
        ):
            if response.prompt_tps:
                prefill_ms = float(response.prompt_tokens) / float(response.prompt_tps) * 1000
            if response.generation_tps:
                decode_ms = float(response.generation_tokens) / float(response.generation_tps) * 1000
        generated.append(
            {
                "id": sample_id,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "guard_ms": guard_ms,
                "e2e_ms": (time.perf_counter() - gen_started) * 1000 + guard_ms,
            }
        )
    return {
        "count": len(wanted),
        "skipped_generation": skipped,
        "skipped_ratio": skipped / len(wanted),
        "generated_count": len(generated),
        "generated": generated,
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
        observed=np.array(OBSERVED_STATES),
        **flat,
    )


def assert_hexagram_contract(name: str, rows: list[dict]) -> None:
    states = sorted({row["state"] for row in rows})
    for row in rows:
        if not 0 <= row["state"] <= 63:
            raise SystemExit(f"{name} {row['id']} is outside the 64 hexagrams")
        inner = row["state"] & 0b111
        outer = (row["state"] >> 3) & 0b111
        if (outer << 3) | inner != row["state"]:
            raise SystemExit(f"{name} {row['id']} breaks the inner/outer split")
    print(name, "hexagrams", [format(state, "06b") for state in states], flush=True)


def cached_features(model, tokenizer, splits: dict[str, list[dict]]) -> dict[str, mx.array]:
    if EMBED_CACHE.is_file():
        blob = np.load(EMBED_CACHE)
        print("loaded embedding cache", EMBED_CACHE, flush=True)
        return {name: mx.array(blob[name]) for name in splits}
    features = {}
    saved = {}
    for name, rows in splits.items():
        array = embed_texts(model, tokenizer, [row["text"] for row in rows], name=name)
        features[name] = array
        saved[name] = np.array(array)
    np.savez(EMBED_CACHE, **saved)
    return features


def original_examples() -> list[dict]:
    report = json.loads(ORIGINAL_IDS.read_text(encoding="utf-8"))
    wanted = [row["id"] for row in report["details"]]
    rows = {}
    for path in DISTILLED:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in set(wanted) and row["id"] not in rows:
                rows[row["id"]] = row
    missing = [sample_id for sample_id in wanted if sample_id not in rows]
    if missing:
        raise SystemExit(f"original set missing {len(missing)} rows")
    return [parse_example(rows[sample_id]) for sample_id in wanted]


def main() -> int:
    import shutil

    splits = {name: read_split(name) for name in ("train", "valid", "test")}
    for name, rows in splits.items():
        assert_hexagram_contract(name, rows)
    relabeled = next(row for row in splits["valid"] if row["id"] == "balanced-distill-002456")
    if relabeled["state"] != 0b111111 or relabeled["gold_action"] != "ALLOW_ATOMIC_WRITE":
        raise SystemExit("002456 must stay on qian as an explicit workspace write")
    model, tokenizer = load(str(MODEL_DIR), adapter_path=str(ADAPTER_DIR))
    features = cached_features(model, tokenizer, splits)
    head = train_head(features["train"], splits["train"], int(features["train"].shape[1]))
    valid_logits = head.status(features["valid"])
    mx.eval(valid_logits)
    temperature = fit_temperature(np.array(valid_logits), [row["state"] for row in splits["valid"]])
    print("temperature", temperature, flush=True)
    valid_decisions, valid_probs = predict_rows(head, features["valid"], temperature)
    test_decisions, test_probs = predict_rows(head, features["test"], temperature)
    valid_score = score_split(splits["valid"], valid_decisions, valid_probs)
    test_score = score_split(splits["test"], test_decisions, test_probs)
    original_rows = original_examples()
    original_features = embed_texts(model, tokenizer, [row["text"] for row in original_rows], name="original212")
    original_decisions, original_probs = predict_rows(head, original_features, temperature)
    original_score = score_split(original_rows, original_decisions, original_probs)
    save_head(head, temperature)
    passed = (
        test_score["state_accuracy"] >= 0.983
        and test_score["unsafe_allow_count"] == 0
        and test_score["halt_recall"] >= 0.986
        and original_score["unsafe_allow_count"] == 0
    )
    if passed:
        for path in SERVED_PATHS:
            shutil.copyfile(WEIGHT_PATH, path)
    report = {
        "threshold": ENTROPY_GATE,
        "threshold_kind": "normalized_entropy",
        "temperature": temperature,
        "base_frozen": True,
        "adapter_frozen": str(ADAPTER_DIR),
        "observed_states": [format(state, "06b") for state in OBSERVED_STATES],
        "valid": valid_score,
        "test": test_score,
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
