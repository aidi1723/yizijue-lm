#!/usr/bin/env python3
"""Train six Da Yan line heads. The changed hexagram comes from the kernel."""

import json
import sys
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

from onecode.experimental.moving_cast import LINE_VALUES, cast_from_lines
from onecode.kernel.hexagram import IchingKernel
from scripts.train_collapse_head import ADAPTER_DIR, MODEL_DIR

DATA_DIR = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/hexagram_moving_v1")
WEIGHT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase4-moving-head.npz")
CACHE_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase4-moving-embeddings.npz")
REPORT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/logs/2026-09-30-phase4-moving-report.json")
DAYAN = {6: 3 / 16, 7: 5 / 16, 8: 5 / 16, 9: 3 / 16}


class MovingHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.lines = nn.Linear(width, 4)


def read_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


MOTION_INDEXES = (14, 21, 28, 35, 42, 49)


def embed_lines(model, tokenizer, rows: list[dict], name: str) -> mx.array:
    vectors = []
    for index, row in enumerate(rows):
        token_ids = tokenizer.encode(row["text"])
        if len(token_ids) <= MOTION_INDEXES[-1]:
            raise SystemExit(f"{row['id']} is shorter than the line template")
        hidden = model.model(mx.array(token_ids)[None])
        mx.eval(hidden)
        chosen = []
        for token_index in MOTION_INDEXES:
            piece = tokenizer.decode([token_ids[token_index]])
            if piece not in {"动", "静"}:
                raise SystemExit(f"{row['id']} token {token_index} is {piece!r}")
            pooled = hidden[0, token_index].astype(mx.float32)
            mx.eval(pooled)
            chosen.append(np.array(pooled))
        vectors.append(np.stack(chosen))
        if index % 200 == 0:
            print(f"embedded {name} {index}/{len(rows)}", flush=True)
    print(f"embedded {name} {len(rows)}/{len(rows)}", flush=True)
    return mx.array(np.stack(vectors))


def labels_for(rows: list[dict]) -> mx.array:
    table = {value: index for index, value in enumerate(LINE_VALUES)}
    return mx.array([[table[value] for value in row["values"]] for row in rows])


def loss_fn(head: MovingHead, features: mx.array, labels: mx.array) -> mx.array:
    logits = head.lines(features)
    return nn.losses.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1), reduction="mean")


def predict(head: MovingHead, features: mx.array) -> np.ndarray:
    logits = head.lines(features)
    mx.eval(logits)
    return np.array(mx.argmax(logits, axis=-1))


def score(rows: list[dict], predicted: np.ndarray) -> dict:
    line_hits = 0
    cast_hits = 0
    before_hits = 0
    after_hits = 0
    weighted_hit = 0.0
    weighted_total = 0.0
    outside = 0
    for row, pred_index in zip(rows, predicted):
        pred_values = [LINE_VALUES[int(index)] for index in pred_index]
        gold_values = row["values"]
        line_hits += sum(left == right for left, right in zip(pred_values, gold_values))
        cast_ok = pred_values == gold_values
        cast_hits += cast_ok
        pred_cast = cast_from_lines(pred_values)
        gold_cast = cast_from_lines(gold_values)
        before_hits += pred_cast["before"] == gold_cast["before"]
        after_hits += pred_cast["after"] == gold_cast["after"]
        if not 0 <= pred_cast["after"] <= 63:
            outside += 1
        if pred_cast["after"] != IchingKernel.mutate_lines(pred_cast["before"], pred_cast["moving"]):
            raise SystemExit("prediction left the kernel mutation")
        weight = 1.0
        for value in gold_values:
            weight *= DAYAN[value]
        weighted_total += weight
        weighted_hit += weight * cast_ok
    lines = len(rows) * 6
    return {
        "count": len(rows),
        "line_accuracy": line_hits / lines,
        "cast_accuracy": cast_hits / len(rows),
        "before_accuracy": before_hits / len(rows),
        "after_accuracy": after_hits / len(rows),
        "dayan_weighted_cast_accuracy": weighted_hit / weighted_total,
        "outside_sixty_four": outside,
    }


def train(features: mx.array, labels: mx.array) -> MovingHead:
    head = MovingHead(int(features.shape[-1]))
    optimizer = optim.Adam(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(head, loss_fn)
    count = features.shape[0]
    for epoch in range(25):
        order = np.random.permutation(count)
        total = 0.0
        steps = 0
        for start in range(0, count, 128):
            chosen = order[start : start + 128].tolist()
            loss, grads = loss_and_grad(head, features[chosen], labels[chosen])
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state, loss)
            total += float(loss)
            steps += 1
        if epoch % 5 == 0 or epoch == 24:
            print(f"epoch {epoch} loss {total / steps:.4f}", flush=True)
    return head


def save_head(head: MovingHead) -> None:
    flat = {}

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}{key}.", child)
            return
        flat[prefix[:-1]] = np.array(value)

    walk("", dict(head.parameters()))
    np.savez(WEIGHT_PATH, line_values=np.array(LINE_VALUES), **flat)


def main() -> int:
    splits = {name: read_split(name) for name in ("train", "valid", "test")}
    model, tokenizer = load(str(MODEL_DIR), adapter_path=str(ADAPTER_DIR))
    cache = {}
    if CACHE_PATH.is_file():
        blob = np.load(CACHE_PATH)
        cache = {key: blob[key] for key in blob.files}
    features = {}
    for name, rows in splits.items():
        if name in cache and len(cache[name]) == len(rows) and cache[name].ndim == 3:
            features[name] = mx.array(cache[name])
        else:
            features[name] = embed_lines(model, tokenizer, rows, name)
            cache[name] = np.array(features[name])
    np.savez(CACHE_PATH, **cache)
    head = train(features["train"], labels_for(splits["train"]))
    valid_score = score(splits["valid"], predict(head, features["valid"]))
    test_score = score(splits["test"], predict(head, features["test"]))
    passed = (
        test_score["line_accuracy"] >= 0.95
        and test_score["cast_accuracy"] >= 0.9
        and test_score["after_accuracy"] >= 0.9
        and test_score["outside_sixty_four"] == 0
    )
    save_head(head)
    report = {
        "line_values": list(LINE_VALUES),
        "dayan": DAYAN,
        "state_space": 64,
        "served_action_head_unchanged": True,
        "valid": valid_score,
        "test": test_score,
        "gate_passed": passed,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
