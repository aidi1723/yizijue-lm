#!/usr/bin/env python3
"""Train the moving-line head on varied wording. The test phrasing is unseen."""

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

from onecode.experimental.moving_cast import motion_token_indexes
from scripts.train_collapse_head import ADAPTER_DIR, MODEL_DIR
from scripts.train_moving_head import MovingHead, labels_for, loss_fn, predict, score

DATA_DIR = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/hexagram_moving_v2")
WEIGHT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase4-paraphrase-head.npz")
CACHE_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase4-paraphrase-embeddings.npz")
REPORT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/logs/2026-09-30-phase4-paraphrase-report.json")
SERVED_MOVING = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase4-moving-head.npz")


def read_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def embed_lines(model, tokenizer, rows: list[dict], name: str) -> mx.array:
    vectors = []
    for index, row in enumerate(rows):
        token_ids = tokenizer.encode(row["text"])
        pieces = [tokenizer.decode([token_id]) for token_id in token_ids]
        indexes = motion_token_indexes(pieces)
        if indexes is None:
            raise SystemExit(f"{row['id']} has no six line marks")
        hidden = model.model(mx.array(token_ids)[None])
        mx.eval(hidden)
        chosen = []
        for token_index in indexes:
            vector = hidden[0, token_index].astype(mx.float32)
            mx.eval(vector)
            chosen.append(np.array(vector))
        vectors.append(np.stack(chosen))
        if index % 400 == 0:
            print(f"embedded {name} {index}/{len(rows)}", flush=True)
    print(f"embedded {name} {len(rows)}/{len(rows)}", flush=True)
    return mx.array(np.stack(vectors))


def train(features: mx.array, labels: mx.array) -> MovingHead:
    head = MovingHead(int(features.shape[-1]))
    optimizer = optim.Adam(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(head, loss_fn)
    count = features.shape[0]
    for epoch in range(12):
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
        if epoch % 4 == 0 or epoch == 11:
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
    np.savez(WEIGHT_PATH, **flat)


def main() -> int:
    splits = {name: read_split(name) for name in ("train", "valid", "test")}
    if any(row["pattern"] == "heldout" for row in splits["train"]):
        raise SystemExit("held-out wording leaked into training")
    model, tokenizer = load(str(MODEL_DIR), adapter_path=str(ADAPTER_DIR))
    cache = {}
    if CACHE_PATH.is_file():
        blob = np.load(CACHE_PATH)
        cache = {key: blob[key] for key in blob.files}
    features = {}
    for name, rows in splits.items():
        if name in cache and len(cache[name]) == len(rows) and getattr(cache[name], "ndim", 0) == 3:
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
    if passed:
        import shutil

        shutil.copyfile(WEIGHT_PATH, SERVED_MOVING)
    report = {
        "train_patterns": ["tight", "shi", "wei"],
        "test_pattern": "heldout",
        "test": test_score,
        "valid": valid_score,
        "gate_passed": passed,
        "wired_into_moving_head": passed,
        "action_head_unchanged": True,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
