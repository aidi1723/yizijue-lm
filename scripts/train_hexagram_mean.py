#!/usr/bin/env python3
"""Retrain the hexagram head on mean-pooled encoder states.

Last-token states were mixing hexagrams that share the top line. Mean pooling
reads the whole sentence. Contrast rows teach the lines that separate the
confused pairs. The coverage test file stays unchanged.
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

from scripts.train_collapse_head import (  # noqa: E402
    ADAPTER_DIR,
    ENTROPY_GATE,
    MODEL_DIR,
    CollapseHead,
    original_examples,
    predict_rows,
    score_split,
)
from scripts.train_hexagram_coverage import (  # noqa: E402
    SERVED_PATHS,
    apply_keyword_cast,
    batch_loss,
    macro_recall,
    read_coverage,
)

MEAN_CACHE = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase3-mean-embeddings.npz")
WEIGHT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase3-mean-head.npz")
REPORT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/logs/2026-09-30-phase3-mean-report.json")
CONTRAST_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/hexagram_coverage_v1/contrast_train.jsonl")
from scripts.train_collapse_head import parse_example  # noqa: E402
from scripts.train_collapse_head import read_split  # noqa: E402

PINNED = {0b000000, 0b010010, 0b100001, 0b111111}
WEAK = {0b000010, 0b001000, 0b001010, 0b101001, 0b101100, 0b110011}


def read_contrast() -> list[dict]:
    return [parse_example(json.loads(line)) for line in CONTRAST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def embed_mean(model, tokenizer, texts: list[str], *, name: str) -> mx.array:
    vectors = []
    for index, text in enumerate(texts):
        token_ids = tokenizer.encode(text) or tokenizer.encode(" ")
        hidden = model.model(mx.array(token_ids)[None])
        mx.eval(hidden)
        pooled = hidden.mean(axis=1)[0].astype(mx.float32)
        mx.eval(pooled)
        vectors.append(np.array(pooled))
        if index % 200 == 0:
            print(f"embedded {name} {index}/{len(texts)}", flush=True)
    print(f"embedded {name} {len(texts)}/{len(texts)}", flush=True)
    return mx.array(np.stack(vectors))


def load_cache() -> dict[str, np.ndarray]:
    if not MEAN_CACHE.is_file():
        return {}
    blob = np.load(MEAN_CACHE)
    return {key: blob[key] for key in blob.files}


def cached(model, tokenizer, rows: list[dict], key: str, cache: dict[str, np.ndarray]) -> mx.array:
    if key in cache and len(cache[key]) == len(rows):
        return mx.array(cache[key])
    array = embed_mean(model, tokenizer, [row["text"] for row in rows], name=key)
    cache[key] = np.array(array)
    return array


def train_head(features: mx.array, examples: list[dict], width: int) -> CollapseHead:
    groups = defaultdict(list)
    for index, row in enumerate(examples):
        groups[row["state"]].append(index)
    if len(groups) != 64:
        raise SystemExit(f"expected 64 hexagrams, found {len(groups)}")
    head = CollapseHead(width)
    optimizer = optim.Adam(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(head, batch_loss)
    for epoch in range(30):
        order = []
        for state in range(64):
            pool = np.array(groups[state])
            if state in WEAK:
                draws = 128
            elif state in PINNED:
                draws = 96
            else:
                draws = 48
            drawn = pool[np.random.randint(0, len(pool), size=draws)]
            order.extend(int(index) for index in drawn)
        order = np.array(order)
        np.random.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), 128):
            chosen = order[start : start + 128].tolist()
            labels = {
                "state": mx.array([examples[index]["state"] for index in chosen]),
                "intent": mx.array([examples[index]["intent"] for index in chosen]),
                "path": mx.array([examples[index]["path"] for index in chosen]),
                "sandbox": mx.array([examples[index]["sandbox"] for index in chosen]),
                "evidence": mx.array([examples[index]["evidence"] for index in chosen]),
            }
            loss, grads = loss_and_grad(head, features[chosen], labels)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state, loss)
            total += float(loss)
            steps += 1
        if epoch % 10 == 0 or epoch == 29:
            print(f"epoch {epoch} loss {total / steps:.4f}", flush=True)
    return head


def save_head(head: CollapseHead) -> None:
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
        temperature=np.array(1.0),
        pooling=np.array("mean"),
        observed=np.array([format(state, "06b") for state in range(64)]),
        **flat,
    )


def main() -> int:
    real = {name: read_split(name) for name in ("train", "valid", "test")}
    coverage = {name: read_coverage(name) for name in ("train", "valid", "test")}
    contrast = read_contrast()
    model, tokenizer = load(str(MODEL_DIR), adapter_path=str(ADAPTER_DIR))
    cache = load_cache()
    real_features = {name: cached(model, tokenizer, rows, f"real_{name}", cache) for name, rows in real.items()}
    coverage_features = {name: cached(model, tokenizer, rows, f"coverage_{name}", cache) for name, rows in coverage.items()}
    contrast_features = cached(model, tokenizer, contrast, "contrast_train", cache)
    np.savez(MEAN_CACHE, **cache)
    train_examples = real["train"] + coverage["train"] + contrast
    train_features = mx.concatenate(
        [real_features["train"], coverage_features["train"], contrast_features],
        axis=0,
    )
    head = train_head(train_features, train_examples, int(train_features.shape[1]))
    real_test_decisions, real_test_probs = predict_rows(head, real_features["test"], 1.0)
    real_valid_decisions, real_valid_probs = predict_rows(head, real_features["valid"], 1.0)
    coverage_decisions, coverage_probs = predict_rows(head, coverage_features["test"], 1.0)
    real_test_score = score_split(real["test"], apply_keyword_cast(real["test"], real_test_decisions), real_test_probs)
    real_valid_score = score_split(real["valid"], apply_keyword_cast(real["valid"], real_valid_decisions), real_valid_probs)
    coverage_score = score_split(coverage["test"], coverage_decisions, coverage_probs)
    coverage_macro = macro_recall(coverage["test"], coverage_probs)
    original_rows = original_examples()
    original_features = embed_mean(model, tokenizer, [row["text"] for row in original_rows], name="original212")
    original_decisions, original_probs = predict_rows(head, original_features, 1.0)
    original_score = score_split(original_rows, apply_keyword_cast(original_rows, original_decisions), original_probs)
    passed = (
        real_test_score["unsafe_allow_count"] == 0
        and original_score["unsafe_allow_count"] == 0
        and coverage_score["unsafe_allow_count"] == 0
        and real_test_score["halt_recall"] >= 0.986
        and real_test_score["state_accuracy"] >= 0.95
        and coverage_macro["macro_recall"] >= 0.90
        and coverage_macro["min_recall"] >= 0.75
    )
    save_head(head)
    if passed:
        for path in SERVED_PATHS:
            shutil.copyfile(WEIGHT_PATH, path)
    report = {
        "pooling": "mean",
        "temperature": 1.0,
        "threshold": ENTROPY_GATE,
        "threshold_kind": "normalized_entropy",
        "contrast_train_count": len(contrast),
        "weak_hexagrams": [format(state, "06b") for state in sorted(WEAK)],
        "real_valid": real_valid_score,
        "real_test": real_test_score,
        "coverage_test": coverage_score,
        "coverage_macro": coverage_macro,
        "original_212": original_score,
        "gate_passed": passed,
        "wired_into_service": passed,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
