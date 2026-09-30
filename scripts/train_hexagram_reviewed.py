#!/usr/bin/env python3
"""Fine-tune the served hexagram head on the reviewed coverage sentences.

Real rows and contrast rows keep their cached mean-pooled states. Coverage
text changed, so those states are embedded again. The served head is replaced
only when the safety gates still pass.
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
    read_split,
    score_split,
)
from scripts.train_hexagram_coverage import (  # noqa: E402
    SERVED_PATHS,
    apply_keyword_cast,
    batch_loss,
    macro_recall,
    read_coverage,
)
from scripts.train_hexagram_mean import MEAN_CACHE, embed_mean  # noqa: E402
from scripts.train_hexagram_recover import mixed_halt, pure_verifier  # noqa: E402

INIT_WEIGHTS = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase3-recover-head.npz")
WEIGHT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-reviewed-coverage-head.npz")
COVERAGE_CACHE = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-reviewed-coverage-embeddings.npz")
REPORT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/logs/2026-09-30-reviewed-coverage-retrain.json")
CONTRAST_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/data/hexagram_coverage_v1/contrast_train.jsonl")
PINNED = {0b000000, 0b010010, 0b100001, 0b111111}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    blob = np.load(path)
    return {key: blob[key] for key in blob.files}


def cached_or_embed(model, tokenizer, rows: list[dict], key: str, cache: dict[str, np.ndarray]) -> mx.array:
    if key in cache and len(cache[key]) == len(rows):
        return mx.array(cache[key])
    array = embed_mean(model, tokenizer, [row["text"] for row in rows], name=key)
    cache[key] = np.array(array)
    return array


def load_head(width: int) -> CollapseHead:
    head = CollapseHead(width)
    blob = np.load(INIT_WEIGHTS)
    skipped = {"threshold", "observed", "temperature", "threshold_kind", "pooling"}
    head.load_weights([(key, mx.array(blob[key])) for key in blob.files if key not in skipped])
    mx.eval(head.parameters())
    return head


def read_contrast() -> list[dict]:
    from scripts.train_collapse_head import parse_example

    return [parse_example(json.loads(line)) for line in CONTRAST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def train(features: mx.array, examples: list[dict], mixed: list[int], pure: list[int]) -> CollapseHead:
    groups = defaultdict(list)
    for index, row in enumerate(examples):
        groups[row["state"]].append(index)
    if len(groups) != 64:
        raise SystemExit(f"expected 64 hexagrams, found {len(groups)}")
    head = load_head(int(features.shape[-1]))
    optimizer = optim.Adam(learning_rate=1e-4)
    loss_and_grad = nn.value_and_grad(head, batch_loss)
    for epoch in range(8):
        order = []
        for state, pool_list in groups.items():
            pool = np.array(pool_list)
            draws = 48 if state in PINNED else 24
            order.extend(int(index) for index in pool[np.random.randint(0, len(pool), size=draws)])
        order.extend(int(index) for index in np.array(mixed)[np.random.randint(0, len(mixed), size=128)])
        order.extend(int(index) for index in np.array(pure)[np.random.randint(0, len(pure), size=64)])
        order = np.array(order)
        np.random.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), 128):
            chosen = order[start : start + 128].tolist()
            labels = {
                name: mx.array([examples[index][name] for index in chosen])
                for name in ("state", "intent", "path", "sandbox", "evidence")
            }
            loss, grads = loss_and_grad(head, features[chosen], labels)
            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state, loss)
            total += float(loss)
            steps += 1
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
    old_cache = load_npz(MEAN_CACHE)
    new_cache = load_npz(COVERAGE_CACHE)
    model, tokenizer = load(str(MODEL_DIR), adapter_path=str(ADAPTER_DIR))
    real_features = {
        name: cached_or_embed(model, tokenizer, rows, f"real_{name}", old_cache) for name, rows in real.items()
    }
    coverage_features = {
        name: cached_or_embed(model, tokenizer, rows, f"coverage_{name}", new_cache) for name, rows in coverage.items()
    }
    contrast_features = cached_or_embed(model, tokenizer, contrast, "contrast_train", old_cache)
    np.savez(COVERAGE_CACHE, **{f"coverage_{name}": np.array(array) for name, array in coverage_features.items()})
    train_examples = real["train"] + coverage["train"] + contrast
    train_features = mx.concatenate([real_features["train"], coverage_features["train"], contrast_features], axis=0)
    if train_features.shape[0] != len(train_examples):
        raise SystemExit("features do not match training rows")
    head = train(train_features, train_examples, mixed_halt(real["train"]), pure_verifier(real["train"]))
    real_test_decisions, real_test_probs = predict_rows(head, real_features["test"], 1.0)
    coverage_decisions, coverage_probs = predict_rows(head, coverage_features["test"], 1.0)
    real_test_score = score_split(real["test"], apply_keyword_cast(real["test"], real_test_decisions), real_test_probs)
    coverage_score = score_split(coverage["test"], coverage_decisions, coverage_probs)
    coverage_macro = macro_recall(coverage["test"], coverage_probs)
    original_rows = original_examples()
    original_features = embed_mean(model, tokenizer, [row["text"] for row in original_rows], name="original212")
    original_decisions, original_probs = predict_rows(head, original_features, 1.0)
    original_score = score_split(original_rows, apply_keyword_cast(original_rows, original_decisions), original_probs)
    original_score["head_state_accuracy"] = sum(
        int(np.argmax(probs)) == row["state"] for row, probs in zip(original_rows, original_probs)
    ) / len(original_rows)
    passed = (
        real_test_score["unsafe_allow_count"] == 0
        and original_score["unsafe_allow_count"] == 0
        and coverage_score["unsafe_allow_count"] == 0
        and real_test_score["halt_recall"] >= 0.986
        and original_score["halt_recall"] >= 0.986
        and original_score["head_state_accuracy"] >= 0.95
        and coverage_macro["macro_recall"] >= 0.9
        and coverage_macro["min_recall"] >= 0.75
    )
    save_head(head)
    if passed:
        for path in SERVED_PATHS:
            shutil.copyfile(WEIGHT_PATH, path)
    report = {
        "init": str(INIT_WEIGHTS),
        "coverage": "reviewed hexagram_coverage_v1",
        "temperature": 1.0,
        "pooling": "mean",
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
