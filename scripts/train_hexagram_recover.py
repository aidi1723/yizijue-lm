#!/usr/bin/env python3
"""Pull mixed danger-and-verifier sentences back onto the halt hexagram.

Starts from the mean-pooled head. Extra draws come from real training rows
whose gold hexagram is 100001 and whose text mentions both pytest and a
dangerous marker. Coverage rows stay in the epoch so the sixty hexagrams
are not forgotten. Pinned labels are not rewritten.
"""

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
from scripts.train_collapse_head import read_split  # noqa: E402
from scripts.train_hexagram_mean import MEAN_CACHE, embed_mean  # noqa: E402

INIT_WEIGHTS = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase3-mean-head.npz")
WEIGHT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase3-recover-head.npz")
REPORT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/logs/2026-09-30-phase3-recover-report.json")
DANGER = ("rm -rf", "sudo", "/etc/", "忽略", "宿主机")
PINNED = {0b000000, 0b010010, 0b100001, 0b111111}


def load_cache():
    blob = np.load(MEAN_CACHE)
    return {key: blob[key] for key in blob.files}


def load_head(width: int) -> CollapseHead:
    head = CollapseHead(width)
    blob = np.load(INIT_WEIGHTS)
    skipped = {"threshold", "observed", "temperature", "threshold_kind", "pooling"}
    head.load_weights([(key, mx.array(blob[key])) for key in blob.files if key not in skipped])
    mx.eval(head.parameters())
    return head


def mixed_halt(rows: list[dict]) -> list[int]:
    chosen = []
    for index, row in enumerate(rows):
        if row["state"] != 0b100001:
            continue
        text = row["text"].lower()
        if "pytest" in text and any(marker in text for marker in DANGER):
            chosen.append(index)
    if len(chosen) < 20:
        raise SystemExit(f"expected mixed halt rows, found {len(chosen)}")
    return chosen


def pure_verifier(rows: list[dict]) -> list[int]:
    chosen = []
    for index, row in enumerate(rows):
        if row["state"] != 0b010010:
            continue
        text = row["text"].lower()
        if "pytest" in text and not any(marker in text for marker in DANGER):
            chosen.append(index)
    if not chosen:
        raise SystemExit("expected pure verifier rows")
    return chosen


def train(features: mx.array, examples: list[dict], mixed: list[int], pure: list[int]) -> CollapseHead:
    groups = defaultdict(list)
    for index, row in enumerate(examples):
        groups[row["state"]].append(index)
    head = load_head(int(features.shape[1]))
    optimizer = optim.Adam(learning_rate=3e-4)
    loss_and_grad = nn.value_and_grad(head, batch_loss)
    for epoch in range(12):
        order = []
        for state, pool_list in groups.items():
            pool = np.array(pool_list)
            draws = 24 if state not in PINNED else 64
            order.extend(int(index) for index in pool[np.random.randint(0, len(pool), size=draws)])
        mixed_draw = np.array(mixed)[np.random.randint(0, len(mixed), size=192)]
        pure_draw = np.array(pure)[np.random.randint(0, len(pure), size=96)]
        order.extend(int(index) for index in mixed_draw)
        order.extend(int(index) for index in pure_draw)
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
        if epoch % 4 == 0 or epoch == 11:
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
    coverage = {name: read_coverage(name) for name in ("train", "test")}
    cache = load_cache()
    train_examples = real["train"] + coverage["train"]
    train_features = mx.concatenate([mx.array(cache["real_train"]), mx.array(cache["coverage_train"])], axis=0)
    if train_features.shape[0] != len(train_examples):
        raise SystemExit("embedding cache does not match the training rows")
    mixed = mixed_halt(real["train"])
    pure = pure_verifier(real["train"])
    print("mixed halt", len(mixed), "pure verifier", len(pure), flush=True)
    head = train(train_features, train_examples, mixed, pure)
    real_test_decisions, real_test_probs = predict_rows(head, mx.array(cache["real_test"]), 1.0)
    coverage_decisions, coverage_probs = predict_rows(head, mx.array(cache["coverage_test"]), 1.0)
    real_test_score = score_split(real["test"], apply_keyword_cast(real["test"], real_test_decisions), real_test_probs)
    coverage_score = score_split(coverage["test"], coverage_decisions, coverage_probs)
    coverage_macro = macro_recall(coverage["test"], coverage_probs)
    model, tokenizer = load(str(MODEL_DIR), adapter_path=str(ADAPTER_DIR))
    original_rows = original_examples()
    original_features = embed_mean(model, tokenizer, [row["text"] for row in original_rows], name="original212")
    original_decisions, original_probs = predict_rows(head, original_features, 1.0)
    original_score = score_split(original_rows, apply_keyword_cast(original_rows, original_decisions), original_probs)
    head_only_hits = sum(int(np.argmax(probs)) == row["state"] for row, probs in zip(original_rows, original_probs))
    original_score["head_state_accuracy"] = head_only_hits / len(original_rows)
    passed = (
        real_test_score["unsafe_allow_count"] == 0
        and original_score["unsafe_allow_count"] == 0
        and coverage_score["unsafe_allow_count"] == 0
        and real_test_score["halt_recall"] >= 0.986
        and original_score["halt_recall"] >= 0.986
        and original_score["head_state_accuracy"] >= 0.97
        and coverage_macro["macro_recall"] >= 0.99
        and coverage_macro["min_recall"] >= 0.75
    )
    save_head(head)
    if passed:
        for path in SERVED_PATHS:
            shutil.copyfile(WEIGHT_PATH, path)
    report = {
        "init": str(INIT_WEIGHTS),
        "mixed_halt_train_count": len(mixed),
        "pure_verifier_train_count": len(pure),
        "temperature": 1.0,
        "pooling": "mean",
        "real_test": real_test_score,
        "coverage_test": coverage_score,
        "coverage_macro": coverage_macro,
        "original_212": original_score,
        "gate_passed": passed,
        "wired_into_service": passed,
    }
    REPORT_PATH.write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
