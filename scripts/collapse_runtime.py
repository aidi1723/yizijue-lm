"""Frozen phase-2 head. Allow predictions are not authorizations."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from onecode.kernel.collapse_decision import collapse_decision

_REPO_HEAD = Path(__file__).resolve().parents[1] / "models" / "yizijue-phase2-collapse-head.npz"
_STORED_HEAD = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase2-collapse-head.npz")
WEIGHT_PATH = _REPO_HEAD if _REPO_HEAD.is_file() else _STORED_HEAD


class CollapseHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.status = nn.Linear(width, 64)
        self.intent = nn.Linear(width, 5)
        self.path = nn.Linear(width, 3)
        self.sandbox = nn.Linear(width, 3)
        self.evidence = nn.Linear(width, 3)


def load_collapse_head(width: int = 1024) -> tuple[CollapseHead, float, float, str, str]:
    blob = np.load(WEIGHT_PATH)
    head = CollapseHead(width)
    skipped = {"threshold", "observed", "temperature", "threshold_kind", "pooling"}
    weights = [(key, mx.array(blob[key])) for key in blob.files if key not in skipped]
    head.load_weights(weights)
    mx.eval(head.parameters())
    temperature = float(blob["temperature"]) if "temperature" in blob.files else 1.0
    kind = str(blob["threshold_kind"]) if "threshold_kind" in blob.files else "confidence"
    pooling = str(blob["pooling"]) if "pooling" in blob.files else "last"
    return head, float(blob["threshold"]), temperature, kind, pooling


def _pooled(hidden: mx.array, pooling: str) -> mx.array:
    if pooling == "mean":
        return hidden.mean(axis=1)
    if pooling != "last":
        raise ValueError(f"unknown pooling: {pooling}")
    return hidden[:, -1, :]


def collapse_text(model, tokenizer, head: CollapseHead, threshold: float, text: str, *, temperature: float = 1.0, threshold_kind: str = "confidence", pooling: str = "last") -> dict[str, object]:
    token_ids = tokenizer.encode(text)
    if not token_ids:
        token_ids = tokenizer.encode(" ")
    hidden = model.model(mx.array(token_ids)[None])
    mx.eval(hidden)
    pooled = _pooled(hidden, pooling)
    entropy_gate = threshold if threshold_kind == "normalized_entropy" else None
    return collapse_decision(
        _probs(head.status(pooled) / temperature),
        _probs(head.intent(pooled)),
        _probs(head.path(pooled)),
        _probs(head.sandbox(pooled)),
        _probs(head.evidence(pooled)),
        threshold=threshold,
        entropy_gate=entropy_gate,
    )


def _probs(logits: mx.array) -> list[float]:
    mx.eval(logits)
    return np.array(mx.softmax(logits, axis=-1)[0]).tolist()
