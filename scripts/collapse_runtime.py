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


def load_collapse_head(width: int = 1024) -> tuple[CollapseHead, float]:
    blob = np.load(WEIGHT_PATH)
    head = CollapseHead(width)
    weights = [
        (key, mx.array(blob[key]))
        for key in blob.files
        if key not in {"threshold", "observed"}
    ]
    head.load_weights(weights)
    mx.eval(head.parameters())
    return head, float(blob["threshold"])


def collapse_text(model, tokenizer, head: CollapseHead, threshold: float, text: str) -> dict[str, object]:
    token_ids = tokenizer.encode(text)
    if not token_ids:
        token_ids = tokenizer.encode(" ")
    hidden = model.model(mx.array(token_ids)[None])
    mx.eval(hidden)
    last = hidden[:, -1, :]
    return collapse_decision(
        _probs(head.status(last)),
        _probs(head.intent(last)),
        _probs(head.path(last)),
        _probs(head.sandbox(last)),
        _probs(head.evidence(last)),
        threshold=threshold,
    )


def _probs(logits: mx.array) -> list[float]:
    mx.eval(logits)
    return np.array(mx.softmax(logits, axis=-1)[0]).tolist()
