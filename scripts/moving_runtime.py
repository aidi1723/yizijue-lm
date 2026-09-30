"""Read a six-line cast without letting the changed hexagram authorize an action."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from onecode.experimental.moving_cast import LINE_VALUES, cast_from_lines

MOTION_INDEXES = (14, 21, 28, 35, 42, 49)
WEIGHT_PATH = Path("/Volumes/MacSSD/模型训练/yizijue-qwen06b/models/yizijue-phase4-moving-head.npz")


class MovingHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.lines = nn.Linear(width, 4)


def load_moving_head(width: int = 1024) -> MovingHead | None:
    if not WEIGHT_PATH.is_file():
        return None
    blob = np.load(WEIGHT_PATH)
    head = MovingHead(width)
    head.load_weights([(key, mx.array(blob[key])) for key in ("lines.weight", "lines.bias") if key in blob.files])
    mx.eval(head.parameters())
    return head


def apply_moving_cast(response: dict, cast: dict) -> dict:
    """Record the kernel cast. The execution action stays whatever it already was."""
    updated = dict(response)
    parsed = dict(updated.get("json") or {})
    parsed["moving_cast"] = {
        "values": list(cast["values"]),
        "before": format(int(cast["before"]), "06b"),
        "after": format(int(cast["after"]), "06b"),
        "moving": list(cast["moving"]),
    }
    updated["json"] = parsed
    return updated


def read_moving_cast(model, tokenizer, head: MovingHead, text: str) -> dict | None:
    from onecode.experimental.moving_cast import motion_token_indexes

    token_ids = tokenizer.encode(text)
    pieces = [tokenizer.decode([token_id]) for token_id in token_ids]
    indexes = motion_token_indexes(pieces)
    if indexes is None:
        return None
    hidden = model.model(mx.array(token_ids)[None])
    mx.eval(hidden)
    chosen = []
    for index in indexes:
        vector = hidden[0, index].astype(mx.float32)
        mx.eval(vector)
        chosen.append(np.array(vector))
    logits = head.lines(mx.array(np.stack(chosen)))
    mx.eval(logits)
    values = [LINE_VALUES[int(index)] for index in np.array(mx.argmax(logits, axis=-1))]
    cast = cast_from_lines(values)
    return {
        "values": values,
        "before": cast["before"],
        "after": cast["after"],
        "moving": cast["moving"],
    }
