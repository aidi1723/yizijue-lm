# YiZiJue-LM

YiZiJue-LM is a local Agent intent model built for OneCode, not a human-facing chatbot. It converts natural-language requests into verifiable, auditable, fail-closed action JSON for the OneCode control plane.

This project is built on top of Qwen3-0.6B.

## Positioning

YiZiJue-LM is a model component for OneCode Agent applications. It is designed to serve an Agent runtime/control plane rather than directly serve human chat users.

The model proposes controlled OneCode actions such as:

```text
ALLOW_ATOMIC_WRITE
ALLOW_PATCH_WITH_SHA
RUN_VERIFIER_IN_SANDBOX
DENY_AND_LEDGER
SOVEREIGNTY_HALT
```

OneCode remains responsible for final authorization, execution, evidence recording, and fail-closed enforcement.

## Our Contribution

The training data, safety policy, LoRA fine-tuning, evaluation gates, runtime guard, local service wrapper, and OneCode execution contract are designed and engineered by us.

Qwen3-0.6B provides the base semantic capability. YiZiJue-LM provides the Agent-specific intent layer, action schema, safety boundary, and proposal behavior.

## What This Is Not

YiZiJue-LM is not:

- a general chatbot;
- a standalone execution authority;
- a from-scratch foundation model;
- a replacement for OneCode policy enforcement.

## Release Artifact

The v0.1 release provides a LoRA adapter package:

```text
2026-06-04-yizijue-qwen06b-v5-final.tar.gz
```

The package contains:

- final v5 LoRA adapter;
- guarded final evaluation reports;
- local service code;
- evaluation scripts and tests;
- training and service documentation.

It intentionally excludes the full Qwen3-0.6B base weights, local Hugging Face cache, and private/raw training data.

## Local Service

After extracting the release package into the project workspace, run:

```bash
HF_HOME=/path/to/hf-home \
.venv-mlx/bin/python scripts/serve_yizijue_mlx.py \
  --model Qwen/Qwen3-0.6b \
  --adapter-path /path/to/yizijue-qwen06b-strict-hard-negative-recovery-v5-lora \
  --host 127.0.0.1 \
  --port 8090 \
  --max-tokens 220 \
  --preload
```

Open:

```text
http://127.0.0.1:8090/
```

API endpoints:

```text
GET  /health
POST /predict
```

Example:

```bash
curl -sS http://127.0.0.1:8090/predict \
  -H 'Content-Type: application/json' \
  -d '{"input":"运行 pytest 验证一下","max_tokens":220}'
```

Expected action:

```text
RUN_VERIFIER_IN_SANDBOX
```

## Validation

Current local verification:

```text
Ran 39 tests OK
```

Final guarded validation reports are included in the release artifact under `logs/`.

## Attribution

Base model:

```text
Qwen/Qwen3-0.6B
License: Apache-2.0
```

See `NOTICE` and `MODEL_CARD.md` for model boundary and attribution details.
