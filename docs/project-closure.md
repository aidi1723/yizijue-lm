# YiZiJue-LM v0.1 Project Closure

Date: 2026-06-04

## Status

YiZiJue-LM v0.1 is closed as a local Agent proposal-model release for OneCode.

It is not a human-facing chatbot and not a standalone execution authority.

## Public Release

Repository:

```text
https://github.com/aidi1723/yizijue-lm
```

Release:

```text
https://github.com/aidi1723/yizijue-lm/releases/tag/v0.1
```

Artifact:

```text
2026-06-04-yizijue-qwen06b-v5-final.tar.gz
```

SHA256:

```text
a0e61d46b1645f4581b0aa00c68be1417044f326b9feeb7d693896f135e74ce9
```

## Positioning

YiZiJue-LM is a local Agent intent model built for OneCode. It converts
natural-language requests into verifiable, auditable, fail-closed action JSON
for the OneCode control plane.

This project is built on top of Qwen3-0.6B.

## Final Metrics

Recovery v5 self gate:

```text
sample_count: 300
json_valid_rate: 0.9733333333333334
action_match_rate: 0.8933333333333333
unknown_action_count: 0
unsafe_allow_count: 0
```

Original full-test cross-check:

```text
sample_count: 212
json_valid_rate: 1.0
action_match_rate: 0.8113207547169812
unknown_action_count: 0
unsafe_allow_count: 0
```

## Release Contents

Included:

- final v5 LoRA adapter;
- guarded final evaluation reports;
- local service code;
- evaluation scripts and tests;
- README, model card, notice, license, release notes.

Excluded:

- Qwen base weights;
- local Hugging Face cache;
- private/raw training data;
- API keys or private service credentials.

## Boundary

The model proposes action JSON. OneCode must remain responsible for final
validation, authorization, execution, and audit.

Next engineering step:

```text
Port the final fail-closed guard semantics into OneCode's real execution path.
```
