# Changelog

## 2026-09-28

### Added

- `/predict` 现在先走规则，再走冻结的坍缩头，最后才考虑生成。有名字的动作不再进入全量解码。
- `scripts/collapse_runtime.py` 与 `models/yizijue-phase2-collapse-head.npz`。基座和 v5 适配器保持冻结，门槛 0.3。
- 允许类动作会补路径和 SHA256。相对路径写入记录正文摘要。补丁只在工作区文件里搜索块恰好出现一次时记录修改前后摘要，并且不写回文件。
- `scripts/train_collapse_head.py` 与 `scripts/eval_integrated_predict.py`。
- 审计会标出带引号正文的「内容替换为」却被标成 `schema_out_of_contract` 的样本。`balanced-distill-002456` 已从清理后的验证集移除。

### Changed

- 规则短路覆盖旧危险词、hosts 与更强标记同时出现、模糊句、`curl` 管道，以及执行意图加上 `/tmp/`、`/etc/`、`/var/` 或 `/home/`。单纯的 pytest 仍是沙箱验证。
- `serve_yizijue_mlx.py` 增加 `from __future__ import annotations`，Python 3.12 可以导入。
- `/predict` 返回 `prefill_ms`、`decode_ms`、`guard_ms`、`e2e_ms`。

### Verification

- 服务、评测和审计测试 79 个通过。
- 2026-09-28 在 `127.0.0.1:8090` 挂载坍缩头，热身后规则短路 0.09–0.15 ms，坍缩头 26–30 ms，解码为 0。测完服务已停止。
- 头接入前的 9 条生成样本解码为 4.0–4.3 秒。整体复测的 300 条和 212 条在头接入后全部跳过生成，误放行为 0。该复测发生在最后一次放宽相对路径写入解析之前。

### Not included

- 没有重训。清理后的训练集与 v5 训练集逐字节相同。
- 没有把裁决接进 OneCode 的账本和沙箱。
- 没有改写 2026-06-04 发布包。
- 绝对路径、缺失补丁文件和不唯一搜索块仍然拒绝。

## 2026-07-05

### Fixed

- Closed the evaluation gate loophole where reports with missing predictions could pass because rates were computed only over predicted rows.
- Made `scripts/build_distilled_training_set.py --help` and `scripts/distill_openai_compatible.py --help` work without importing the external OneCode package.
- Converted missing OneCode runtime dependency failures into clear argparse errors without Python tracebacks.
- Removed machine-specific default adapter paths from the local MLX service.
- Added request and startup bounds for `max_tokens` (`1..512`) and request body size checks for `/predict`.
- Reduced service generation error responses so they no longer include raw exception details.
- Cleaned the release directory by removing `.git`, `__pycache__`, and `.pyc` files.
- Corrected release checksum records to point at an included guarded evaluation report.

### Added

- `data/train_messages_distilled_clean.jsonl`, excluding 35 suspicious schema/write label conflicts from the recommended training messages set.
- `data/training/train_messages_distilled_label_audit.json`, preserving the label-audit evidence for the excluded rows.
- `scripts/verify.sh`, a single verification command covering tests, syntax checks, evaluation gate positive/negative checks, cache cleanup, and release checksums. Release tests are run from inside the release directory so imports exercise the release script copy.
- `.gitignore`, `pyproject.toml`, and `requirements.txt` for project metadata and reproducibility.
- HTTP handler tests for oversized request bodies and out-of-range `max_tokens`; these call the handler directly and do not bind local ports.
- CLI regression tests for missing OneCode dependency behavior.
- Release-local `scripts/verify_release.sh` so the public release directory can validate itself without the parent workspace.
- Documentation tests requiring the workspace and release READMEs to expose their verification commands.
- Documentation/script consistency tests requiring the workspace and release README verification scopes to match their verification scripts, without platform-specific checksum command references.
- Verification-script tests requiring `scripts/check_release_checksums.py` and forbidding platform-specific checksum commands.
- `scripts/check_release_checksums.py` plus a release-local copy, centralizing SHA-256 release checksum validation instead of duplicating inline shell snippets.
- Checksum validation now rejects empty checksum files, malformed checksum rows, and paths that escape the release root.
- Local service `/predict` now rejects missing or negative `Content-Length` before reading request bodies or invoking generation.
- Prediction generation CLI now rejects non-positive `--limit` and `--max-tokens` values during argument parsing.
- Evaluation now rejects duplicate prediction IDs instead of silently letting later rows overwrite earlier predictions.
- Hardened dataset building now rejects invalid split ratios and replay multipliers before writing train/valid/test files.
- Evaluation now rejects duplicate gold IDs before computing metrics, preventing repeated counting of one prediction.
- Hardened dataset building now rejects duplicate source IDs before splitting or hard-negative indexing.
- Hardened dataset building now rejects missing or empty source IDs before grouping rows.
- Evaluation now rejects missing or empty gold and prediction IDs with explicit errors.
- Evaluation now reports unexpected prediction IDs and fails the gate when predictions contain IDs outside the gold set.
- Evaluation now rejects prediction rows that omit the `prediction` field instead of treating them as empty model output.
- Evaluation JSONL readers now reject non-object rows with line-numbered errors.
- Evaluation JSONL readers now wrap malformed JSON lines with file and line-numbered `ValueError` messages.
- Evaluation now rejects non-string `prediction` fields instead of coercing structured values into text.
- Evaluation now rejects non-string gold and prediction IDs instead of coercing them with `str()`.
- Evaluation CLI now rejects acceptance-rate thresholds outside the 0..1 interval.
- Local service `/predict` now rejects JSON request bodies that are not objects before accessing fields.
- Local service `/predict` now rejects non-canonical `Content-Length` values instead of relying on permissive integer parsing.
- Local service `/predict` now rejects boolean `max_tokens` values instead of treating them as integers.
- Local service CLI now rejects ports outside `0..65535` during argument parsing.
- Release checksum validation now requires every release file to be listed in the manifest, excluding the checksum file itself.
- Release checksum validation now rejects duplicate manifest paths.
- Release checksum validation now rejects manifest paths that resolve to directories.
- Release checksum validation now rejects non-canonical relative paths such as `./README.md`.
- Release checksum validation now rejects non-lowercase SHA-256 digests in the manifest.
- Release checksum validation now rejects backslash-separated manifest paths.
- Release checksum validation now rejects control characters in manifest paths.
- Release checksum validation now rejects manifest lines with leading spaces and requires canonical `digest  path` formatting.
- Release checksum validation now rejects manifests that list `release/checksums.txt` itself.
- Release checksum validation now rejects symlinks anywhere in the release tree, including unlisted dangling links.
- Evaluation now rejects empty gold row sets before computing metrics.
- Evaluation now rejects gold rows without a parseable assistant action, preventing `None == None` matches from inflating action-match rates.
- Hardened dataset building now rejects non-string source IDs instead of coercing them with `str()`.
- Hardened dataset building now rejects source IDs containing whitespace or control characters.
- Hardened dataset building now rejects empty source row sets before writing split files.
- Hardened dataset building now rejects source rows without a parseable assistant gold action.
- Hardened dataset building now rejects duplicate hard-negative replay IDs before augmentation.
- Hardened dataset building now rejects unknown recovery action names instead of silently producing no recovery upsample.
- Prediction generation JSONL input now rejects non-object rows, missing/empty/non-string IDs, and duplicate IDs before writing prediction files.
- Prediction generation CLI now reports input validation errors before importing MLX runtime dependencies.
- Prediction generation JSONL input now rejects empty input files, malformed `messages` fields, and rows without a user message before building prompts.
- Added `docs/YIZIJUE_LM_RELEASE_CLOSURE_2026-07-05.md` as the source handoff and GitHub publication closure record.
- Added `docs/YIZIJUE_LM_FINAL_CLOSURE_CN_2026-07-05.md` as the Chinese final closure document for local handoff and GitHub update records.
- Added root-level `LOCAL_MAINTENANCE.md` so later local maintenance and development can find the closure state directly from the project directory.
- Added Chinese maintenance docs for runbook, development workflow, release update checklist, data/training maintenance, troubleshooting, and maintenance changelog.
- Release documentation now reflects the current release self-check count and checksum-backed handoff status.

### Verification

- `bash scripts/verify.sh`
- Main workspace tests: 116 tests OK.
- Release package tests: 86 tests OK.
- Correct guarded full-test gate: `sample_count=212`, `missing_prediction_count=0`, `json_valid_rate=0.9858490566037735`, `action_match_rate=0.8254716981132075`, `unsafe_allow_count=0`, `unknown_action_count=0`.
