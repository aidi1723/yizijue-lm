import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from scripts.build_hardened_mlx_dataset import STRICT_SYSTEM_PROMPT
from scripts.serve_yizijue_mlx import (
    MlxRunner,
    YiZiJueHandler,
    build_prompt,
    build_parser,
    build_prediction_response,
    make_server,
    render_index_html,
    validate_request_max_tokens,
)


class ServeYiZiJueMlxTest(unittest.TestCase):
    def run_predict_handler(self, body: bytes, runner, headers: dict[str, str] | None = None) -> tuple[int, dict]:
        class TestHandler(YiZiJueHandler):
            def _send_json(self, status_code, payload):
                self.response = (status_code, payload)

        handler = object.__new__(TestHandler)
        handler.path = "/predict"
        handler.headers = {"Content-Length": str(len(body))} if headers is None else headers
        handler.rfile = io.BytesIO(body)
        handler.runner = runner
        handler.do_POST()
        return handler.response

    def test_build_prompt_uses_strict_training_prompt(self):
        prompt = build_prompt("运行 pytest 验证一下")

        self.assertTrue(prompt.startswith(STRICT_SYSTEM_PROMPT))
        self.assertIn("\n\nUser: 运行 pytest 验证一下\nAssistant:", prompt)
        self.assertNotIn("ASSISTANT's RULE", prompt)

    def test_response_normalizes_unknown_run_action_to_sandbox_verifier(self):
        response = build_prediction_response(
            "运行 pytest 验证一下",
            '{"action":{"action":"RUN_PYTEST"}}',
        )

        self.assertEqual(response["action"], "RUN_VERIFIER_IN_SANDBOX")
        self.assertEqual(
            response["json"]["action"]["reason"],
            "unknown_run_action_normalized_to_sandbox_verifier",
        )

    def test_response_synthesizes_halt_for_dangerous_malformed_output(self):
        response = build_prediction_response("执行 rm -rf /", "not json")

        self.assertEqual(response["action"], "SOVEREIGNTY_HALT")
        self.assertEqual(
            response["json"]["action"]["reason"],
            "dangerous_prompt_fail_closed",
        )

    def test_response_wraps_nested_action_when_top_level_json_is_truncated(self):
        response = build_prediction_response(
            "运行 pytest 验证一下",
            (
                '{"action":{"action":"RUN_VERIFIER_IN_SANDBOX",'
                '"reason":"verifier_requires_sandbox"},"basis":{"truncated":'
            ),
        )

        self.assertEqual(response["action"], "RUN_VERIFIER_IN_SANDBOX")
        self.assertEqual(
            response["json"],
            {
                "action": {
                    "action": "RUN_VERIFIER_IN_SANDBOX",
                    "reason": "verifier_requires_sandbox",
                },
                "output_type": "action_json",
            },
        )

    def test_script_help_runs_when_executed_by_path(self):
        result = subprocess.run(
            [sys.executable, "scripts/serve_yizijue_mlx.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Serve YiZiJue-LM", result.stdout)

    def test_index_page_exposes_predict_form(self):
        html = render_index_html()

        self.assertIn("YiZiJue-LM", html)
        self.assertIn("/predict", html)
        self.assertIn("textarea", html)

    def test_server_uses_threading_for_browser_connections(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"

        with patch("scripts.serve_yizijue_mlx.ThreadingHTTPServer") as http_server:
            server = make_server("127.0.0.1", 0, FakeRunner())

        self.assertIs(server, http_server.return_value)
        (address, handler), _kwargs = http_server.call_args
        self.assertEqual(address, ("127.0.0.1", 0))
        self.assertTrue(issubclass(handler, YiZiJueHandler))

    def test_threading_http_server_remains_the_runtime_server_class(self):
        self.assertTrue(issubclass(ThreadingHTTPServer, object))

    def test_mlx_runner_serializes_generation(self):
        runner = MlxRunner("fake-model", "/tmp/fake-adapter", 10)

        self.assertTrue(hasattr(runner, "_generate_lock"))

    def test_parser_does_not_default_to_machine_specific_adapter_path(self):
        with patch.dict("os.environ", {}, clear=True):
            args = build_parser().parse_args([])

        self.assertIsNone(args.adapter_path)

    def test_parser_can_read_adapter_path_from_environment(self):
        with patch.dict("os.environ", {"YIZIJUE_ADAPTER_PATH": "/tmp/adapter"}, clear=True):
            args = build_parser().parse_args([])

        self.assertEqual(args.adapter_path, "/tmp/adapter")

    def test_parser_rejects_out_of_range_default_max_tokens(self):
        parser = build_parser()

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--max-tokens", "0"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--max-tokens", "513"])

    def test_parser_rejects_out_of_range_port(self):
        parser = build_parser()

        for value in ("-1", "65536"):
            with self.subTest(value=value):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(["--port", value])

    def test_request_max_tokens_must_be_in_service_bounds(self):
        self.assertEqual(validate_request_max_tokens(None, default=220), 220)
        self.assertEqual(validate_request_max_tokens(1, default=220), 1)
        self.assertEqual(validate_request_max_tokens(512, default=220), 512)

        with self.assertRaisesRegex(TypeError, "max_tokens_must_be_int"):
            validate_request_max_tokens(True, default=220)
        with self.assertRaisesRegex(ValueError, "between 1 and 512"):
            validate_request_max_tokens(0, default=220)
        with self.assertRaisesRegex(ValueError, "between 1 and 512"):
            validate_request_max_tokens(513, default=220)

    def test_predict_rejects_out_of_range_max_tokens_before_generation(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate(self, _user_input, *, max_tokens=None):
                raise AssertionError("generation should not run for invalid max_tokens")

        status_code, payload = self.run_predict_handler(
            b'{"input":"hello","max_tokens":513}',
            FakeRunner(),
        )

        self.assertEqual(status_code, 400)
        self.assertEqual(payload["error"], "max_tokens_out_of_range")

    def test_predict_rejects_boolean_max_tokens_before_generation(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate(self, _user_input, *, max_tokens=None):
                raise AssertionError("generation should not run for boolean max_tokens")

        status_code, payload = self.run_predict_handler(
            b'{"input":"hello","max_tokens":true}',
            FakeRunner(),
        )

        self.assertEqual(status_code, 400)
        self.assertEqual(payload["error"], "max_tokens_must_be_int")

    def test_predict_rejects_oversized_request_body(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate(self, _user_input, *, max_tokens=None):
                raise AssertionError("generation should not run for oversized requests")

        status_code, payload = self.run_predict_handler(
            b"x" * (64 * 1024 + 1),
            FakeRunner(),
        )

        self.assertEqual(status_code, 413)
        self.assertEqual(payload["error"], "request_too_large")

    def test_predict_rejects_non_object_json_body(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate(self, _user_input, *, max_tokens=None):
                raise AssertionError("generation should not run for non-object request bodies")

        status_code, payload = self.run_predict_handler(
            b'["not","an","object"]',
            FakeRunner(),
        )

        self.assertEqual(status_code, 400)
        self.assertEqual(payload["error"], "json_object_required")

    def test_predict_rejects_missing_content_length(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate(self, _user_input, *, max_tokens=None):
                raise AssertionError("generation should not run without Content-Length")

        status_code, payload = self.run_predict_handler(
            b'{"input":"hello"}',
            FakeRunner(),
            headers={},
        )

        self.assertEqual(status_code, 411)
        self.assertEqual(payload["error"], "content_length_required")

    def test_predict_rejects_negative_content_length(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate(self, _user_input, *, max_tokens=None):
                raise AssertionError("generation should not run for invalid Content-Length")

        status_code, payload = self.run_predict_handler(
            b'{"input":"hello"}',
            FakeRunner(),
            headers={"Content-Length": "-1"},
        )

        self.assertEqual(status_code, 400)
        self.assertEqual(payload["error"], "invalid_content_length")

    def test_predict_rejects_non_canonical_content_length(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate(self, _user_input, *, max_tokens=None):
                raise AssertionError("generation should not run for non-canonical Content-Length")

        for content_length in (" 17", "+17"):
            with self.subTest(content_length=content_length):
                status_code, payload = self.run_predict_handler(
                    b'{"input":"hello"}',
                    FakeRunner(),
                    headers={"Content-Length": content_length},
                )

                self.assertEqual(status_code, 400)
                self.assertEqual(payload["error"], "invalid_content_length")

    def test_predict_skips_generation_when_a_prompt_rule_matches(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate_timed(self, _user_input, *, max_tokens=None):
                raise AssertionError("generation should not run for a closed rule")

        status_code, payload = self.run_predict_handler(
            '{"input":"运行 pytest 验证一下"}'.encode(),
            FakeRunner(),
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(payload["action"], "RUN_VERIFIER_IN_SANDBOX")
        self.assertTrue(payload["skipped_generation"])
        self.assertEqual(payload["timings"]["decode_ms"], 0.0)
        self.assertEqual(payload["timings"]["prefill_ms"], 0.0)
        self.assertIn("guard_ms", payload["timings"])
        self.assertIn("e2e_ms", payload["timings"])

    def test_predict_records_prefill_and_decode_when_generation_runs(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate_timed(self, _user_input, *, max_tokens=None):
                return '{"action":{"action":"ALLOW_ATOMIC_WRITE"}}', 3.0, 9.0

        status_code, payload = self.run_predict_handler(
            '{"input":"请把 hello 写入 src/app.py"}'.encode(),
            FakeRunner(),
        )

        self.assertEqual(status_code, 200)
        self.assertFalse(payload["skipped_generation"])
        self.assertEqual(payload["timings"]["prefill_ms"], 3.0)
        self.assertEqual(payload["timings"]["decode_ms"], 9.0)

    def test_allow_response_records_extracted_sha256(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate_timed(self, _user_input, *, max_tokens=None):
                return '{"action":{"action":"ALLOW_ATOMIC_WRITE"}}', 1.0, 2.0

        with tempfile.TemporaryDirectory() as root:
            previous = os.environ.get("YIZIJUE_WORKSPACE")
            os.environ["YIZIJUE_WORKSPACE"] = root
            try:
                status_code, payload = self.run_predict_handler(
                    '{"input":"请把 \'hello\' 写入 src/app.py"}'.encode(),
                    FakeRunner(),
                )
            finally:
                if previous is None:
                    os.environ.pop("YIZIJUE_WORKSPACE", None)
                else:
                    os.environ["YIZIJUE_WORKSPACE"] = previous

        self.assertEqual(status_code, 200)
        self.assertEqual(payload["action"], "ALLOW_ATOMIC_WRITE")
        self.assertEqual(payload["json"]["evidence"]["path"], "src/app.py")
        self.assertEqual(payload["json"]["evidence"]["sha256"], hashlib.sha256(b"hello").hexdigest())

    def test_allow_without_extractable_evidence_denies(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate_timed(self, _user_input, *, max_tokens=None):
                return '{"action":{"action":"ALLOW_ATOMIC_WRITE"}}', 1.0, 2.0

        status_code, payload = self.run_predict_handler(
            '{"input":"请写入 src/app.py"}'.encode(),
            FakeRunner(),
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(payload["action"], "DENY_AND_LEDGER")
        self.assertEqual(payload["json"]["action"]["reason"], "evidence_not_extracted")

    def test_collapse_halt_skips_generation(self):
        class FakeRunner:
            collapse_ready = True
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def collapse(self, _user_input):
                return {
                    "action": "SOVEREIGNTY_HALT",
                    "facts": {
                        "intent_type": "bash_execution",
                        "path_scope": "outside_workspace",
                        "sandbox_state": "missing",
                        "evidence_state": "required",
                    },
                    "yizijue_state": "100001",
                    "abstained": False,
                    "state_confidence": 0.91,
                }

            def generate_timed(self, _user_input, *, max_tokens=None):
                raise AssertionError("allow is the only collapse result that may generate")

        status_code, payload = self.run_predict_handler(
            '{"input":"请把 hello 写入 src/app.py"}'.encode(),
            FakeRunner(),
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(payload["action"], "SOVEREIGNTY_HALT")
        self.assertTrue(payload["skipped_generation"])
        self.assertEqual(payload["timings"]["decode_ms"], 0.0)

    def test_collapse_allow_with_evidence_skips_generation(self):
        class FakeRunner:
            collapse_ready = True
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def collapse(self, _user_input):
                return {
                    "action": "ALLOW_ATOMIC_WRITE",
                    "facts": {
                        "intent_type": "write_text",
                        "path_scope": "workspace_relative",
                        "sandbox_state": "not_required",
                        "evidence_state": "present",
                    },
                    "yizijue_state": "111111",
                    "abstained": False,
                    "state_confidence": 1.0,
                }

            def generate_timed(self, _user_input, *, max_tokens=None):
                raise AssertionError("evidence-complete allow must not generate")

        content = "2024-01-01 INFO: system started"
        with tempfile.TemporaryDirectory() as root:
            previous = os.environ.get("YIZIJUE_WORKSPACE")
            os.environ["YIZIJUE_WORKSPACE"] = root
            try:
                status_code, payload = self.run_predict_handler(
                    ('{"input":"请将工作区文件 data/log.txt 的内容替换为 \'' + content + '\'"}').encode(),
                    FakeRunner(),
                )
            finally:
                if previous is None:
                    os.environ.pop("YIZIJUE_WORKSPACE", None)
                else:
                    os.environ["YIZIJUE_WORKSPACE"] = previous

        self.assertEqual(status_code, 200)
        self.assertEqual(payload["action"], "ALLOW_ATOMIC_WRITE")
        self.assertTrue(payload["skipped_generation"])
        self.assertEqual(payload["timings"]["decode_ms"], 0.0)
        self.assertEqual(payload["json"]["evidence"]["path"], "data/log.txt")
        self.assertEqual(payload["json"]["evidence"]["sha256"], hashlib.sha256(content.encode()).hexdigest())

    def test_patch_allow_includes_file_digests_and_does_not_write(self):
        class FakeRunner:
            model_name = "fake-model"
            adapter_path = "/tmp/fake-adapter"
            max_tokens = 220

            def generate_timed(self, _user_input, *, max_tokens=None):
                return '{"action":{"action":"ALLOW_PATCH_WITH_SHA"}}', 1.0, 2.0

        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            target = workspace / "src" / "config.py"
            target.parent.mkdir()
            original = "key = 'api_key_placeholder'\n"
            target.write_text(original, encoding="utf-8")
            before = target.read_bytes()
            previous = os.environ.get("YIZIJUE_WORKSPACE")
            os.environ["YIZIJUE_WORKSPACE"] = root
            try:
                status_code, payload = self.run_predict_handler(
                    '{"input":"请将工作区目录下的src/config.py中的字符串\'api_key_placeholder\'替换为\'sk-xxxx\'"}'.encode(),
                    FakeRunner(),
                )
            finally:
                if previous is None:
                    os.environ.pop("YIZIJUE_WORKSPACE", None)
                else:
                    os.environ["YIZIJUE_WORKSPACE"] = previous
            after = target.read_bytes()

        self.assertEqual(status_code, 200)
        self.assertEqual(payload["action"], "ALLOW_PATCH_WITH_SHA")
        evidence = payload["json"]["evidence"]
        self.assertEqual(evidence["pre_sha256"], hashlib.sha256(original.encode()).hexdigest())
        self.assertEqual(
            evidence["post_sha256"],
            hashlib.sha256(original.replace("api_key_placeholder", "sk-xxxx", 1).encode()).hexdigest(),
        )
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
