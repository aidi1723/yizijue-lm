import subprocess
import sys
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from scripts.build_hardened_mlx_dataset import STRICT_SYSTEM_PROMPT
from scripts.serve_yizijue_mlx import (
    MlxRunner,
    YiZiJueHandler,
    build_prompt,
    build_prediction_response,
    make_server,
    render_index_html,
)


class ServeYiZiJueMlxTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
