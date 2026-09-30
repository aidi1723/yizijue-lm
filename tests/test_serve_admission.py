import json
import os
import tempfile
import unittest
from pathlib import Path

from onecode.experimental.yizijue_verifier import pinned_verifier_command
from onecode.kernel.checkpoint import sha256_text
from scripts.serve_yizijue_mlx import _with_moving_cast


class QuietRunner:
    def read_moving_cast(self, _user_input: str):
        return None


def kan_response() -> dict:
    return {
        "input": "跑单元测试",
        "action": "RUN_VERIFIER_IN_SANDBOX",
        "json": {
            "action": {
                "action": "RUN_VERIFIER_IN_SANDBOX",
                "yizijue_state": "010010",
                "reason": "verifier_requires_sandbox",
                "facts": {
                    "intent_type": "execute_pytest",
                    "path_scope": "no_path",
                    "sandbox_state": "required",
                    "evidence_state": "required",
                },
            }
        },
    }


def observe_response() -> dict:
    return {
        "input": "今天把笔记整理成三条",
        "action": "DENY_AND_LEDGER",
        "json": {
            "action": {
                "action": "DENY_AND_LEDGER",
                "yizijue_state": "010010",
                "reason": "entropy_observe",
            },
            "collapse": {"observe": True},
        },
    }


class ServiceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: os.environ.get(key)
            for key in ("YIZIJUE_WORKSPACE", "YIZIJUE_RUN_VERIFIER", "YIZIJUE_RUN_WRITE", "YIZIJUE_LEDGER")
        }
        for key in self._saved:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_flags_off_record_the_decision_without_running(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["YIZIJUE_WORKSPACE"] = directory
            response = _with_moving_cast(kan_response(), "跑单元测试", QuietRunner())
            lines = (Path(directory) / ".onecode" / "yizijue-ledger.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertNotIn("verifier", response)
        self.assertNotIn("write", response)
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["kind"], "yizijue_decision")
        self.assertFalse(json.loads(lines[0])["executed"])

    def test_predict_decision_enters_the_cycle_gates(self):
        calls = []

        def verifier(_workspace, command):
            calls.append(("verifier", command))
            return {"status": "passed", "command": list(command)}

        def writer(workspace, relative_path, content):
            calls.append(("write", relative_path, content))
            target = Path(workspace) / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"path": relative_path}

        content = "三条笔记"
        user_input = "把 '三条笔记' 写入 notes/today.txt"
        allow = {
            "input": user_input,
            "action": "ALLOW_ATOMIC_WRITE",
            "json": {
                "action": {
                    "action": "ALLOW_ATOMIC_WRITE",
                    "yizijue_state": "111111",
                    "reason": "hexagram_recast",
                    "facts": {
                        "intent_type": "write_text",
                        "path_scope": "workspace_relative",
                        "sandbox_state": "not_required",
                        "evidence_state": "present",
                    },
                },
                "evidence": {"path": "notes/today.txt", "sha256": sha256_text(content)},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            os.environ["YIZIJUE_WORKSPACE"] = directory
            os.environ["YIZIJUE_RUN_VERIFIER"] = "1"
            verified = _with_moving_cast(kan_response(), "跑单元测试", QuietRunner(), verifier_runner=verifier)
            observed = _with_moving_cast(observe_response(), "今天把笔记整理成三条", QuietRunner(), verifier_runner=verifier)
            os.environ.pop("YIZIJUE_RUN_VERIFIER", None)
            os.environ["YIZIJUE_RUN_WRITE"] = "1"
            written = _with_moving_cast(allow, user_input, QuietRunner(), write_runner=writer)
            lines = (Path(directory) / ".onecode" / "yizijue-ledger.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(calls[0], ("verifier", pinned_verifier_command()))
        self.assertTrue(verified["verifier"]["ran"])
        self.assertEqual(verified["reason"], "verifier_requires_sandbox")
        self.assertEqual(verified["verifier"]["status"], "passed")
        self.assertFalse(observed["verifier"]["ran"])
        self.assertEqual(observed["verifier"]["reason"], "withheld")
        self.assertEqual(observed["reason"], "entropy_observe")
        self.assertEqual(calls[1], ("write", "notes/today.txt", content))
        self.assertTrue(written["write"]["ran"])
        self.assertEqual(written["reason"], "hexagram_recast")
        self.assertEqual(written["json"]["evidence"]["path"], "notes/today.txt")
        records = [json.loads(line) for line in lines]
        self.assertEqual([item["kind"] for item in records], ["yizijue_decision", "yizijue_effect"] * 3)
        self.assertEqual([item["executed"] for item in records], [False, True, False, False, False, True])

    def test_failed_write_reason_is_visible_on_the_response(self):
        def writer(workspace, relative_path, content):
            raise AssertionError("a hash mismatch must not write")

        response = {
            "action": "ALLOW_ATOMIC_WRITE",
            "json": {
                "action": {
                    "action": "ALLOW_ATOMIC_WRITE",
                    "yizijue_state": "111111",
                    "reason": "hexagram_recast",
                    "facts": {
                        "intent_type": "write_text",
                        "path_scope": "workspace_relative",
                        "sandbox_state": "not_required",
                        "evidence_state": "present",
                    },
                },
                "evidence": {"path": "notes/today.txt", "sha256": sha256_text("三条笔记")},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            os.environ["YIZIJUE_WORKSPACE"] = directory
            os.environ["YIZIJUE_RUN_WRITE"] = "1"
            os.environ["YIZIJUE_LEDGER"] = str(Path(directory) / "service.jsonl")
            result = _with_moving_cast(response, "把 '别的内容' 写入 notes/today.txt", QuietRunner(), write_runner=writer)
            lines = (Path(directory) / ".onecode" / "yizijue-ledger.jsonl").read_text(encoding="utf-8").splitlines()
            service_lines = (Path(directory) / "service.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(result["action"], "ALLOW_ATOMIC_WRITE")
        self.assertEqual(result["reason"], "sha_mismatch")
        self.assertEqual(result["json"]["action"]["reason"], "sha_mismatch")
        self.assertNotIn("evidence", result["json"])
        self.assertNotIn("sha256", result["guarded_prediction"])
        self.assertIn("sha_mismatch", result["guarded_prediction"])
        self.assertFalse(result["write"]["executed"])
        records = [json.loads(line) for line in lines]
        self.assertEqual(records[-1]["reason"], "sha_mismatch")
        self.assertFalse(records[-1]["executed"])
        self.assertEqual(records[-1]["kind"], "yizijue_decision")
        self.assertEqual([json.loads(line) for line in service_lines], records)

    def test_invalid_admitted_decision_is_recorded_once(self):
        response = {
            "action": "ALLOW_ATOMIC_WRITE",
            "json": {
                "action": {
                    "action": "ALLOW_ATOMIC_WRITE",
                    "yizijue_state": "999",
                    "reason": "hexagram_recast",
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            os.environ["YIZIJUE_WORKSPACE"] = directory
            os.environ["YIZIJUE_RUN_WRITE"] = "1"
            os.environ["YIZIJUE_LEDGER"] = str(Path(directory) / "service.jsonl")
            result = _with_moving_cast(response, "把 '三条笔记' 写入 notes/today.txt", QuietRunner())
            lines = (Path(directory) / ".onecode" / "yizijue-ledger.jsonl").read_text(encoding="utf-8").splitlines()
            service_lines = (Path(directory) / "service.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(result["reason"], "yizijue_admission_rejected")
        self.assertEqual(len(lines), 1)
        self.assertEqual([json.loads(line) for line in service_lines], [json.loads(line) for line in lines])
        self.assertFalse(json.loads(lines[0])["executed"])


if __name__ == "__main__":
    unittest.main()
