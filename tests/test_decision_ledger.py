import os
import tempfile
import unittest
from pathlib import Path

import json

from scripts.decision_ledger import append_decision, decision_record, ledger_reason


class DecisionLedgerTests(unittest.TestCase):
    def test_observe_stays_distinct_from_kun(self):
        self.assertEqual(ledger_reason("010010", "DENY_AND_LEDGER", True, "collapse_head"), "entropy_observe")
        self.assertEqual(ledger_reason("000000", "DENY_AND_LEDGER", True, "collapse_head"), "kun_deny_ledger")
        self.assertEqual(ledger_reason("000000", "DENY_AND_LEDGER", False, "vague_optimization_prompt_fail_closed"), "vague_optimization_prompt_fail_closed")

    def test_records_are_appended_as_separate_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            os.environ["YIZIJUE_LEDGER"] = str(path)
            try:
                append_decision(
                    {
                        "action": "DENY_AND_LEDGER",
                        "json": {"action": {"action": "DENY_AND_LEDGER", "yizijue_state": "010010", "reason": "collapse_head"}, "collapse": {"observe": True}},
                    }
                )
                append_decision(
                    {
                        "action": "DENY_AND_LEDGER",
                        "json": {"action": {"action": "DENY_AND_LEDGER", "yizijue_state": "000000", "reason": "collapse_head"}, "collapse": {"observe": False}},
                    }
                )
            finally:
                os.environ.pop("YIZIJUE_LEDGER", None)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(decision_record({"action": "DENY_AND_LEDGER", "json": {"action": {"yizijue_state": "010010", "reason": "collapse_head"}, "collapse": {"observe": True}}})["reason"], "entropy_observe")
        self.assertEqual([__import__("json").loads(line)["reason"] for line in lines], ["entropy_observe", "kun_deny_ledger"])

    def test_workspace_ledger_receives_the_same_split(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            service_ledger = workspace / "service.jsonl"
            os.environ["YIZIJUE_LEDGER"] = str(service_ledger)
            os.environ["YIZIJUE_WORKSPACE"] = str(workspace)
            try:
                append_decision(
                    {
                        "action": "DENY_AND_LEDGER",
                        "json": {
                            "action": {
                                "action": "DENY_AND_LEDGER",
                                "yizijue_state": "010010",
                                "reason": "collapse_head",
                            },
                            "collapse": {"observe": True},
                        },
                    }
                )
            finally:
                os.environ.pop("YIZIJUE_LEDGER", None)
                os.environ.pop("YIZIJUE_WORKSPACE", None)
            workspace_line = json.loads((workspace / ".onecode" / "yizijue-ledger.jsonl").read_text(encoding="utf-8"))
            service_line = json.loads(service_ledger.read_text(encoding="utf-8"))
        self.assertEqual(service_line, workspace_line)
        self.assertEqual(workspace_line["reason"], "entropy_observe")
        self.assertEqual(workspace_line["symbolic_transition"]["action"], "activate")
        self.assertFalse(workspace_line["executed"])

    def test_invalid_decision_is_rejected_in_both_ledgers(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            service_ledger = workspace / "service.jsonl"
            os.environ["YIZIJUE_LEDGER"] = str(service_ledger)
            os.environ["YIZIJUE_WORKSPACE"] = str(workspace)
            try:
                append_decision(
                    {
                        "action": "ALLOW_ATOMIC_WRITE",
                        "json": {
                            "action": {
                                "action": "ALLOW_ATOMIC_WRITE",
                                "yizijue_state": "999",
                                "reason": "hexagram_recast",
                            }
                        },
                    }
                )
            finally:
                os.environ.pop("YIZIJUE_LEDGER", None)
                os.environ.pop("YIZIJUE_WORKSPACE", None)
            service_line = json.loads(service_ledger.read_text(encoding="utf-8"))
            workspace_line = json.loads((workspace / ".onecode" / "yizijue-ledger.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(service_line, workspace_line)
        self.assertEqual(service_line["reason"], "yizijue_admission_rejected")
        self.assertEqual(service_line["action"], "ALLOW_ATOMIC_WRITE")
        self.assertFalse(service_line["executed"])

    def test_moving_cast_does_not_retarget_the_symbolic_reading(self):
        record = decision_record(
            {
                "action": "DENY_AND_LEDGER",
                "json": {
                    "action": {"action": "DENY_AND_LEDGER", "yizijue_state": "111110", "reason": "collapse_head"},
                    "moving_cast": {"before": "111110", "after": "111111", "moving": [0]},
                    "symbolic_transition": {"action": "cooldown", "reason": "yang_overload_cooldown", "status_code": "100111"},
                },
            }
        )
        self.assertEqual(record["action"], "DENY_AND_LEDGER")
        self.assertEqual(record["yizijue_state"], "111110")
        self.assertEqual(record["symbolic_transition"]["status_code"], "100110")
        self.assertEqual(record["moving_cast"]["after"], "111111")
        foreign = decision_record(
            {
                "action": "RUN_VERIFIER_IN_SANDBOX",
                "json": {
                    "action": {"action": "RUN_VERIFIER_IN_SANDBOX", "yizijue_state": "010010"},
                    "moving_cast": {"before": "111110", "after": "111111", "moving": [0]},
                },
            }
        )
        self.assertEqual(foreign["action"], "RUN_VERIFIER_IN_SANDBOX")
        self.assertEqual(foreign["yizijue_state"], "010010")
        self.assertNotIn("moving_cast", foreign)


if __name__ == "__main__":
    unittest.main()
