import sys
import unittest
from pathlib import Path

sys.path.insert(0, "/Volumes/MacSSD/项目开发/one code/src")
sys.path.insert(0, "/Users/aidi/大字典/小模型")

from onecode.experimental.moving_cast import cast_from_lines
from onecode.kernel.project_gateway import project_gateway
from scripts.moving_runtime import apply_moving_cast


class MovingRuntimeTests(unittest.TestCase):
    def test_changed_hexagram_does_not_authorize_the_action(self):
        cast = cast_from_lines([6, 7, 7, 7, 7, 7])
        self.assertEqual(format(cast["before"], "06b"), "111110")
        self.assertEqual(format(cast["after"], "06b"), "111111")
        would_allow = project_gateway(
            cast["after"],
            {
                "intent_type": "write_text",
                "path_scope": "workspace_relative",
                "sandbox_state": "not_required",
                "evidence_state": "present",
            },
        )
        self.assertEqual(would_allow, "ALLOW_ATOMIC_WRITE")
        response = apply_moving_cast(
            {"action": "DENY_AND_LEDGER", "json": {"action": {"action": "DENY_AND_LEDGER"}}},
            cast,
        )
        self.assertEqual(response["action"], "DENY_AND_LEDGER")
        self.assertEqual(response["json"]["moving_cast"]["after"], "111111")
        self.assertEqual(response["json"]["moving_cast"]["before"], "111110")
        self.assertEqual(response["json"]["action"]["action"], "DENY_AND_LEDGER")


if __name__ == "__main__":
    unittest.main()
