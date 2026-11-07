"""Replay the 2026-11-07 18:17 turns: a board-mounted voice assistant must not promise
"tonight", "tomorrow morning" or "three to four weeks" for an iOS app it cannot build."""
from __future__ import annotations

import unittest

from learning.brain import teacher_system


class TimelinePromiseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = teacher_system("你是星期一，板子上的语音助手。", "")

    def test_system_forbids_delivery_timeline_promises(self) -> None:
        self.assertIn("工期", self.system)
        self.assertIn("今晚", self.system)
        self.assertIn("明早", self.system)
        self.assertIn("周", self.system)
        self.assertIn("不能", self.system)
        # must tell the model to separate "能做" from "我能替你做完"
        self.assertIn("替", self.system)

    def test_timeline_rules_reach_final_system(self) -> None:
        # guard against dead code: rules must be appended, not just defined
        self.assertIn("不是一支开发团队", self.system)
        self.assertIn("需要主人自己", self.system)

    def test_memory_card_rules_still_present(self) -> None:
        self.assertIn("【记忆】", self.system)


if __name__ == "__main__":
    unittest.main()
