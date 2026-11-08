from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from orchestrator.dialogue_policy import unclear_reply  # noqa: E402


RECENT = [
    {"q": "这就谁的", "a": "这台路由器是您姐上次来装的", "who": "local"},
]


class UnclearFallbackTests(unittest.TestCase):
    def test_first_unclear_gives_a_guess_from_context(self) -> None:
        reply = unclear_reply(RECENT, 1, previous_reply="")
        # 第一轮不能只是「没听明白，再说一遍」，要带上文线索的具体猜测
        self.assertNotIn("换个说法再说一遍", reply)
        self.assertIn("路由器", reply)

    def test_second_unclear_offers_two_options(self) -> None:
        first = unclear_reply(RECENT, 1, previous_reply="")
        second = unclear_reply(RECENT, 2, previous_reply=first)
        self.assertIn("A", second)
        self.assertIn("B", second)
        self.assertNotEqual(first, second)

    def test_third_unclear_admits_not_heard(self) -> None:
        prev = unclear_reply(RECENT, 2, previous_reply="")
        third = unclear_reply(RECENT, 3, previous_reply=prev)
        self.assertTrue("没听清" in third or "没听明白" in third)

    def test_reply_never_repeats_previous_one(self) -> None:
        prev = "这句我没听明白，能换个说法再说一遍吗？"
        for n in (1, 2, 3, 4, 5):
            reply = unclear_reply(RECENT, n, previous_reply=prev)
            self.assertNotEqual(reply.strip(), prev.strip())
            self.assertTrue(reply.strip())
            prev = reply

    def test_no_context_still_varies(self) -> None:
        a = unclear_reply([], 1, previous_reply="")
        b = unclear_reply([], 2, previous_reply=a)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
