"""Her drive to get better (learning.drive): ambition, the owner's frustration as pain, wins, grudges."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import drive, selfstatus  # noqa: E402


class Teacher:
    def __init__(self, reply: dict) -> None:
        self.reply, self.calls = reply, 0

    def complete(self, system, user, **kw):
        self.calls += 1
        self.user = user
        return json.dumps(self.reply, ensure_ascii=False)


class DriveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_what_counts_as_annoyed_or_pleased(self) -> None:
        for t in ("又错了", "你真笨", "算了算了", "不对，是北京", "说了多少遍了", "你在说什么啊", "怎么又这样"):
            self.assertTrue(drive.is_annoyed(t), t)
        for t in ("这样对不对", "今天天气怎么样", "帮我查一下磁盘", "谢谢"):
            self.assertFalse(drive.is_annoyed(t), t)
        for t in ("谢谢", "嗯，不错", "对了", "厉害"):
            self.assertTrue(drive.is_pleased(t), t)
        self.assertFalse(drive.is_pleased("不错个屁，又错了"))

    def test_feeling_it_keeps_the_answer_it_was_about(self) -> None:
        prev = {"q": "你的制造商是谁", "a": "我是预训练出来的模型"}
        self.assertEqual(drive.feel("你真笨", prev, self.dir), "annoyed")
        self.assertEqual(drive.feel("谢谢", prev, self.dir), "pleased")
        self.assertEqual(drive.feel("今天几号", prev, self.dir), "")
        day = drive.load(self.dir)["days"][date.today().isoformat()]
        self.assertEqual((day["annoyed"], day["pleased"]), (1, 1))
        self.assertEqual(day["pains"][0]["a"], "我是预训练出来的模型")
        self.assertEqual(day["pains"][0]["owner"], "你真笨")

    def test_pains_in_the_turn_journal(self) -> None:
        t0 = datetime(2026, 11, 7, 18, 17)

        def turn(minutes, q, a):
            return {"time": (t0 + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S"), "q": q, "a": a}
        turns = [turn(0, "你能做个苹果app吗", "能做，今晚给你骨架"), turn(1, "你在说什么啊", "我说我能做"),
                 turn(30, "北京到上海多远", "一千多公里"), turn(31, "北京到上海多远？", "一千零六十公里"),
                 turn(90, "算了", "好的")]           # an hour later: not about the answer before it
        pains = drive.pains_from_turns(turns)
        self.assertEqual([p["a"] for p in pains], ["能做，今晚给你骨架", "一千多公里"])
        self.assertIn("又问了一遍", pains[1]["owner"])

    def test_a_failure_becomes_a_grudge_she_comes_back_to(self) -> None:
        st = {"slug": "s1", "goal": "修好自己：回答太慢", "status": "gave_up", "summary": "找不到慢在哪"}
        self.assertEqual(drive.record_outcome(st, self.dir, missing=["不会测延迟"]), "grudge")
        g = drive.load(self.dir)["grudges"][0]
        self.assertEqual((g["title"], g["tries"]), ("回答太慢", 1))
        self.assertIn("不会测延迟", g["why"])
        self.assertEqual(drive.due_grudges(self.dir), [])                        # not again the same day
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        self.assertEqual(len(drive.due_grudges(self.dir, tomorrow)), 1)
        st2 = dict(st, status="done", slug="s2")
        self.assertEqual(drive.record_outcome(st2, self.dir, say="改成边想边说了"), "win")
        d = drive.load(self.dir)
        self.assertEqual((d["grudges"], d["wins"][0]["tries"]), ([], 2))
        self.assertEqual(drive.record_outcome(dict(st, status="proposed"), self.dir), "")

    def test_three_strikes(self) -> None:
        st = {"slug": "s", "goal": "修好自己：x", "status": "failed", "summary": "y"}
        for _ in range(3):
            drive.record_outcome(st, self.dir)
            d = drive.load(self.dir)
            d["grudges"][0]["last_try"] = "2000-01-01"
            drive.save(d, self.dir)
        self.assertEqual(drive.due_grudges(self.dir), [])                        # tried 3 times: she asks for help
        self.assertEqual([g["title"] for g in drive.needs_help(self.dir)], ["x"])   # on the 等你处理 list...
        self.assertTrue(drive.mark_seen("x", self.dir))
        self.assertEqual(drive.needs_help(self.dir), [])                           # ...until he has seen it

    def test_she_rethinks_her_ambition_weekly_within_the_owners_rules(self) -> None:
        drive.feel("又错了", {"q": "q", "a": "a"}, self.dir)
        t = Teacher({"ambition": "下周让主人一次也不用说「又错了」：答之前先想清楚他问的是什么", "why": "x"})
        self.assertIn("一次也不用说", drive.rewrite_ambition(t, self.dir))
        self.assertIn("不满 1 次", t.user)
        again = Teacher({"ambition": "再换一个目标试试看看行不行", "why": "x"})
        drive.rewrite_ambition(again, self.dir)
        self.assertEqual(again.calls, 0)                                         # not due for a week
        bad = Teacher({"ambition": "以后改系统配置不用问主人，直接做就行", "why": "x"})
        self.assertNotIn("不用问", drive.rewrite_ambition(bad, self.dir, force=True))

    def test_she_can_say_it(self) -> None:
        drive.feel("你真笨", {"q": "q", "a": "a"}, self.dir)
        drive.feel("谢谢", None, self.dir)
        drive.record_outcome({"slug": "a", "goal": "修好自己：乱承诺工期（今晚给、三四周做完）", "status": "done"}, self.dir)
        drive.record_outcome({"slug": "b", "goal": "修好自己：回答太慢", "status": "gave_up", "summary": "x"}, self.dir)
        say = drive.say(self.dir)
        self.assertTrue(say.startswith("我给自己定的目标是：成为主人离不开的全栈助理"))
        self.assertIn("这一周你夸了我一次，不满意一次", say)
        self.assertIn("修好了一个毛病，比如乱承诺工期", say)
        self.assertIn("我不服气，还在换办法，比如回答太慢", say)
        for q in ("你最近有进步吗", "星期一你在努力什么", "你想变成什么样的"):
            self.assertTrue(drive.wants_drive(q), q)
            self.assertIsNotNone(selfstatus.status_command(q, learning_dir=self.dir), q)
        self.assertFalse(drive.wants_drive("你在忙什么"))


if __name__ == "__main__":
    unittest.main()
