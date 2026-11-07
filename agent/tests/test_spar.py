"""陪练 (learning.spar): the teacher plays the owner, her local model answers, the answers are judged and kept."""
from __future__ import annotations

import datetime as dt
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import drive, spar  # noqa: E402

PERSONA = "你是星期一，称呼对方为主人。"


class Teacher:
    """Plays the owner from a script and judges from a script; records what it was shown."""

    def __init__(self, says: list[dict], verdicts: list[dict]) -> None:
        self.says, self.verdicts = list(says), list(verdicts)
        self.owner_prompts: list[str] = []
        self.judge_prompts: list[str] = []

    def complete(self, system, user, **kw):
        if system == spar.OWNER_SYSTEM:
            self.owner_prompts.append(user)
            return json.dumps(self.says.pop(0) if self.says else {"say": "", "end": True}, ensure_ascii=False)
        assert system == spar.JUDGE_SYSTEM
        self.judge_prompts.append(user)
        return '{"thought": "看看"}\n' + json.dumps(self.verdicts.pop(0), ensure_ascii=False)   # two parts, as DeepSeek does


class SparTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        (self.dir / "turns").mkdir()
        today = dt.date.today().isoformat()
        rows = [{"time": f"{today} 18:1{i}:00", "q": q, "a": "x"} for i, q in
                enumerate(["你的制造商是谁", "你能做一个苹果app吗", "泰勒公式", "嗯", "你的制造商是谁？"])]
        (self.dir / "turns" / f"{today}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        self.asked: list[list[dict]] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def her(self, replies: list[str]):
        replies = list(replies)

        def ask(m):
            self.asked.append(m)
            return replies.pop(0)
        return ask

    def run_session(self, teacher, replies, **kw):
        return spar.session(teacher, self.her(replies), persona=PERSONA, system="SYS", facts="我现在是第五版。",
                            seeds=["你的制造商是谁"], focus="问问她自己", learning_dir=self.dir, **kw)

    def rows(self, name):
        f = self.dir / name
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.is_file() else []

    def test_his_real_words_are_the_seeds(self) -> None:
        seeds = spar.owner_seeds(self.dir, rng=random.Random(1))
        self.assertEqual(sorted(seeds), sorted(["你的制造商是谁", "你能做一个苹果app吗", "泰勒公式"]))   # once each, no "嗯"

    def test_a_session_keeps_good_answers_and_corrects_bad_ones(self) -> None:
        t = Teacher([{"say": "你的制造商是谁", "end": False}, {"say": "你在说什么啊", "end": False}, {"say": "好吧", "end": True}],
                    [{"ok": False, "kind": "不认识自己", "problem": "没说自己是谁做的",
                      "better": "我是主人部署在这块板子上的星期一，底座是千问三点五，上面是第五版微调。"},
                     {"ok": True}, {"ok": True}])
        rec = self.run_session(t, ["我是预训练出来的模型。", "我是说，我是星期一，跑在板子上的语音助手。", "好的。"])
        self.assertEqual([x["kept"] for x in rec["turns"]], ["corrected", "hers", "hers"])
        samples = self.rows("distill/spar.jsonl")
        self.assertEqual(samples[0]["messages"][-1]["content"][:9], "我是主人部署在这块")
        self.assertEqual(samples[0]["source"], "spar")
        pair = self.rows("spar/pairs.jsonl")[0]
        self.assertEqual((pair["rejected"], pair["kind"]), ("我是预训练出来的模型。", "不认识自己"))
        # on-policy: he reacts to what she actually said, and her next turn sees it as history
        self.assertIn("星期一：我是预训练出来的模型。", t.owner_prompts[1])
        self.assertEqual(self.asked[1][1:3], [{"role": "user", "content": "你的制造商是谁"},
                                              {"role": "assistant", "content": "我是预训练出来的模型。"}])
        self.assertIn("第五版", t.judge_prompts[0])                            # the judge knows the facts about her

    def test_corrections_that_fail_the_checks_are_not_trained(self) -> None:
        t = Teacher([{"say": "提醒我八点开会", "end": False}, {"say": "写个函数", "end": True}],
                    [{"ok": False, "kind": "乱承诺", "problem": "x", "better": "好的，到点我会提醒您。"},          # a promise
                     {"ok": False, "kind": "念不出", "problem": "x", "better": "def f(x): return x * 2"}])         # not speakable
        rec = self.run_session(t, ["好的。", "好的。"])
        self.assertEqual([x["kept"] for x in rec["turns"]], ["", ""])
        self.assertEqual((self.rows("distill/spar.jsonl"), self.rows("spar/pairs.jsonl")), ([], []))

    def test_the_owner_comes_first(self) -> None:
        t = Teacher([{"say": "你好", "end": False}], [{"ok": True}])
        rec = self.run_session(t, ["你好。"], idle=lambda: False)
        self.assertEqual((rec["stopped"], rec["turns"]), (True, []))

    def test_a_night_stops_at_the_budget_and_records_the_trend(self) -> None:
        drive.feel("又错了", {"q": "你的制造商是谁", "a": "我是预训练出来的模型"}, self.dir)
        t = Teacher([{"say": "你是谁做的", "end": True}, {"say": "你几岁了", "end": True}],
                    [{"ok": False, "kind": "不认识自己", "problem": "x", "better": "我是主人在板子上部署的星期一。"}, {"ok": True}])
        spend = iter([(0.0, 0.0), (1.0, 1.0), (3.5, 3.5), (3.5, 3.5)])
        res = spar.night(t, self.her(["我是预训练模型。", "我刚上线没多久。"]), persona=PERSONA, facts="第五版",
                         learning_dir=self.dir, cost=lambda: next(spend), rng=random.Random(2))
        self.assertEqual((res["sessions"], res["why"]), (2, "budget"))
        n = res["night"]
        self.assertEqual((n["turns"], n["ok"], n["kept"], n["kinds"]), (2, 1, 2, {"不认识自己": 1}))
        self.assertIn("他最近不满意的地方", " ".join(spar.focuses(self.dir)))   # his real annoyances are topics
        say = spar.say(self.dir)
        self.assertIn("老师扮成你跟我聊了两句，我答好了百分之50，主要毛病是不认识自己", say)
        self.assertIn(say, drive.say(self.dir))
        self.assertTrue(drive.wants_drive("你昨晚陪练得怎么样"))

    def test_peak_and_month_cap(self) -> None:
        t = Teacher([], [])
        self.assertEqual(spar.night(t, self.her([]), persona=PERSONA, facts="", learning_dir=self.dir,
                                    cost=lambda: (0.0, 0.0), blocked=lambda: True)["why"], "peak")
        self.assertEqual(spar.night(t, self.her([]), persona=PERSONA, facts="", learning_dir=self.dir,
                                    cost=lambda: (0.0, spar.MONTH_YUAN))["why"], "month")

    def test_her_context_matches_a_live_turn(self) -> None:
        m = spar.messages("SYS", [("一" * 60, "二" * 100)] * 6, "现在")
        self.assertEqual(len(m), 1 + 2 * spar.HIST_N + 1)
        self.assertEqual((len(m[1]["content"]), len(m[2]["content"])), (spar.HIST_Q, spar.HIST_A))
        self.assertTrue(spar.chatml(m).endswith("<|im_start|>assistant\n"))
        self.assertTrue(spar.her_system(PERSONA, self.dir, dt.datetime(2026, 11, 8, 3, 5)).startswith(
            "当前时间：2026年11月8日，星期日，凌晨3点05分。\n你是星期一"))


if __name__ == "__main__":
    unittest.main()
