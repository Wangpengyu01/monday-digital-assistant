"""She looks through her own conversations and logs and starts a repair (learning.selfreview)."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import drive, project, selfreview  # noqa: E402

# what the log looked like on 2026-11-07 when she could not say her own version
LOG = """17:48:02 INFO [llm] ack before answer: 嗯嗯，让我想一想，
17:48:03 INFO [agent] task: 查一下你自己（星期一）当前的版本号
17:48:47 INFO [tts] speak (1/3): 这个我还不太会，
17:48:55 INFO [agent] approval 54057132: 执行 ls -la /userdata/agent (danger, 涉及系统目录或星期一自己的代码)
17:50:24 WARNING [agent] unusable plan (attempt 1): {"thought": "README 里提到 v1.0"}
17:50:26 INFO [agent] done ok=False learned=False steps=3: 老师给的步骤我没看懂，这次先没做成。
17:51:00 ERROR [turn] processing failed
Traceback (most recent call last):
  File "/userdata/agent/orchestrator/main.py", line 10, in x
KeyError: 'q'
17:52:00 INFO [tts] speak (1/1): 没等到你的确认，这一步我就不做了。
18:18:20 INFO [tts] speak (1/1): 嗯，我想一想哈，
18:18:21 INFO [intent] chat (teacher): 你的制造商是谁？
18:18:35 INFO [tts] speak (1/1): 我是预训练出来的模型。
18:19:18 INFO [intent] knowledge (teacher): 泰勒公式
18:19:22 INFO [tts] speak (1/2): 泰勒公式就是……
"""


class Teacher:
    def __init__(self, problems: list[dict]) -> None:
        self.problems, self.calls = problems, []

    def complete(self, system, user, **kw):
        assert system == selfreview.REVIEW_SYSTEM
        self.calls.append(user)
        return '{"thought": "看了日志"}\n' + json.dumps({"problems": self.problems}, ensure_ascii=False)  # two parts, as DeepSeek does


PROBLEM = {"title": "问她版本号答不上来", "symptom": "主人问了四次版本", "evidence": "17:48 老师给的步骤我没看懂",
           "cause_guess": "没有版本信息", "files": ["orchestrator/main.py"], "verify": "问版本能直接答"}


class SelfReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root, self.learning, self.projects = base / "agent", base / "learning", base / "projects"
        for d in (self.root / "logs", self.root / "learning", self.learning / "turns", self.projects):
            d.mkdir(parents=True)
        (self.root / "learning" / "agent.py").write_text("x = 1\n", encoding="utf-8")
        self.log = self.root / "logs" / "orchestrator.log"
        self.log.write_text(LOG, encoding="utf-8")
        now = datetime.now()
        turns = [{"time": (now - timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"), "who": w, "reason": r, "q": q, "a": a}
                 for h, w, r, q, a in [(3, "teacher", "time-sensitive", "现在的版本", "直接查程序名加 version 参数"),
                                       (2, "agent", "failed", "查一下你自己的版本号", "老师给的步骤我没看懂"),
                                       (1, "local", "chat", "现在的版本？", "我也不清楚"),
                                       (1, "local", "chat", "澳大利亚的首都", "堪培拉")]]
        (self.learning / "turns" / f"{now.date().isoformat()}.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in turns) + "\n", encoding="utf-8")
        self.started: list[str] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def review(self, teacher, **kw):
        return selfreview.review(teacher, learning_dir=self.learning, root=self.root, projects_root=self.projects,
                                 start_project=lambda slug: self.started.append(slug) or "", busy=lambda: False, **kw)

    def test_the_evidence_names_what_the_owner_saw(self) -> None:
        body, info = selfreview.evidence(self.learning, self.root)
        self.assertTrue(info["found"])
        self.assertIn("办事没办成｜主人：查一下你自己的版本号", body)
        self.assertIn("主人同一个问题问了 2 次：「现在的版本", body)          # asked again: she did not get it
        self.assertIn("老师给的步骤她解析不了：1 次", body)
        self.assertIn("当成危险操作去问主人", body)
        self.assertIn("问主人要不要做，主人没来得及回答就取消了：1 次", body)
        self.assertIn("KeyError: 'q'", body)                                # the exception, not just "Traceback"
        self.assertIn("后 44 秒才开口", body)
        every, flagged = body.split("明显出问题的地方")[0], body.split("明显出问题的地方")[1].split("日志信号")[0]
        self.assertIn("堪培拉", every)                 # every turn is read: soft mistakes only show in good-looking ones
        self.assertNotIn("堪培拉", flagged)
        self.assertIn("「你的制造商是谁？」14 秒才说出答案", body)

    def test_a_problem_becomes_a_repair_project(self) -> None:
        t = Teacher([PROBLEM])
        rec = self.review(t)
        self.assertEqual((rec["started"], rec["title"]), (self.started[0], "问她版本号答不上来"))
        st = project.load_state(self.projects / rec["started"])
        self.assertEqual((st["kind"], st["goal"]), ("repair", "修好自己：问她版本号答不上来"))
        brief = (self.projects / rec["started"] / "brief.md").read_text(encoding="utf-8")
        self.assertIn("主人问了四次版本", brief)
        self.assertIn("orchestrator/main.py", brief)
        self.assertIn("learning/agent.py", t.calls[0])                       # the teacher saw her code map
        self.assertEqual(selfreview.recent_reviews(self.learning)[-1]["started"], rec["started"])

    def test_the_log_is_read_from_where_the_last_review_stopped(self) -> None:
        self.review(Teacher([PROBLEM]), start=False)
        body, info = selfreview.evidence(self.learning, self.root)
        self.assertNotIn("老师给的步骤她解析不了", body)
        with self.log.open("a", encoding="utf-8") as f:
            f.write("18:30:00 WARNING [agent] unusable plan (attempt 1): {}\n")
        body, _ = selfreview.evidence(self.learning, self.root)
        self.assertIn("老师给的步骤她解析不了：1 次", body)

    def test_nothing_twice_and_nothing_while_busy(self) -> None:
        self.review(Teacher([PROBLEM]))
        rec = self.review(Teacher([PROBLEM]), hours=24)
        self.assertEqual((rec["started"], rec["skipped"]), ("", "找到的问题最近都修过了"))
        rec = selfreview.review(Teacher([dict(PROBLEM, title="别的问题")]), learning_dir=self.learning, root=self.root,
                                projects_root=self.projects, start_project=lambda s: "", busy=lambda: True, hours=24)
        self.assertEqual(rec["skipped"], "手上还有项目在做")
        rec = self.review(Teacher([dict(PROBLEM, title="第三个问题")]), hours=24, start=False)
        self.assertEqual((rec["started"], len(rec["problems"])), ("", 1))

    def test_a_quiet_stretch(self) -> None:
        self.review(Teacher([PROBLEM]), start=False)            # moves the saved place to the end
        time.sleep(1.1)
        t = Teacher([PROBLEM])
        rec = self.review(t, grow=False)
        self.assertEqual((rec["skipped"], t.calls), ("最近没发现问题", []))
        # by default she still wants to get better: something toward her own ambition
        grow = dict(PROBLEM, title="回答再快一点", kind="grow")
        t = Teacher([grow])
        rec = self.review(t)
        self.assertEqual((rec["title"], rec["kind"]), ("回答再快一点", "grow"))
        self.assertIn("我的目标：成为主人离不开的全栈助理", t.calls[0])
        self.assertIn("给自己找了个能变得更好的地方", selfreview.started_notice(rec))
        self.assertEqual(drive.load(self.learning)["focus"], "回答再快一点")

    def test_the_owners_frustration_comes_first(self) -> None:
        drive.feel("你真笨", {"q": "你的制造商是谁", "a": "我是预训练出来的模型"}, self.learning)
        body, info = selfreview.evidence(self.learning, self.root)
        pains = body.split("主人不满意的地方")[1].split("上次没修好")[0]
        self.assertIn("主人说「你真笨」——因为她答「我是预训练出来的模型」", pains)
        rec = self.review(Teacher([PROBLEM]))
        self.assertEqual(rec["pains"], 1)
        self.assertIn("你有一次对我不太满意", selfreview.started_notice(rec))

    def test_what_she_failed_at_she_tries_again_another_way(self) -> None:
        rec = self.review(Teacher([PROBLEM]))
        drive.record_outcome({"slug": rec["started"], "goal": "修好自己：问她版本号答不上来", "status": "gave_up",
                              "summary": "找不到版本信息在哪"}, self.learning)
        d = drive.load(self.learning)
        d["grudges"][0]["last_try"] = "2000-01-01"            # a day has passed
        drive.save(d, self.learning)
        rec = self.review(Teacher([PROBLEM]), hours=24)
        self.assertTrue(rec["started"])                        # not skipped as "recently tried"
        brief = (self.projects / rec["started"] / "brief.md").read_text(encoding="utf-8")
        self.assertIn("这是第 2 次修这个问题。上次没成：找不到版本信息在哪", brief)

    def test_a_monthly_cap(self) -> None:
        rec = self.review(Teacher([PROBLEM]))
        st = project.load_state(self.projects / rec["started"])
        st["spent"] = selfreview.MONTH_YUAN
        project.save_state(self.projects / rec["started"], st)
        t = Teacher([dict(PROBLEM, title="别的")])
        rec = self.review(t, hours=24)
        self.assertIn("上限", rec["skipped"])
        self.assertEqual(t.calls, [])


if __name__ == "__main__":
    unittest.main()
