"""发散 (learning.associate): her association graph, nightly wandering, and associations in live answers."""
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

from learning import associate, selfstatus  # noqa: E402

GRAPH = {"nodes": [{"name": "HomePod", "kind": "设备"}, {"name": "睡不好", "kind": "习惯"}, {"name": "白噪音", "kind": "事物"},
                   {"name": "AirPlay", "kind": "事物"}],
         "edges": [{"a": "HomePod", "b": "AirPlay", "rel": "用途", "why": "推流放音乐要走 AirPlay"},
                   {"a": "HomePod", "b": "白噪音", "rel": "用途", "why": "音箱能放白噪音助眠"},
                   {"a": "白噪音", "b": "睡不好", "rel": "因果", "why": "稳定的声音能盖住打断睡眠的杂音"},
                   {"a": "x", "b": "睡不好", "rel": "瞎编", "why": "不在关系表里"}]}
BETTER = ("由找音箱想到推流，因为放歌得先连上它；再由放歌想到睡前放点白噪音，因为主人最近睡不好，"
          "稳定的声音能盖住杂音；最后想到可以让我睡前十一点自动把白噪音推到音箱上，主人不用自己动手。")


class Teacher:
    def __init__(self, graph: dict, verdicts: list[dict]) -> None:
        self.graph, self.verdicts = graph, list(verdicts)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, **kw):
        self.calls.append((system, user))
        if system == associate.GRAPH_SYSTEM:
            return json.dumps(self.graph, ensure_ascii=False)
        assert system == associate.JUDGE_SYSTEM
        return '{"thought": "看看"}\n' + json.dumps(self.verdicts.pop(0), ensure_ascii=False)


def verdict(**kw) -> dict:
    return {"score": 4, "comment": "从找设备跳到了助眠", "better": BETTER, "idea": "主人睡不好的话，睡前可以让我往 HomePod 推一段白噪音。",
            "triggers": ["睡不好", "失眠", "HomePod"], "useful": True, **kw}


class AssociateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        (self.dir / "turns").mkdir()
        now = dt.datetime.now()
        rows = [{"time": (now - dt.timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S"), "q": q, "a": "好"}
                for m, q in ((30, "帮我在局域网里找 HomePod"), (20, "我最近老是睡不好"))]
        (self.dir / "turns" / f"{now.date().isoformat()}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        self.asked: list[str] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def local(self, system, user):
        self.asked.append(user)
        return "由找 HomePod 想到查它的 IP，因为要先找到设备。想法：先确认在同一个网络。"

    def night(self, teacher, **kw):
        return associate.nightly(teacher, self.local, persona="你是星期一。", facts=["主人最近睡眠不好"],
                                 learning_dir=self.dir, rng=random.Random(1), **kw)

    def test_the_graph_is_built_from_the_days_talk(self) -> None:
        t = Teacher(GRAPH, [verdict()] * 6)
        res = self.night(t, n_seeds=0)
        self.assertEqual((res["turns"], res["new_nodes"], res["new_edges"]), (2, 4, 3))   # the made-up relation is dropped
        d = associate.load(self.dir)
        self.assertEqual([e["b"] for e in associate.neighbors(d, "HomePod")], ["AirPlay", "白噪音"])
        self.assertIn("主人最近睡眠不好", t.calls[0][1])
        again = self.night(Teacher(GRAPH, []), n_seeds=0)
        self.assertEqual(again["turns"], 0)                            # the same turns are not read twice

    def test_she_wanders_first_then_the_teacher_shows_a_bolder_chain(self) -> None:
        t = Teacher(GRAPH, [verdict(), verdict(useful=False, idea=""), verdict(score=2, better="→ 不能念")] + [verdict()] * 3)
        res = self.night(t, n_seeds=3)
        self.assertEqual(res["wandered"], 3)
        self.assertTrue(all(u.startswith("从这件事开始联想：") for u in self.asked))       # her own attempt comes first
        self.assertTrue(any("把两件看起来不相干的事连起来" in u for u in self.asked))       # one forced leap
        self.assertIn("她的联想：\n由找 HomePod 想到查它的 IP", t.calls[1][1])
        ideas = associate.load(self.dir)["ideas"]
        self.assertEqual(len(ideas), 1)                                   # useful once; the same idea is not kept twice
        samples = (self.dir / "distill" / "assoc.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(samples), 2)                                 # the arrow one is not speakable: not trained
        self.assertEqual(json.loads(samples[0])["messages"][-1]["content"], BETTER)

    def test_live_context_lights_up_neighbours_and_ideas(self) -> None:
        self.night(Teacher(GRAPH, [verdict()] * 6), n_seeds=1)
        today = dt.date.today().isoformat()
        ctx = associate.context_for("我昨晚又睡不好了", self.dir, today=today)
        self.assertIn("睡不好和白噪音（因果", ctx)
        self.assertIn("你之前琢磨到的：主人睡不好的话", ctx)
        self.assertIn("别硬扯", ctx)
        self.assertNotIn("琢磨到", associate.context_for("我昨晚又睡不好了", self.dir, today=today))   # resting now
        later = (dt.date.today() + dt.timedelta(days=associate.IDEA_REST_DAYS + 1)).isoformat()
        self.assertIn("琢磨到", associate.context_for("失眠怎么办", self.dir, today=later))
        self.assertEqual(associate.context_for("今天几号", self.dir), "")
        self.assertLessEqual(len(ctx.splitlines()[0]), associate.CONTEXT_CHARS)

    def test_she_can_say_what_she_came_to(self) -> None:
        self.assertIn("还没琢磨出什么", associate.say_ideas(self.dir))
        self.night(Teacher(GRAPH, [verdict()] * 6), n_seeds=1)
        self.assertTrue(associate.wants_ideas("你最近想到什么了"))
        say = selfstatus.status_command("星期一你最近有什么想法", learning_dir=self.dir)()
        self.assertTrue(say.startswith("我最近琢磨到一件事。主人睡不好的话"))

    def test_the_owner_stops_it(self) -> None:
        res = self.night(Teacher(GRAPH, [verdict()] * 6), n_seeds=3, idle=lambda: False)
        self.assertEqual((res["wandered"], res.get("stopped")), (0, True))


if __name__ == "__main__":
    unittest.main()
