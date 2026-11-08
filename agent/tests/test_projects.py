"""Projects (learning.project), their sandbox (learning.projbox), the internet-only proxy (learning.egress_proxy)
and automatic delivery with rollback (learning.deliver)."""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import deliver, egress_proxy, project, projbox  # noqa: E402


class ProxyTests(unittest.TestCase):
    def test_only_public_addresses(self) -> None:
        self.assertTrue(egress_proxy.public_ip("93.184.216.34"))
        self.assertTrue(egress_proxy.public_ip("198.18.0.7"))          # the LAN DNS's fake-ip for public names
        for ip in ("192.168.100.4", "10.0.0.177", "127.0.0.1", "169.254.1.1", "::1", "0.0.0.0"):
            self.assertFalse(egress_proxy.public_ip(ip), ip)

    def test_resolve_refuses_lan_and_odd_ports(self) -> None:
        def fake(host, port, type=0):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ({"pypi.org": "151.101.0.223", "gpu": "192.168.100.4"}[host], port))]
        with mock.patch.object(socket, "getaddrinfo", fake):
            self.assertEqual(egress_proxy.resolve_public("pypi.org", 443), ("151.101.0.223", ""))
            self.assertEqual(egress_proxy.resolve_public("gpu", 443)[0], "")
            self.assertEqual(egress_proxy.resolve_public("pypi.org", 22)[0], "")

    def test_request_heads(self) -> None:
        self.assertEqual(egress_proxy.parse_head(b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org\r\n\r\n")[:3],
                         ("CONNECT", "pypi.org", 443))
        m, host, port, up = egress_proxy.parse_head(b"GET http://example.com/a?b=1 HTTP/1.1\r\nHost: example.com\r\n"
                                                    b"Proxy-Connection: keep-alive\r\n\r\n")
        self.assertEqual((m, host, port), ("GET", "example.com", 80))
        self.assertTrue(up.startswith(b"GET /a?b=1 HTTP/1.1\r\n"))
        self.assertNotIn(b"Proxy-Connection", up)
        with self.assertRaises(ValueError):
            egress_proxy.parse_head(b"GET /relative HTTP/1.1\r\n\r\n")


class SandboxTests(unittest.TestCase):
    def test_layout(self) -> None:
        a = projbox.argv(Path("/userdata/projects/p1"), "make test")
        s = " ".join(a)
        self.assertIn("--ro-bind / /", s)
        self.assertIn("--bind /userdata/projects/p1/work /home/work", s)
        self.assertIn("--unshare-all", a)                               # no network, own pids/users
        self.assertNotIn("--share-net", a)
        self.assertIn("https_proxy", a)
        self.assertEqual(a[-1], "make test")
        self.assertEqual(a[a.index("--chdir") + 1], "/home/work")


class FakeWorld:
    """A planner that follows a script; a sandbox that runs nothing."""

    def __init__(self, script: list[dict], test_code: int = 0) -> None:
        self.script, self.test_code, self.ran, self.delivered = list(script), test_code, [], []
        self.user_msgs: list[str] = []

    def ctx(self, spent: float = 0.0, peak=lambda: False) -> project.Ctx:
        def plan(system, user):
            self.user_msgs.append(user)
            return json.dumps(self.script.pop(0) if self.script else {"tool": "give_up", "why": "脚本走完了"}, ensure_ascii=False)

        def sandbox(pdir, cmd, timeout):
            self.ran.append(cmd)
            return (self.test_code, "tests ok") if cmd == "python3 test_hello.py" else (0, f"ran {cmd}")
        return project.Ctx(plan=plan, consult=lambda s, q: "用 argparse。", sandbox=sandbox,
                           search=lambda q: (True, "1. result"), fetch=lambda u, s: (True, "page"),
                           deliver=lambda files, why, slug: self.delivered.append(files) or {"ok": True, "stage": "done"},
                           spent=lambda slug: spent, peak=peak, sleep=lambda s: None)


class ProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.slug = project.create("写一个 hello 程序", root=self.root)
        self.pdir = self.root / self.slug

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_project_runs_to_done_only_after_its_test_passes(self) -> None:
        w = FakeWorld([
            {"tool": "note", "text": "计划：写 hello.py 和测试"},
            {"tool": "done", "summary": "做完了"},                                # no test yet: refused
            {"tool": "write", "path": "hello.py", "content": "print('hello')\n"},
            {"tool": "write", "path": "hello.py", "content": "print('hello, world')\n"},
            {"tool": "search", "query": "python argparse"},
            {"tool": "ask_ai", "question": "怎么写测试"},
            {"tool": "set_test", "cmd": "python3 test_hello.py"},
            {"tool": "done", "summary": "hello 程序写好了"},
        ])
        st = project.run(self.pdir, w.ctx())
        self.assertEqual((st["status"], st["steps"]), ("done", 8))
        self.assertEqual((self.pdir / "work" / "hello.py").read_text(encoding="utf-8"), "print('hello, world')\n")
        # versions/<n>/ keeps what a file was just before step n changed it
        self.assertEqual((self.pdir / "versions" / "4" / "hello.py").read_text(encoding="utf-8"), "print('hello')\n")
        log = project.steps(self.pdir)
        self.assertEqual(log[1]["status"], "fail")                             # done before set_test
        self.assertIn("计划", (self.pdir / "notes.md").read_text(encoding="utf-8"))
        self.assertIn("计划：写 hello.py", w.user_msgs[-1])                       # notes reach the planner
        self.assertIn("python3 test_hello.py", w.ran)

    def test_failed_acceptance_keeps_going(self) -> None:
        w = FakeWorld([{"tool": "set_test", "cmd": "python3 test_hello.py"}, {"tool": "done", "summary": "x"}], test_code=1)
        st = project.run(self.pdir, w.ctx())
        self.assertEqual(st["status"], "gave_up")                              # script ran out -> give_up
        self.assertIn("验收没通过", project.steps(self.pdir)[1]["result"])

    def test_empty_and_cut_off_replies(self) -> None:
        replies = iter(["", '{"thought":"写文件","tool":"write","path":"big.py","content":"print(1)\\nprint(',
                        json.dumps({"tool": "write", "path": "big.py", "content": "a = 1\n"}),
                        json.dumps({"tool": "append", "path": "big.py", "content": "b = 2\n"}),
                        json.dumps({"tool": "give_up", "why": "x"})])
        w = FakeWorld([])
        ctx = w.ctx()
        ctx.plan = lambda system, user: next(replies)
        st = project.run(self.pdir, ctx)
        log = project.steps(self.pdir)
        self.assertEqual(st["steps"], 4)                                       # the empty reply was not a step
        self.assertEqual(log[0]["status"], "fail")
        self.assertIn("append", log[0]["result"])                              # told how to avoid it
        self.assertIn("print(", log[0]["raw_tail"])
        self.assertEqual((self.pdir / "work" / "big.py").read_text(encoding="utf-8"), "a = 1\nb = 2\n")

    def test_writes_stay_inside_the_workspace(self) -> None:
        w = FakeWorld([{"tool": "write", "path": "../../etc/passwd", "content": "x"},
                       {"tool": "write", "path": "/home/work/ok.txt", "content": "y"}])
        project.run(self.pdir, w.ctx())
        self.assertEqual(project.steps(self.pdir)[0]["status"], "fail")
        self.assertTrue((self.pdir / "work" / "ok.txt").is_file())

    def test_budget_stops_it(self) -> None:
        st = project.run(self.pdir, FakeWorld([{"tool": "note", "text": "x"}]).ctx(spent=10.5))
        self.assertEqual((st["status"], st["steps"]), ("over_budget", 0))

    def test_peak_hours_wait_without_calling_the_teacher(self) -> None:
        calls = iter([True, True, False])
        w = FakeWorld([{"tool": "give_up", "why": "测试"}])
        st = project.run(self.pdir, w.ctx(peak=lambda: next(calls)))
        self.assertEqual((st["status"], len(w.user_msgs)), ("gave_up", 1))

    def test_deliver_goes_through_the_delivery_gate(self) -> None:
        (self.pdir / "work" / "skill.py").write_text("x = 1\n", encoding="utf-8")
        w = FakeWorld([{"tool": "deliver", "files": [{"src": "skill.py", "dst": "skills_extra/skill.py"}], "why": "新技能"},
                       {"tool": "deliver", "files": [{"src": "nope.py", "dst": "x.py"}], "why": "x"}])
        project.run(self.pdir, w.ctx())
        self.assertEqual(w.delivered[0][0]["dst"], "skills_extra/skill.py")
        self.assertEqual(project.steps(self.pdir)[1]["status"], "fail")

    def test_retrospective_and_notice(self) -> None:
        st = project.run(self.pdir, FakeWorld([{"tool": "give_up", "why": "没有 pip"}]).ctx())
        ctx = project.Ctx(plan=lambda s, u: json.dumps({"learned": ["先查板子上有什么工具"], "missing": ["不会装 pip"], "say": "没做成，缺 pip。"}, ensure_ascii=False),
                          consult=None, sandbox=None, search=None, fetch=None, deliver=None, spent=lambda s: 0)
        learning = self.root / "learning"
        out = project.retrospective(self.pdir, st, ctx, learning)
        self.assertEqual(out["missing"], ["不会装 pip"])
        self.assertIn("不会装 pip", project.review_lines(learning))
        project.notify("主人，项目没做成。", self.root / "notices")
        project.notify("主人，要你批准。", self.root / "notices", ask=True)
        names = sorted(p.name for p in (self.root / "notices").glob("*.txt"))
        self.assertEqual([n.rsplit("-", 1)[-1] for n in names], ["project.txt", "ask.txt"] if names[0].endswith("project.txt") else ["ask.txt", "project.txt"])


class VoiceCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.started: list[str] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def cmd(self, text, active=lambda slug: True):
        return project.project_command(text, root=self.root, start=lambda slug: self.started.append(slug) or "",
                                       active=active, is_peak=lambda: False)

    def test_start_status_and_one_at_a_time(self) -> None:
        self.assertIsNone(self.cmd("今天天气怎么样"))
        self.assertIsNone(self.cmd("开个项目"))                                   # no goal
        say = self.cmd("星期一，开个项目：写一个把日志压缩归档的小工具")()
        self.assertTrue(say.startswith("好，我开个项目：写一个把日志压缩归档的小工具"))
        self.assertEqual(project.load_state(self.root / self.started[0])["goal"], "写一个把日志压缩归档的小工具")
        self.assertIn("还有一个项目没做完", self.cmd("再开个项目：做个天气播报")())
        self.assertIn("正在做", self.cmd("项目做得怎么样了")())

    def test_a_dead_project_does_not_block_the_next(self) -> None:
        self.cmd("开个项目：第一个目标是这个")()
        say = self.cmd("开个项目：第二个目标是那个", active=lambda slug: False)()
        self.assertTrue(say.startswith("好，"))
        self.assertEqual(project.load_state(self.root / self.started[0])["status"], "stopped")


class RepairTests(unittest.TestCase):
    """Repair projects (kind "repair", started by learning.selfreview) and proposals the owner approves."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root, self.agent = base / "projects", base / "agent"
        for rel, text in (("learning/agent.py", "A = 1\n"), ("orchestrator/main.py", "M = 1\n"), ("learning/deliver.py", "D\n")):
            (self.agent / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.agent / rel).write_text(text, encoding="utf-8")
        self.slug = project.create("修好自己：老师的回复解析不了", self.root, kind="repair",
                                   brief="现象：老师分两段发 JSON，她说「老师给的步骤我没看懂」")
        self.pdir = self.root / self.slug
        work = self.pdir / "work" / "agent"
        for rel, text in (("learning/agent.py", "A = 2\n"), ("orchestrator/main.py", "M = 2\n"), ("learning/deliver.py", "D2\n")):
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            (work / rel).write_text(text, encoding="utf-8")
        self.systems: list[str] = []
        self.patch = mock.patch.object(project, "AGENT_ROOT", self.agent)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()
        self._tmp.cleanup()

    def run_script(self, script: list[dict], slug: str | None = None) -> dict:
        w = FakeWorld(script)
        ctx = w.ctx()
        plan = ctx.plan

        def watching(system, user):
            self.systems.append(system)
            self.user = user
            return plan(system, user)
        ctx.plan = watching
        return project.run(self.root / (slug or self.slug), ctx)

    def propose_step(self, *dsts: str) -> dict:
        return {"tool": "propose", "why": "合并分两段的 JSON", "say": "老师的回复分两段时我读不懂，改好了。",
                "files": [{"src": f"agent/{d}", "dst": d} for d in dsts]}

    def test_a_repair_project_gets_the_repair_rules_and_the_problem(self) -> None:
        self.run_script([{"tool": "give_up", "why": "x"}])
        self.assertIn("这次的项目是修你自己", self.systems[0])
        self.assertIn("老师给的步骤我没看懂", self.user)
        other = project.create("写个小工具", self.root)
        self.run_script([{"tool": "give_up", "why": "x"}], other)
        self.assertNotIn("修你自己", self.systems[-1])

    def test_a_protected_change_is_handed_to_the_owner(self) -> None:
        st = self.run_script([self.propose_step("learning/agent.py", "orchestrator/main.py")])
        self.assertEqual(st["status"], "proposed")
        prop = project.load_proposal(self.pdir)
        self.assertEqual((prop["status"], [f["dst"] for f in prop["files"]]), ("pending", ["learning/agent.py", "orchestrator/main.py"]))
        self.assertEqual((self.pdir / prop["files"][0]["src"]).read_text(encoding="utf-8"), "A = 2\n")
        diff = (self.pdir / "proposal.diff").read_text(encoding="utf-8")
        self.assertIn("-A = 1", diff)
        self.assertIn("+A = 2", diff)
        self.assertEqual((self.agent / "learning" / "agent.py").read_text(encoding="utf-8"), "A = 1\n")  # nothing applied

    def test_what_even_the_owner_cannot_approve_and_what_needs_no_approval(self) -> None:
        self.run_script([self.propose_step("learning/deliver.py"), self.propose_step("orchestrator/main.py")])
        log = project.steps(self.pdir)
        self.assertIn("不能由你来改", log[0]["result"])
        self.assertIn("直接用 deliver", log[1]["result"])
        other = project.create("写个小工具", self.root)
        self.run_script([self.propose_step("learning/agent.py")], other)
        self.assertIn("只有修自己的项目", project.steps(self.root / other)[0]["result"])

    def test_the_owner_says_yes_or_no(self) -> None:
        self.run_script([self.propose_step("learning/agent.py")])
        started = []
        cmd = lambda text: project.project_command(text, root=self.root, start=lambda slug: started.append(slug) or "",
                                                   active=lambda s: False, is_peak=lambda: False)
        self.assertIsNone(cmd("可以"))                           # a bare "可以" answers agent steps, not proposals
        self.assertIn("等你批准", cmd("项目做得怎么样")())
        self.assertIn("这就上线", cmd("批准修改。")())
        self.assertEqual((started, project.load_state(self.pdir)["status"]), ([self.slug], "applying"))
        delivered = []
        ctx = FakeWorld([]).ctx()
        ctx.deliver = lambda files, why, slug, allow=(): delivered.append((files, allow)) or {"ok": True, "stage": "done"}
        project.apply_proposal(self.pdir, ctx)
        files, allow = delivered[0]
        self.assertEqual(allow, ("learning/agent.py",))
        self.assertEqual(files[0]["src"].read_text(encoding="utf-8"), "A = 2\n")
        self.assertEqual(project.load_state(self.pdir)["status"], "done")
        self.assertIsNone(cmd("批准修改"))                        # nothing is waiting any more

    def test_no_closes_it(self) -> None:
        self.run_script([self.propose_step("learning/agent.py")])
        say = project.project_command("不用改了", root=self.root, start=lambda slug: "", active=lambda s: False)()
        self.assertIn("不上线", say)
        self.assertEqual((project.load_state(self.pdir)["status"], project.load_proposal(self.pdir)["status"]),
                         ("rejected", "rejected"))

    def test_her_own_words_when_a_repair_ends(self) -> None:
        learning = self.root / "learning"
        st = {"slug": "s", "goal": "修好自己：回答太慢", "status": "gave_up", "summary": "找不到慢在哪"}
        self.assertIn("我不服气，下次换个办法再来", project.repair_notice(st, {"say": "", "missing": []}, learning))
        project.repair_notice(st, {}, learning)
        self.assertIn("试了三次还是没修好，靠我自己不行了", project.repair_notice(st, {}, learning))
        st["status"] = "done"
        self.assertEqual(project.repair_notice(st, {"say": "主人，回答慢的问题修好了。"}, learning), "主人，回答慢的问题修好了。")

    def test_check_yourself_starts_a_review(self) -> None:
        from learning import selfreview
        with mock.patch.object(selfreview, "start_unit", lambda: ""):
            say = project.project_command("你去检查一下你自己", root=self.root, active=lambda s: False)()
        self.assertTrue(say.startswith("好，我去翻翻最近的对话和日志"))
        for text in ("看看自己哪里傻", "自己修修自己", "修一下你自己"):
            self.assertIsNotNone(project._REVIEW.search(text), text)
        self.assertIsNone(project._REVIEW.search("检查一下你自己的服务状态"))      # that one is a task


class DeliverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "agent"
        (self.root / "orchestrator").mkdir(parents=True)
        (self.root / "orchestrator" / "feature.py").write_text("OLD\n", encoding="utf-8")
        self.src = Path(self._tmp.name) / "work"
        self.src.mkdir()
        (self.src / "feature.py").write_text("NEW\n", encoding="utf-8")
        (self.src / "extra.py").write_text("EXTRA\n", encoding="utf-8")
        self.learning = Path(self._tmp.name) / "learning"
        self.learning.mkdir()
        self.saved: dict[str, str] = {}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def snapshot(self, label, reason):
        self.saved = {p.relative_to(self.root).as_posix(): p.read_text(encoding="utf-8") for p in self.root.rglob("*.py")}
        return "snap-1"

    def restore(self, sid):
        for rel, text in self.saved.items():
            (self.root / rel).write_text(text, encoding="utf-8")

    def go(self, files, *, tests_after=frozenset(), exam=(30, 30), quiet=True, allow=()):
        runs = iter([{"tests.old"}, set(tests_after) | {"tests.old"}])
        scores = iter(exam)
        restarted = []
        res = deliver.deliver(files, "新功能", project="p1", learning_dir=self.learning, root=self.root,
                              exam_score=lambda: next(scores), wait_quiet=lambda: quiet, tests=lambda: next(runs),
                              snapshot=self.snapshot, restore=self.restore,
                              restart_units=lambda units: restarted.append(list(units)) or "", allow=allow)
        return res, restarted

    def files(self):
        return [{"src": self.src / "feature.py", "dst": "orchestrator/feature.py"},
                {"src": self.src / "extra.py", "dst": "orchestrator/extra.py"}]

    def test_good_change_stays_and_restarts_what_it_touched(self) -> None:
        res, restarted = self.go(self.files())
        self.assertTrue(res["ok"])
        self.assertEqual(restarted, [["r1-orchestrator"]])
        self.assertEqual((self.root / "orchestrator" / "feature.py").read_text(encoding="utf-8"), "NEW\n")

    def test_new_test_failure_rolls_everything_back(self) -> None:
        res, _ = self.go(self.files(), tests_after={"tests.test_x.T.test_y"})
        self.assertFalse(res["ok"])
        self.assertTrue(res["rolled_back"])
        self.assertEqual((self.root / "orchestrator" / "feature.py").read_text(encoding="utf-8"), "OLD\n")
        self.assertFalse((self.root / "orchestrator" / "extra.py").exists())     # new files removed too

    def test_exam_drop_rolls_back(self) -> None:
        res, _ = self.go(self.files(), exam=(30, 25))
        self.assertEqual((res["ok"], res["stage"]), (False, "exam"))
        self.assertEqual((self.root / "orchestrator" / "feature.py").read_text(encoding="utf-8"), "OLD\n")

    def test_protected_files_are_refused_before_anything_happens(self) -> None:
        for dst in ("learning/agent.py", "learning/exam.py", "config/agent.yaml", "ops/snapshot.py", "../etc/x", "run/x"):
            res, _ = self.go([{"src": self.src / "feature.py", "dst": dst}])
            self.assertEqual((res["ok"], res["stage"]), (False, "check"), dst)

    def test_owner_approved_proposals_may_touch_only_the_proposable_files(self) -> None:
        (self.root / "learning").mkdir()
        (self.root / "learning" / "agent.py").write_text("OLD\n", encoding="utf-8")
        res, _ = self.go([{"src": self.src / "feature.py", "dst": "learning/agent.py"}], allow=("learning/agent.py",))
        self.assertTrue(res["ok"])
        self.assertEqual((self.root / "learning" / "agent.py").read_text(encoding="utf-8"), "NEW\n")
        for dst in ("learning/deliver.py", "learning/projbox.py", "ops/snapshot.py", "config/agent.yaml"):
            res, _ = self.go([{"src": self.src / "feature.py", "dst": dst}], allow=(dst,))
            self.assertEqual((res["ok"], res["stage"]), (False, "check"), dst)

    def test_waits_for_the_owner_to_be_quiet(self) -> None:
        res, restarted = self.go(self.files(), quiet=False)
        self.assertEqual((res["ok"], res["stage"], restarted), (False, "wait", []))
        self.assertEqual((self.root / "orchestrator" / "feature.py").read_text(encoding="utf-8"), "OLD\n")

    def test_log(self) -> None:
        self.go(self.files())
        rows = [json.loads(x) for x in (self.learning / deliver.LOG_FILE).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["snapshot"], "snap-1")


if __name__ == "__main__":
    unittest.main()
