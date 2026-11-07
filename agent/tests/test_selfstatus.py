"""She answers "你现在什么版本" / "汇报一下你的进度" from the board (learning.selfstatus)."""
from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import selfprompt, selfstatus, weakspots  # noqa: E402

CONF = """# LoRA v5
[Service]
Environment=LLM_MODEL_DIR=/userdata/models/Qwen3.5-4B-monday-v5
ExecStart=
ExecStart=/userdata/agent/bin/llm_daemon /userdata/models/Qwen3.5-4B-monday-v5/Qwen3.5-4B.rknn 768 64 0xff
"""


class SelfStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.root, self.learning, self.projects = base / "agent", base / "learning", base / "projects"
        for d in (self.root / "learning", self.learning, self.projects):
            d.mkdir(parents=True)
        self.conf = base / "lora.conf"
        self.conf.write_text(CONF, encoding="utf-8")
        code = self.root / "learning" / "agent.py"
        code.write_text("x = 1\n", encoding="utf-8")
        self.t = time.mktime(datetime.datetime(2026, 11, 7, 15, 28).timetuple())
        os.utime(code, (self.t, self.t))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def version(self) -> str:
        return selfstatus.version_say(self.root, self.conf, self.learning)

    def test_version_says_model_code_time_and_rules(self) -> None:
        marker = self.root / "DEPLOYED_COMMIT"
        marker.write_text("64ecf93\n", encoding="utf-8")
        os.utime(marker, (self.t + 5, self.t + 5))
        say = self.version()
        self.assertIn("第五版", say)
        self.assertIn("千问三点五的四B模型", say)
        self.assertIn("十一月七号下午三点二十八分", say)
        self.assertIn("提交号 64ecf93", say)
        self.assertIn("还没改过", say)
        selfprompt.save({"agent": ["结论里不要提命令名"]}, "x", learning_dir=self.learning)
        self.assertIn("改到第一版", self.version())

    def test_a_stale_commit_is_not_claimed(self) -> None:
        marker = self.root / "DEPLOYED_COMMIT"
        marker.write_text("0d8f7e4\n", encoding="utf-8")
        os.utime(marker, (self.t - 86400, self.t - 86400))   # written a day before the code changed
        self.assertNotIn("0d8f7e4", self.version())
        self.assertNotIn("提交号", self.version())

    def test_no_drop_in_is_said_plainly(self) -> None:
        self.assertIn("读不到", selfstatus.version_say(self.root, self.conf.with_name("none"), self.learning))

    def test_progress_says_project_classes_weak_spots_and_tuning(self) -> None:
        pdir = self.projects / "p1"
        pdir.mkdir()
        (pdir / "state.json").write_text(json.dumps({"slug": "p1", "goal": "写一个 Python 小工具", "status": "done",
                                                     "steps": 25}, ensure_ascii=False), encoding="utf-8")
        weakspots.add(self.learning, "agent", "最占内存的进程属于哪个服务", "答成了不是 r1- 的服务", "eval")
        (self.learning / "selftune.jsonl").write_text('{"decision": "dropped"}\n{"decision": "aborted"}\n', encoding="utf-8")
        say = selfstatus.progress_say(self.learning, projects_root=self.projects, enabled=lambda u: False,
                                      active=lambda s: False)
        self.assertIn("上一个项目「写一个 Python 小工具」做完了", say)
        self.assertIn("每天的课现在是停着的", say)
        self.assertIn("还有一个短板没补上", say)
        self.assertIn("自我优化试了二轮，还没有一轮留下新规矩", say)
        busy = selfstatus.progress_say(self.learning, projects_root=self.projects, doing="查 196 的磁盘",
                                       enabled=lambda u: True, active=lambda s: False)
        self.assertTrue(busy.startswith("我手上正在办：查 196 的磁盘"))
        self.assertIn("课照常在上", busy)

    def test_which_questions_it_answers(self) -> None:
        mine = ["现在的版本是什么？", "你现在什么版本", "查一下你自己（星期一）当前的版本号", "版本号多少",
                "汇报一下你当前各项任务的进度。", "你最近在忙什么", "说说你的进度"]
        not_mine = ["继续在 196 上查刚才那个程序的版本号。", "Python 现在什么版本", "回退到上一个版本", "你的系统版本是多少",
                    "最新版本的苹果手机多少钱", "196 上的任务进度", "你知道他在做什么吗", "你几岁了", "项目做得怎么样"]
        for t in mine:
            self.assertIsNotNone(selfstatus.status_command(t), t)
        for t in not_mine:
            self.assertIsNone(selfstatus.status_command(t), t)
        both = selfstatus.status_command("说说你的版本和进度", root=self.root, conf=self.conf, learning_dir=self.learning,
                                         projects_root=self.projects, enabled=lambda u: False, active=lambda s: False)()
        self.assertIn("第五版", both)
        self.assertIn("课现在是停着的", both)


if __name__ == "__main__":
    unittest.main()
