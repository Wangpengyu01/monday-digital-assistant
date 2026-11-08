from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning.agent import PLAN_SYSTEM, VERIFY_SYSTEM, Agent  # noqa: E402
from learning.grants import GrantStore  # noqa: E402
from learning.skills import SkillStore  # noqa: E402


class PlanTeacher:
    """Replays fixed steps; records the plan prompts it was given."""

    available = True

    def __init__(self, steps: list[dict]) -> None:
        self.steps = list(steps)
        self.plan_prompts: list[str] = []

    def complete(self, system: str, user: str, **kw) -> str:
        if system == PLAN_SYSTEM:
            self.plan_prompts.append(user)
            return json.dumps({"thought": "", "step": self.steps.pop(0)}, ensure_ascii=False)
        assert system == VERIFY_SYSTEM
        return json.dumps({"done": True, "why": "有图", "task": "画图", "skill_steps": []}, ensure_ascii=False)


class DrawRequestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.skills = SkillStore(Path(self._tmp.name) / "memory.db")
        self.grants = GrantStore(Path(self._tmp.name) / "memory.db")
        self.said: list[str] = []

    async def asyncTearDown(self) -> None:
        self.skills.close()
        self.grants.close()
        self._tmp.cleanup()

    async def say(self, text: str) -> None:
        self.said.append(text)

    async def approve(self, step: dict, why: str, risk: str) -> str:
        return "once"

    def runner(self, step: dict) -> tuple[int, str]:
        return 0, "/userdata/agent/run/plots/taylor_sin.png"

    def agent(self, teacher) -> Agent:
        return Agent(teacher, self.skills, approve=self.approve, say=self.say, runner=self.runner,
                     grants=self.grants, ssh_hosts=["196（gp@192.168.100.196）"], snapshot=lambda a, b: "snap")

    async def test_plan_prompt_carries_recent_context(self) -> None:
        """The plan prompt must carry the recent conversation, so "上一个" can be resolved."""
        teacher = PlanTeacher([{"tool": "done", "success": True, "say": "好了"}])
        agent = self.agent(teacher)
        task = ("画图：给我画出上一个泰勒公式程序的图像\n\n最近对话：\n"
                "主人：写个 Python 演示正弦函数逐阶展开逼近真实值的泰勒公式\n星期一：写好了 /userdata/agent/run/taylor_sin.py")
        await agent.run(task)
        self.assertTrue(teacher.plan_prompts)
        self.assertIn("泰勒", teacher.plan_prompts[0])
        self.assertIn("最近对话", teacher.plan_prompts[0])

    async def test_plan_system_forbids_asking_which_machine_for_draw(self) -> None:
        """PLAN_SYSTEM must tell the model: for draw requests, find the program yourself; can write PNG but cannot send images."""
        self.assertIn("画图", PLAN_SYSTEM)
        self.assertIn("发不出", PLAN_SYSTEM)
        self.assertNotIn("不要挨个去猜", PLAN_SYSTEM)


class _Orch:
    """Minimal stand-in to check the task text handed to the agent carries recent turns."""


class DrawTaskTextTests(unittest.TestCase):
    def test_recent_turns_are_appended_to_draw_task(self) -> None:
        from orchestrator.main import _task_with_recent

        recent = [
            {"who": "user", "q": "写个 Python 演示正弦函数逐阶展开逼近真实值的泰勒公式", "a": "写好了 /userdata/agent/run/taylor_sin.py"},
            
        ]
        text = _task_with_recent("给我画出上一个泰勒公式程序的图像", recent)
        self.assertIn("给我画出上一个泰勒公式程序的图像", text)
        self.assertIn("泰勒", text.split("最近对话")[-1])
        self.assertIn("taylor_sin.py", text)

    def test_empty_recent_leaves_task_alone(self) -> None:
        from orchestrator.main import _task_with_recent

        self.assertEqual(_task_with_recent("看一下磁盘", []), "看一下磁盘")


if __name__ == "__main__":
    unittest.main()
