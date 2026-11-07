"""What 星期一 can say about herself right away - which version she is and what she is busy with - read off
the board, not guessed by a model and not looked up step by step with commands.

    python3 -m learning.selfstatus            # both, the way she would say them

Asked "现在的版本是什么" she used to treat it as a general question for the teacher ("查程序名加 version 参数"),
and "汇报一下你各项任务的进度" became a shell task. status_command() catches those before the intent step.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
LEARNING = AGENT_ROOT / "run" / "learning"
LORA_CONF = Path("/etc/systemd/system/r1-llm-daemon.service.d/lora.conf")
CODE_DIRS = ("orchestrator", "learning", "llm", "asr", "tts", "ops", "ui", "devices", "daemon", "scripts")
CODE_EXT = {".py", ".sh", ".cpp", ".cc", ".h", ".html", ".js"}
SKIP_DIRS = {"__pycache__", "tests", "board_tests", "thirdparty", "third_party", "build", "bin", "internvl_sdk"}
CLASS_UNITS = ("r1-distill.timer", "r1-study.service")   # the nightly classes (stopped at the owner's word)
STALE_S = 120   # code newer than DEPLOYED_COMMIT by more than this: the recorded commit is not what runs

_DIGITS = "零一二三四五六七八九"


def cn(n: int) -> str:
    """0..99 in Chinese, the way it is said (十一, 二十八)."""
    if n < 10:
        return _DIGITS[n]
    tens, ones = divmod(n, 10)
    return ("" if tens == 1 else _DIGITS[tens]) + "十" + (_DIGITS[ones] if ones else "")


def count(n: int) -> str:
    """A count: 两个, not 二个."""
    return "两" if n == 2 else cn(min(n, 99))


def say_time(t: datetime.datetime) -> str:
    part = "凌晨" if t.hour < 6 else "上午" if t.hour < 12 else "中午" if t.hour < 13 else "下午" if t.hour < 18 else "晚上"
    h = t.hour if t.hour <= 12 else t.hour - 12
    return f"{cn(t.month)}月{cn(t.day)}号{part}{cn(h)}点" + (f"{cn(t.minute)}分" if t.minute else "")


# ---------------------------------------------------------------- version
def code_version(root: Path = AGENT_ROOT) -> dict:
    """{"commit": the deployed commit ("" if unknown or stale), "updated": newest code file time or None}."""
    newest = 0.0
    for d in CODE_DIRS:
        for dirpath, dirnames, files in os.walk(root / d):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS and not x.startswith(".")]
            for f in files:
                if os.path.splitext(f)[1] in CODE_EXT:
                    try:
                        newest = max(newest, os.stat(os.path.join(dirpath, f)).st_mtime)
                    except OSError:
                        pass
    commit = ""
    marker = root / "DEPLOYED_COMMIT"
    try:
        words = marker.read_text(encoding="utf-8").split()
        if words and marker.stat().st_mtime + STALE_S >= newest:
            commit = words[0][:12]
    except OSError:
        pass
    return {"commit": commit, "updated": datetime.datetime.fromtimestamp(newest) if newest else None}


def model_version(conf: Path = LORA_CONF) -> dict:
    """{"name": "Qwen3.5-4B-monday-v5", "base": "Qwen3.5-4B", "lora": 5} from the llm daemon's drop-in."""
    try:
        text = conf.read_text(encoding="utf-8")
    except OSError:
        return {"name": "", "base": "", "lora": 0}
    m = re.search(r"LLM_MODEL_DIR=(\S+)", text) or re.search(r"/userdata/models/([^/\s]+)", text)
    name = Path(m.group(1)).name if m else ""
    v = re.search(r"-v(\d+)$", name)
    return {"name": name, "base": name.split("-monday")[0], "lora": int(v.group(1)) if v else 0}


def rules_version(learning_dir: Path = LEARNING) -> int:
    from learning import selfprompt

    try:
        return int(selfprompt.load(learning_dir).get("version") or 0)
    except (OSError, ValueError, TypeError):
        return 0


def _base_say(base: str) -> str:
    m = re.match(r"(?i)qwen(\d+)(?:\.(\d+))?-(\d+)b", base or "")
    if not m:
        return base
    ver = cn(int(m.group(1))) + (f"点{cn(int(m.group(2)))}" if m.group(2) else "")
    return f"千问{ver}的{cn(int(m.group(3)))}B模型"


def version_say(root: Path = AGENT_ROOT, conf: Path = LORA_CONF, learning_dir: Path = LEARNING) -> str:
    model, code, rules = model_version(conf), code_version(root), rules_version(learning_dir)
    parts = []
    if model["lora"]:
        parts.append(f"我现在是第{cn(model['lora'])}版，底座是{_base_say(model['base'])}，上面加了第{cn(model['lora'])}版微调")
    elif model["name"]:
        parts.append(f"我现在的模型是 {model['name']}")
    else:
        parts.append("我的模型版本这会儿读不到")
    if code["updated"]:
        parts.append(f"代码是{say_time(code['updated'])}更新的那一版" + (f"，提交号 {code['commit']}" if code["commit"] else ""))
    parts.append(f"自己给自己定的说话规矩改到第{cn(rules)}版了" if rules else "自己给自己定的说话规矩还没改过")
    return "。".join(parts) + "。"


# ---------------------------------------------------------------- what she is busy with
def unit_enabled(unit: str) -> bool:
    try:
        return subprocess.run(["systemctl", "is-enabled", unit], capture_output=True, text=True,
                              timeout=5).stdout.strip() == "enabled"
    except (OSError, subprocess.SubprocessError):
        return False


def progress_say(learning_dir: Path = LEARNING, *, projects_root: Path | None = None, doing: str = "",
                 enabled: Callable[[str], bool] = unit_enabled, active: Callable[[str], bool] | None = None) -> str:
    from learning import project, selftune, weakspots

    parts = []
    if doing:
        parts.append(f"我手上正在办：{doing[:30]}")
    try:
        busy = project.current(projects_root, active or project.unit_active)
        if busy:
            parts.append(f"项目「{busy['goal'][:20]}」正在做，走了{busy.get('steps', 0)}步")
        else:
            done = project.list_projects(projects_root)
            if done:
                p = done[0]
                parts.append(f"上一个项目「{p['goal'][:20]}」{project.STATUS_SAY.get(p.get('status'), p.get('status', ''))}")
            elif not doing:
                parts.append("现在没有在做的项目")
    except OSError:
        pass
    parts.append("每天夜里的课照常在上" if any(enabled(u) for u in CLASS_UNITS) else "每天的课现在是停着的")
    open_spots = weakspots.load(learning_dir)["open"]
    if open_spots:
        example = re.split(r"[，,。？?；;]", open_spots[0]["task"])[0][:30]   # the first clause, never cut mid-word
        parts.append(f"还有{count(len(open_spots))}个短板没补上，比如{example}")
    else:
        parts.append("短板清单是空的")
    tunes = selftune.recent(learning_dir, 50)
    if tunes:
        kept = sum(1 for r in tunes if r.get("decision") == "kept")
        parts.append(f"自我优化试了{count(len(tunes))}轮，" + (f"留下了{count(kept)}次新规矩" if kept else "还没有一轮留下新规矩"))
    return "。".join(parts) + "。"


# ---------------------------------------------------------------- catching the question
_ME = re.compile(r"你|星期一|自己")
_NOW = re.compile(r"现在|当前|目前|眼下")
# a version of something else, or a request to change versions: not a question about her
_OTHER = re.compile(r"(?i)python|java|node|ubuntu|linux|系统|内核|固件|驱动|镜像|手机|电脑|服务器|浏览器|chrome|app|应用|"
                    r"\d+\.\d+|196|那个|它|他|她|别人")
_CHANGE = re.compile(r"回退|回滚|退回|换回|切回|切到|换成|换到|升级|降级|更新到|安装|下载|改成|改回|发布|上线")
_PROGRESS = re.compile(r"(你|星期一|自己).{0,10}(进度|进展|在忙什么|在忙啥|在忙些什么|在做什么|在干什么|在干嘛|在搞什么|忙什么|"
                       r"忙啥|学得怎么样|学到哪|近况)|各项任务|任务.{0,3}(进度|进展)|(汇报|报告|说说|讲讲).{0,4}(进度|进展|近况)")


def wants_version(text: str) -> bool:
    t = text or ""
    return ("版本" in t and not _OTHER.search(t) and not _CHANGE.search(t)
            and bool(_ME.search(t) or _NOW.search(t) or len(t) <= 8))


def wants_progress(text: str) -> bool:
    t = text or ""
    return bool(_PROGRESS.search(t)) and not _OTHER.search(t)


def status_command(text: str, *, doing: str = "", root: Path = AGENT_ROOT, conf: Path = LORA_CONF,
                   learning_dir: Path = LEARNING, projects_root: Path | None = None,
                   enabled: Callable[[str], bool] = unit_enabled,
                   active: Callable[[str], bool] | None = None) -> Callable[[], str] | None:
    """"你现在什么版本" / "汇报一下你的进度" -> a function returning what to say; anything else -> None."""
    t = re.sub(r"\s+", "", text or "")
    from learning import drive
    if drive.wants_drive(t):   # "你在努力什么 / 你最近有进步吗": her ambition, the owner's mood, wins and grudges
        return lambda: drive.say(learning_dir)
    version, progress = wants_version(t), wants_progress(t)
    if not (version or progress):
        return None

    def answer() -> str:
        out = []
        if version:
            out.append(version_say(root, conf, learning_dir))
        if progress:
            out.append(progress_say(learning_dir, projects_root=projects_root, doing=doing, enabled=enabled, active=active))
        return "".join(out)
    return answer


def main() -> None:
    ap = argparse.ArgumentParser(description="星期一's version and what she is busy with")
    ap.add_argument("what", nargs="?", choices=("version", "progress", "both"), default="both")
    a = ap.parse_args()
    if a.what in ("version", "both"):
        print(version_say())
    if a.what in ("progress", "both"):
        print(progress_say())


if __name__ == "__main__":
    main()
