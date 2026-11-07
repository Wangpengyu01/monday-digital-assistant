"""自我检查: she reads her own recent conversations and logs, finds where she did badly, and starts a repair
project (learning.project, kind "repair") for the worst one - the way a developer would: evidence first,
then the code, a test that reproduces it, the fix, the full test suite, then deliver (auto rollback) or,
for protected files, a proposal the owner approves ("批准修改").

    python3 -m learning.selfreview --dry-run      # evidence + the teacher's diagnosis, nothing started
    python3 -m learning.selfreview                # ... and start a repair project for the top problem

What counts as evidence (since the last review; run/learning/selfreview_state.json keeps the place):
  - turns (run/learning/turns/*.jsonl): tasks that failed, "没听明白", the owner correcting her, the
    owner asking the same thing again and again;
  - the orchestrator log: teacher replies she could not read, approvals that timed out, danger-level
    questions, tracebacks, repeated answers, long silences after a task started.
Every review is logged to run/learning/selfreview.jsonl (problems found, what was started, why not).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
LEARNING = AGENT_ROOT / "run" / "learning"
LOG = logging.getLogger("selfreview")
STATE_FILE = "selfreview_state.json"
LOG_FILE = "selfreview.jsonl"
FIRST_LOOK_BYTES = 3 << 20     # no saved place yet (or the log was rotated): read the last 3 MB
MAX_LOG_BYTES = 8 << 20
EVIDENCE_CHARS = 12000
TURNS_CHARS = 5000             # all turns since the last review (the newest when they do not all fit)
SLOW_S = 12                    # an answer's first sentence later than this after the intent step is slow
SILENCE_S = 15                 # a task's first sentence later than this is a silence the owner noticed
REDO_DAYS = 3                  # a problem worked on in the last few days is not started again (unless it failed)
MONTH_YUAN = 60.0              # what her self-chosen repairs may cost per month in all
CODE_DIRS = ("orchestrator", "learning", "llm", "asr", "tts", "ops", "ui", "devices", "mcp_servers")

# log line -> what it means for the owner
SIGNALS = (
    ("teacher_unreadable", re.compile(r"\[agent\] unusable plan"), "老师给的步骤她解析不了"),
    ("approval_timeout", re.compile(r"没等到你的确认"), "问主人要不要做，主人没来得及回答就取消了"),
    ("danger_ask", re.compile(r"\[agent\] approval \w+: .*\((?:danger),"), "当成危险操作去问主人（每次都问、还存备份）"),
    ("task_failed", re.compile(r"\[agent\] done ok=False"), "办事没办成"),
    ("traceback", re.compile(r"Traceback \(most recent call last\)"), "程序报错"),
    ("repeat", re.compile(r"repetition detected|suppressed repeated"), "说话复读"),
    ("play_failed", re.compile(r"\[tts\] play failed"), "声音没放出来"),
    ("teacher_down", re.compile(r"\[agent\] teacher failed|teacher.*(?:timed out|HTTP 5\d\d)"), "老师联系不上"),
)
_TIME = re.compile(r"^(\d\d):(\d\d):(\d\d) ")
_ASKED = re.compile(r"\[intent\] (?:knowledge|ask_teacher|chat) ")   # a question, answered by talking

REVIEW_SYSTEM = (
    "你在帮语音助手「星期一」做自我检查：她跑在一块 RK3588 板子上，代码在 /userdata/agent。下面是她最近跟主人对话里"
    "出问题的地方和日志信号，还有她的代码文件清单和最近修过的问题。\n"
    "先逐条读「全部对话」（主人的话是语音识别出来的，可能有错字），像挑剔的用户一样挑出答得不好的：答非所问、"
    "不懂装懂或说错、乱承诺做不到的事（她是板子上的语音助手，不能随口答应「今晚给你」「三周做完」）、"
    "该动手做却只是嘴上讲、不认识自己（自己是什么、谁做的、能做什么）、太慢、太啰嗦、办不成事、反复问主人、"
    "同一个问题被问好几次。再看「明显出问题的地方」和「日志信号」。\n"
    "找出最影响主人体验、能靠改她自己的代码（提示词、路由、规则、流程）修好的问题，按严重程度排，最多 3 个。"
    "主人亲身感受到的优先，纯日志噪音靠后。\n"
    "每个问题写清：现象（主人看到了什么）、证据（引用原话或日志行，带时间）、你猜的原因和可能的代码位置、"
    "修好以后怎么验证。不是代码问题的（主人自己没回答、网络断了、硬件没接）不要列；最近修过而且之后没再出现的不要列，"
    "修过但还在出现的可以再列，并说明上次没修好。\n"
    "「主人不满意的地方」是她最在意的，排在最前面。「上次没修好、她不服气要再试的」可以再列（标题照抄），"
    "cause_guess 里写上次为什么没成、这次换什么思路。"
    "如果对话里没什么问题，就对照「我的目标」挑一个最值得提升、能靠改代码做到的地方（比如回答更快、更懂自己），"
    "kind 写 grow；修问题的 kind 写 fix。\n"
    '只输出 JSON：{"problems":[{"title":"一句话","symptom":"...","evidence":"...","cause_guess":"...",'
    '"files":["相对路径"],"verify":"...","kind":"fix 或 grow"}]}'
)


# ---------------------------------------------------------------- evidence
def load_state(learning_dir: Path) -> dict:
    try:
        return json.loads((learning_dir / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(learning_dir: Path, st: dict) -> None:
    learning_dir.mkdir(parents=True, exist_ok=True)
    tmp = learning_dir / f".{STATE_FILE}.tmp"
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, learning_dir / STATE_FILE)


def _norm(q: str) -> str:
    return re.sub(r"[\s，,。.！!？?、~～…了吗呢啊呀吧]", "", q or "")


def read_turns(learning_dir: Path, since: datetime) -> list[dict]:
    out = []
    day = since.date()
    while day <= datetime.now().date():
        f = learning_dir / "turns" / f"{day.isoformat()}.jsonl"
        for line in (f.read_text(encoding="utf-8").splitlines() if f.is_file() else []):
            try:
                t = json.loads(line)
            except ValueError:
                continue
            try:
                if datetime.strptime(t.get("time", ""), "%Y-%m-%d %H:%M:%S") >= since:
                    out.append(t)
            except ValueError:
                continue
        day += timedelta(days=1)
    return out


def turn_lines(turns: list[dict], budget: int = 0) -> list[str]:
    """Every turn, the newest ones that fit: the teacher judges the answers the way the owner heard them."""
    budget = budget or TURNS_CHARS
    out, used = [], 0
    for t in reversed(turns):
        line = (f"[{t.get('time', '')[5:16]}] {t.get('who', '')}/{t.get('reason', '')}｜主人：{t.get('q', '')[:70]}"
                f"｜她：{t.get('a', '')[:150]}")
        if used + len(line) > budget:
            break
        out.append(line)
        used += len(line) + 1
    return out[::-1]


def turn_findings(turns: list[dict]) -> list[str]:
    """The turns that went wrong, and the questions the owner had to ask again."""
    bad = []
    for t in turns:
        who, why = t.get("who", ""), t.get("reason", "")
        if (who == "agent" and why == "failed") or (who == "intent" and why == "unclear") or why == "correction":
            label = {"failed": "办事没办成", "unclear": "没听明白", "correction": "主人纠正她"}.get(why, why)
            bad.append(f"[{t.get('time', '')}] {label}｜主人：{t.get('q', '')[:80]}｜她：{t.get('a', '')[:100]}")
    asked: dict[str, list[dict]] = {}
    for t in turns:
        asked.setdefault(_norm(t.get("q", ""))[:24], []).append(t)
    for key, ts in asked.items():
        if key and len(ts) >= 2:
            bad.append(f"主人同一个问题问了 {len(ts)} 次：「{ts[0].get('q', '')[:40]}」，她每次答："
                       + " / ".join(x.get("a", "")[:40] for x in ts[:3]))
    return bad


def read_log(path: Path, offset: int | None) -> tuple[str, int]:
    """-> (new text since offset, the new offset). A missing place or a rotated log reads the tail."""
    try:
        size = path.stat().st_size
    except OSError:
        return "", 0
    if offset is None or offset > size:
        offset = max(0, size - FIRST_LOOK_BYTES)
    offset = max(offset, size - MAX_LOG_BYTES)
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    return data.decode("utf-8", errors="replace"), size


def log_findings(text: str) -> list[str]:
    """Counts and a few examples of each signal, plus silences after a task started."""
    lines = text.splitlines()
    hits: dict[str, list[str]] = {}
    task_at, task_text = None, ""
    ask_at, ask_text = None, ""
    silences, slow = [], []
    for i, line in enumerate(lines):
        for key, pat, _ in SIGNALS:
            if pat.search(line):
                sample = line[:260]
                if key == "traceback":   # the exception is a few lines down
                    tail = [x for x in lines[i + 1:i + 12] if x.strip() and not _TIME.match(x)]
                    sample = (lines[i - 1][:120] + " … " if i else "") + " | ".join(x.strip()[:160] for x in tail[-3:])
                hits.setdefault(key, []).append(sample)
        m = _TIME.match(line)
        if m and _ASKED.search(line):
            ask_at, ask_text = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)), line.split(": ", 1)[-1][:50]
        elif m and ask_at is not None and "[tts] speak (1/" in line:
            gap = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) - ask_at) % 86400
            if gap > SLOW_S:
                slow.append(f"{line[:8]} 「{ask_text}」{gap} 秒才说出答案")
            ask_at = None
        if m and "[agent] task:" in line:
            task_at, task_text = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)), line[line.find("task:"):][:80]
        elif m and task_at is not None and "[tts] speak (1/" in line:
            gap = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) - task_at) % 86400
            if gap > SILENCE_S:
                silences.append(f"{line[:8]} 接了任务（{task_text}）后 {gap} 秒才开口")
            task_at = None
    out = []
    for key, pat, meaning in SIGNALS:
        if key in hits:
            ex = hits[key]
            out.append(f"{meaning}：{len(ex)} 次。例：" + "；".join(dict.fromkeys(ex[-3:])))
    if silences:
        out.append(f"接任务后长时间不出声：{len(silences)} 次。例：" + "；".join(silences[-3:]))
    if slow:
        out.append(f"回答慢（超过 {SLOW_S} 秒才说出答案）：{len(slow)} 次。例：" + "；".join(slow[-3:]))
    return out


def code_files(root: Path = AGENT_ROOT, limit: int = 160) -> str:
    rows = []
    for d in CODE_DIRS:
        for dirpath, dirnames, files in os.walk(root / d):
            dirnames[:] = sorted(x for x in dirnames if x not in ("__pycache__", "tests", "thirdparty", "build"))
            for f in sorted(files):
                if f.endswith(".py"):
                    rows.append(Path(dirpath, f).relative_to(root).as_posix())
    return "、".join(rows[:limit])


def recent_reviews(learning_dir: Path, n: int = 8) -> list[dict]:
    f = learning_dir / LOG_FILE
    rows = []
    for line in (f.read_text(encoding="utf-8").splitlines()[-n:] if f.is_file() else []):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def evidence(learning_dir: Path, root: Path = AGENT_ROOT, *, hours: float | None = None,
             projects_root: Path | None = None) -> tuple[str, dict]:
    """-> (the evidence text for the teacher, the new state to save once the review is logged)."""
    st = load_state(learning_dir)
    if hours is not None:
        since = datetime.now() - timedelta(hours=hours)
    else:
        try:
            since = datetime.strptime(st["last"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            since = datetime.now() - timedelta(days=3)
    turns = read_turns(learning_dir, since)
    text, offset = read_log(root / "logs" / "orchestrator.log", None if hours is not None else st.get("log_offset"))
    from learning import drive

    dv = drive.load(learning_dir)
    pains = _pains(turns, dv, since)
    grudges = drive.due_grudges(learning_dir)
    parts = [f"我的目标：{dv['ambition']}",
             "主人不满意的地方（她最在意，最优先）：\n" + ("\n".join(
                 f"[{p['time']}] 主人说「{p['owner']}」——因为她答「{p['a'][:120]}」（主人原来问的是「{p['q'][:50]}」）"
                 for p in pains[-12:]) if pains else "（没有）"),
             "上次没修好、她不服气要再试的：\n" + ("\n".join(
                 f"- {g['title']}（已经试了 {g['tries']} 次；上次没成是因为：{g.get('why', '')[:160]}）" for g in grudges)
                 if grudges else "（没有）"),
             f"检查范围：{since:%Y-%m-%d %H:%M} 以来，对话 {len(turns)} 轮。"]
    from learning import spar

    last = spar.latest(learning_dir)
    if last and last[1]["turns"] and last[0] >= (since - timedelta(days=1)).date().isoformat():
        day, n = last
        parts.append(f"陪练里她常犯的毛病（{day}，老师扮成主人跟她聊了 {n['turns']} 句，答好 {n['ok']} 句；是模拟的，"
                     "排在真实对话后面）：\n" + "\n".join(f"- {p}" for p in n["problems"][-8:]))
    tf, lf, every = turn_findings(turns), log_findings(text), turn_lines(turns)
    parts.append(f"全部对话（{len(every)} 轮；who/reason 是谁答的、为什么这样答）：\n" + ("\n".join(every) if every else "（没有）"))
    parts.append("明显出问题的地方：\n" + ("\n".join(tf[-25:]) if tf else "（没有）"))
    parts.append("日志信号：\n" + ("\n".join(lf) if lf else "（没有）"))
    tried = []
    for r in recent_reviews(learning_dir):
        if r.get("started"):
            p = _project_state(r["started"], projects_root)
            tried.append(f"{r.get('time', '')[:10]} 修过「{r.get('title', '')}」→ {p.get('status', '?')}：{p.get('summary', '')[:60]}")
    parts.append("最近修过的：\n" + ("\n".join(tried) if tried else "（没有）"))
    body = "\n\n".join(parts)
    if len(body) > EVIDENCE_CHARS:
        body = body[:EVIDENCE_CHARS] + "\n…"
    new_state = {"last": time.strftime("%Y-%m-%d %H:%M:%S"), "log_offset": offset}
    return body, {"found": bool(turns or lf or grudges), "pains": pains, "grudges": grudges, "state": new_state}


def _pains(turns: list[dict], dv: dict, since: datetime) -> list[dict]:
    """The owner's frustration since `since`: the live ones (learning.drive.feel) and the ones in the turn
    journal, once each."""
    from learning import drive

    out, seen = [], set()
    for day, v in sorted(dv["days"].items()):
        for p in v["pains"]:
            stamp = f"{day} {p['time']}"
            if stamp >= since.strftime("%Y-%m-%d %H:%M:%S"):
                out.append(dict(p, time=stamp))
    for p in drive.pains_from_turns(turns):
        out.append(p)
    uniq = []
    for p in sorted(out, key=lambda x: x["time"]):
        key = (p["time"][:16], p["owner"][:10])
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _project_state(slug: str, projects_root: Path | None) -> dict:
    from learning import project
    from learning import projbox

    return project.load_state((projects_root or projbox.PROJECTS) / slug)


# ---------------------------------------------------------------- diagnosis and repair
def diagnose(teacher, body: str, root: Path = AGENT_ROOT) -> list[dict]:
    from learning.agent import json_object

    raw = teacher.complete(REVIEW_SYSTEM, f"{body}\n\n代码文件：{code_files(root)}", json_mode=True,
                           max_tokens=2500, timeout_s=180)
    probs = json_object(raw).get("problems") or []
    return [p for p in probs if isinstance(p, dict) and str(p.get("title") or "").strip()][:3]


def brief_of(p: dict) -> str:
    files = "、".join(str(f) for f in p.get("files") or []) or "（没给）"
    return (f"现象：{p.get('symptom', '')}\n证据：{p.get('evidence', '')}\n猜的原因：{p.get('cause_guess', '')}\n"
            f"可能的代码位置：{files}\n怎么验证修好了：{p.get('verify', '')}")


def recently_tried(title: str, learning_dir: Path) -> bool:
    """Worked on in the last few days - unless it failed and is due again (a grudge: she does not let go)."""
    from learning import drive

    key = _norm(title)[:12]
    if any(_norm(g["title"])[:12] == key for g in drive.due_grudges(learning_dir)):
        return False
    cutoff = (datetime.now() - timedelta(days=REDO_DAYS)).strftime("%Y-%m-%d")
    return any(r.get("started") and r.get("time", "")[:10] >= cutoff and _norm(r.get("title", ""))[:12] == key
               for r in recent_reviews(learning_dir, 30))


def month_spent(projects_root: Path | None = None) -> float:
    """What her self-repair projects cost this month (the cap keeps "always wanting to improve" affordable)."""
    from learning import project

    month = time.strftime("%Y-%m")
    return sum(float(p.get("spent") or 0) for p in project.list_projects(projects_root)
               if p.get("kind") == "repair" and str(p.get("started", "")).startswith(month))


def review(teacher, *, learning_dir: Path = LEARNING, root: Path = AGENT_ROOT, start: bool = True,
           hours: float | None = None, projects_root: Path | None = None,
           start_project: Callable[[str], str] | None = None, busy: Callable[[], bool] | None = None,
           grow: bool = True) -> dict:
    """One review -> the logged record {"problems", "started", "title", "skipped", "pains"}.
    grow: on a quiet day she still picks something to get better at, toward her own ambition (drive)."""
    from learning import drive, project

    body, info = evidence(learning_dir, root, hours=hours, projects_root=projects_root)
    rec: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "evidence_chars": len(body), "problems": [],
                 "started": "", "title": "", "skipped": "", "pains": len(info["pains"])}
    if not info["found"] and not grow:
        rec["skipped"] = "最近没发现问题"
    elif month_spent(projects_root) >= MONTH_YUAN:
        rec["skipped"] = f"这个月自己修自己已经花到 {MONTH_YUAN:.0f} 元的上限"
    else:
        rec["problems"] = diagnose(teacher, body, root)
        todo = [p for p in rec["problems"] if not recently_tried(str(p["title"]), learning_dir)]
        if not rec["problems"]:
            rec["skipped"] = "老师没找出能改代码修的问题"
        elif not todo:
            rec["skipped"] = "找到的问题最近都修过了"
        elif not start:
            rec["skipped"] = "只检查，不动手（--dry-run）"
        elif (busy or (lambda: project.current(projects_root) is not None))():
            rec["skipped"] = "手上还有项目在做"
        else:
            p = todo[0]
            again = next((g for g in drive.due_grudges(learning_dir) if _norm(g["title"])[:12] == _norm(str(p["title"]))[:12]), None)
            retry = (f"\n\n这是第 {again['tries'] + 1} 次修这个问题。上次没成：{again.get('why', '')}。这次换个思路。"
                     if again else "")
            slug = project.create(f"修好自己：{str(p['title'])[:40]}", projects_root, kind="repair",
                                  brief=brief_of(p) + retry + "\n\n原始证据（节选）：\n" + body[:2500])
            err = (start_project or project.start_unit)(slug)
            if err:
                rec["skipped"] = f"项目没启动：{err[:80]}"
            else:
                rec["started"], rec["title"] = slug, str(p["title"])[:60]
                rec["kind"] = str(p.get("kind") or "fix")
                drive.set_focus(rec["title"], learning_dir)
    learning_dir.mkdir(parents=True, exist_ok=True)
    with (learning_dir / LOG_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if hours is None:
        save_state(learning_dir, info["state"])
    return rec


def started_notice(rec: dict) -> str:
    """What she tells the owner (in the morning: notices wait out the night) when she has started on herself."""
    title = rec["title"][:30]
    if rec.get("pains"):
        return (f"主人，昨天你有{_cn_count(rec['pains'])}次对我不太满意，我挺在意的。"
                f"我自己在改「{title}」，改好了告诉你。")
    if rec.get("kind") == "grow":
        return f"主人，昨天没出什么大毛病，我给自己找了个能变得更好的地方：「{title}」，正在自己练，好了告诉你。"
    return f"主人，我翻了最近的对话，发现「{title}」做得不好，正在自己修，修好了告诉你。"


def _cn_count(n: int) -> str:
    from learning.selfstatus import count

    return count(n)


def start_unit() -> str:
    """Run a review as its own systemd unit (it has the teacher key; the voice command starts it)."""
    cmd = ["sudo", "-n", "systemd-run", "--no-block", "--quiet", "--collect", "--unit=r1-selfreview",
           "--uid=youyeetoo", "--gid=youyeetoo", "-p", "EnvironmentFile=/userdata/secrets/deepseek.env",
           "-p", f"WorkingDirectory={AGENT_ROOT}", "-p", f"Environment=PYTHONPATH={AGENT_ROOT}",
           "-p", "Environment=MONDAY_USAGE_TAG=自我检查", "-p", "Nice=5",
           "-p", f"StandardOutput=append:{AGENT_ROOT}/logs/selfreview.log",
           "-p", f"StandardError=append:{AGENT_ROOT}/logs/selfreview.log",
           "/usr/bin/python3", "-m", "learning.selfreview"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)[:200]
    return "" if p.returncode == 0 else (p.stderr or p.stdout).strip()[:200]


def main() -> None:
    ap = argparse.ArgumentParser(description="星期一 looks through her own conversations and fixes what went wrong")
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    ap.add_argument("--dry-run", action="store_true", help="find and diagnose, start nothing")
    ap.add_argument("--hours", type=float, help="look at the last N hours (does not move the saved place)")
    ap.add_argument("--max-wait-h", type=float, default=8.0, help="wait this long for the teacher's peak hours to pass")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.environ.setdefault("MONDAY_USAGE_TAG", "自我检查")
    from learning import project, usage
    from learning.reflect import build_runtime

    deadline = time.time() + a.max_wait_h * 3600
    while usage.peak_blocked() and time.time() < deadline:
        time.sleep(60)
    if usage.peak_blocked():
        LOG.info("still the teacher's peak hours: no review")
        return
    from learning import drive

    rt = build_runtime(a.config)
    if not a.dry_run:
        drive.rewrite_ambition(rt.teacher, rt.learning_dir)   # once a week she rethinks who she wants to become
    rec = review(rt.teacher, learning_dir=rt.learning_dir, start=not a.dry_run, hours=a.hours)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    if rec["started"]:
        project.notify(started_notice(rec))
    elif not a.dry_run and rec["skipped"] not in ("最近没发现问题",):
        LOG.info("no repair started: %s", rec["skipped"])


if __name__ == "__main__":
    main()
