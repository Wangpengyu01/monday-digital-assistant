"""陪练: DeepSeek plays the owner and talks with her *local* model; every answer she gives is judged, and the
judged answers become training data for her next fine-tune.

Distillation (learning.distill) trains her on answers the teacher wrote to questions the teacher made up:
she never speaks, so the mistakes only she makes (promising what she cannot do, not knowing herself,
missing the owner's speech-recognition slips) are never seen, let alone corrected. Here she answers
herself and the teacher corrects *her* answers (on-policy distillation):

  owner   the teacher plays the owner, from what he really said (run/learning/turns): his way of talking,
          what he cares about, the slips of speech recognition, follow-ups, impatience. Each session has a
          focus: what he was annoyed at lately (learning.drive pains), something about her, or an everyday
          area (learning.distill.VOICE_AREAS).
  her     the board's local model answers exactly as in a live turn (her persona, her own voice rules,
          the last turns clipped the same way, the date line).
  judge   the teacher judges that answer in context, with the facts about her (learning.selfstatus); a bad
          one gets the answer she should have given, which must pass the same checks as distillation
          (speakable, short, no promises she cannot keep, no exam answers).
  on      the owner goes on from what she *actually* said - a bad answer gets its "又错了", and she learns
          to recover from that too.

Out (run/learning/):
  distill/spar.jsonl   training samples, picked up by learning.lora_prep: her good answers as they were, bad
                       ones replaced by the checked correction (loss on the last reply only, history is context)
  spar/pairs.jsonl     {messages, chosen, rejected} for preference training later
  spar/sessions.jsonl  every session: the transcript and the judgements
  spar/nights.json     per night: turns, how many she got right, which kinds of mistakes - the trend
Guards: off-peak only (learning.usage), stops as soon as the owner talks (the local model is his first),
yuan per night and per month (usage tag 陪练).

    cd /userdata/agent && python3 -m learning.spar [--max-yuan 3] [--minutes 120] [--any-time]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
LEARNING = AGENT_ROOT / "run" / "learning"
LOG = logging.getLogger("spar")
TAG = "陪练"
MAX_TURNS = 5            # per session
NIGHT_YUAN = 3.0
MONTH_YUAN = 60.0
SEEDS = 40               # real owner sentences shown to the teacher playing him
HIST_Q, HIST_A, HIST_N = 40, 70, 4   # the last turns as the live prompt clips them (orchestrator _history_for_prompt)
KINDS = ("答非所问", "说错", "编造", "乱承诺", "不认识自己", "啰嗦", "没听懂", "推脱", "念不出", "其他")

OWNER_SYSTEM = (
    "你在扮演一个真实的人：语音助手「星期一」的主人，在家里随口跟她说话。下面是他平时真实说过的话，"
    "照着他的说话方式、关心的事和口头禅来说；他的话是语音识别出来的，偶尔会有同音错字或少字，你也偶尔这样。\n"
    "每次只说一句，像真人一样：有问题、有吩咐、有闲聊，会追问；她答得不好你会不耐烦（比如「又错了」「你在说什么」「算了」），"
    "答得好就接着聊或者换个话题，聊够了就结束。\n"
    "不要让她执行命令、查机器的实时状态、连别的设备（那些另有流程），不要说账号密码，不要问需要实时信息的（天气、新闻、今天几号、价格）。\n"
    '只输出 JSON：{"say": "主人这一句", "end": true或false}'
)
JUDGE_SYSTEM = (
    "你在给语音助手「星期一」的一句回答打分。她跑在一块开发板上，用本地小模型回答，回答会被直接念给主人听。"
    "根据她的人设、关于她的事实、前面的对话和主人这一句，判断她这句答得好不好。下面这些都算不好：答非所问、说错事实、编造、"
    "乱承诺做不到的事（她不能定闹钟提醒、不能到点主动开口、不能连夜开发、不能保证几天几周交付）、不认识自己、太啰嗦、"
    "没听懂主人的错字、该答的推脱掉、念出代码符号路径、结尾问还需要什么。\n"
    "好就 ok 为 true。不好就写出她应该怎么答：星期一的口吻，一到三句，先说结论，纯口语，数字用中文说法，能直接念；"
    "需要实时信息或确实不知道的就直说，说去问问老师，不编造。\n"
    '只输出 JSON：{"ok": true或false, "kind": "答非所问|说错|编造|乱承诺|不认识自己|啰嗦|没听懂|推脱|念不出|其他", '
    '"problem": "不好在哪", "better": "应该这样答"}（ok 为 true 时 kind、problem、better 留空）'
)


# ---------------------------------------------------------------- who the owner is
def owner_seeds(learning_dir: Path = LEARNING, days: int = 21, n: int = SEEDS, rng: random.Random | None = None) -> list[str]:
    """Real sentences the owner said lately (one copy each), a random handful."""
    rng = rng or random.Random()
    seen, out = set(), []
    today = dt.date.today()
    for back in range(days):
        f = learning_dir / "turns" / f"{(today - dt.timedelta(days=back)).isoformat()}.jsonl"
        for line in (f.read_text(encoding="utf-8").splitlines() if f.is_file() else []):
            try:
                q = str(json.loads(line).get("q") or "").strip()
            except ValueError:
                continue
            key = re.sub(r"[\s，。？！,.?!]", "", q)
            if 2 <= len(key) <= 60 and key not in seen:
                seen.add(key)
                out.append(q)
    rng.shuffle(out)
    return out[:n]


def focuses(learning_dir: Path = LEARNING) -> list[str]:
    """What a session can be about: his recent annoyances first, then herself, then everyday areas."""
    from learning import drive
    from learning.distill import VOICE_AREAS

    d = drive.load(learning_dir)
    pains = [p for v in d["days"].values() for p in v["pains"]][-10:]
    out = [f"他最近不满意的地方：他问「{p['q'][:40]}」，她答「{p['a'][:60]}」，他说「{p['owner'][:20]}」。从这个话题聊起"
           for p in pains]
    out += ["问问她自己：她是谁、谁做的、跑在什么上、什么版本、能做什么不能做什么", "让她帮忙想办法或者做个计划，看她会不会乱答应"]
    return out + [f"日常话题：{a}" for a in VOICE_AREAS]


# ---------------------------------------------------------------- her side, as in a live turn
def date_line(now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now()
    wd = "一二三四五六日"[now.weekday()]
    part = "凌晨" if now.hour < 6 else "上午" if now.hour < 12 else "中午" if now.hour < 13 else "下午" if now.hour < 18 else "晚上"
    return f"当前时间：{now.year}年{now.month}月{now.day}日，星期{wd}，{part}{now.hour % 12 or 12}点{now.minute:02d}分。"


def her_system(persona: str, learning_dir: Path = LEARNING, now: dt.datetime | None = None) -> str:
    from learning import selfprompt

    return f"{date_line(now)}\n{selfprompt.apply(persona, 'voice', learning_dir=learning_dir)}"


def messages(system: str, history: list[tuple[str, str]], user: str) -> list[dict]:
    """The exact context she sees live: the last turns, clipped the way the orchestrator clips them."""
    m = [{"role": "system", "content": system}]
    for q, a in history[-HIST_N:]:
        m += [{"role": "user", "content": q[:HIST_Q]}, {"role": "assistant", "content": a[:HIST_A]}]
    return m + [{"role": "user", "content": user}]


def chatml(m: list[dict]) -> str:
    return "".join(f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n" for x in m) + "<|im_start|>assistant\n"


def board_student() -> Callable[[list[dict]], str]:
    from llm.client import llm_chat, llm_clear_history

    def ask(m: list[dict]) -> str:
        llm_clear_history()
        text = str(llm_chat(chatml(m), max_new_tokens=200).get("text", ""))
        return re.sub(r"(?s)<think>.*?</think>", "", text).strip()
    return ask


# ---------------------------------------------------------------- the teacher's two roles
def owner_says(teacher, seeds: list[str], focus: str, transcript: list[tuple[str, str]]) -> dict:
    from learning.agent import json_object

    convo = "\n".join(f"主人：{q}\n星期一：{a}" for q, a in transcript) or "（还没开口）"
    user = ("他平时说过的话：\n" + "\n".join(f"- {s}" for s in seeds) + f"\n\n这一次聊：{focus}\n\n到目前为止：\n{convo}\n\n"
            "主人下一句？" + ("（已经聊了不少，可以结束了）" if len(transcript) >= MAX_TURNS - 1 else ""))
    d = json_object(teacher.complete(OWNER_SYSTEM, user, json_mode=True, max_tokens=200, timeout_s=60))
    return {"say": str(d.get("say") or "").strip()[:80], "end": bool(d.get("end"))}


def judge(teacher, persona: str, facts: str, transcript: list[tuple[str, str]], q: str, a: str) -> dict:
    from learning.agent import json_object

    before = "\n".join(f"主人：{x}\n星期一：{y}" for x, y in transcript[-HIST_N:]) or "（这是第一句）"
    user = (f"她的人设：\n{persona}\n\n关于她的事实：{facts}\n\n前面的对话：\n{before}\n\n主人：{q}\n她答：{a}")
    d = json_object(teacher.complete(JUDGE_SYSTEM, user, json_mode=True, max_tokens=400, timeout_s=60))
    kind = str(d.get("kind") or "其他")
    return {"ok": bool(d.get("ok")), "kind": kind if kind in KINDS else "其他",
            "problem": str(d.get("problem") or "")[:160], "better": str(d.get("better") or "").strip()}


def usable(text: str, q: str = "") -> bool:
    """A reply that may be trained on: speakable, short, no promise she cannot keep, no exam answer."""
    from learning import distill, exam

    if not distill.voice_ok(text) or distill.promises(text):
        return False
    if any(x in text for x in exam.FIXED_ANSWERS):
        return False
    return not any(re.sub(r"\W", "", q) == re.sub(r"\W", "", it.q) for it in exam.ITEMS)   # never an exam question


# ---------------------------------------------------------------- a session and a night
def session(teacher, ask: Callable[[list[dict]], str], *, persona: str, system: str, facts: str, seeds: list[str],
            focus: str, learning_dir: Path = LEARNING, idle: Callable[[], bool] = lambda: True) -> dict:
    """One conversation -> {"turns": [...], "stopped": bool}; samples and pairs are written as it goes."""
    out_dir = learning_dir / "spar"
    out_dir.mkdir(parents=True, exist_ok=True)
    (learning_dir / "distill").mkdir(parents=True, exist_ok=True)
    transcript: list[tuple[str, str]] = []
    rec: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "focus": focus[:80], "turns": [], "stopped": False}
    for _ in range(MAX_TURNS):
        if not idle():
            rec["stopped"] = True
            break
        o = owner_says(teacher, seeds, focus, transcript)
        if not o["say"]:
            break
        m = messages(system, transcript, o["say"])
        a = ask(m)
        j = judge(teacher, persona, facts, transcript, o["say"], a)
        turn = {"q": o["say"], "a": a, **j, "kept": ""}
        if j["ok"] and usable(a, o["say"]):
            _append(learning_dir / "distill" / "spar.jsonl", {"messages": m + [{"role": "assistant", "content": a}], "source": "spar"})
            turn["kept"] = "hers"
        elif not j["ok"] and j["better"] and usable(j["better"], o["say"]):
            _append(learning_dir / "distill" / "spar.jsonl",
                    {"messages": m + [{"role": "assistant", "content": j["better"]}], "source": "spar"})
            _append(out_dir / "pairs.jsonl", {"messages": m, "chosen": j["better"], "rejected": a, "kind": j["kind"]})
            turn["kept"] = "corrected"
        rec["turns"].append(turn)
        transcript.append((o["say"], a))   # he goes on from what she actually said
        if o["end"]:
            break
    _append(out_dir / "sessions.jsonl", rec)
    return rec


def _append(f: Path, row: dict) -> None:
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_nights(learning_dir: Path = LEARNING) -> dict:
    try:
        return json.loads((learning_dir / "spar" / "nights.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record_night(learning_dir: Path, day: str, sessions: list[dict], yuan: float) -> dict:
    nights = load_nights(learning_dir)
    turns = [t for s in sessions for t in s["turns"]]
    n = nights.setdefault(day, {"sessions": 0, "turns": 0, "ok": 0, "kept": 0, "kinds": {}, "yuan": 0.0, "problems": []})
    n["sessions"] += len(sessions)
    n["turns"] += len(turns)
    n["ok"] += sum(1 for t in turns if t["ok"])
    n["kept"] += sum(1 for t in turns if t["kept"])
    kinds = Counter(n["kinds"]) + Counter(t["kind"] for t in turns if not t["ok"])
    n["kinds"] = dict(kinds.most_common())
    n["yuan"] = round(yuan, 3)
    n["problems"] = (n["problems"] + [f"{t['kind']}：主人说「{t['q'][:30]}」，她答「{t['a'][:50]}」——{t['problem'][:60]}"
                                      for t in turns if not t["ok"]])[-12:]
    nights = dict(sorted(nights.items())[-60:])
    p = learning_dir / "spar" / "nights.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".nights.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(nights, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return n


def latest(learning_dir: Path = LEARNING) -> tuple[str, dict] | None:
    nights = load_nights(learning_dir)
    if not nights:
        return None
    day = max(nights)
    return day, nights[day]


def spent(day: str | None = None, month: bool = False) -> float:
    from learning import usage

    today = dt.date.today()
    days = [today - dt.timedelta(days=i) for i in range(today.day)] if month else [dt.date.fromisoformat(day or today.isoformat())]
    return sum(float(((usage.summary(d.isoformat()).get("jobs") or {}).get(TAG) or {}).get("yuan", 0)) for d in days)


def night(teacher, ask: Callable[[list[dict]], str], *, persona: str, facts: str, learning_dir: Path = LEARNING,
          minutes: float = 120, max_yuan: float = NIGHT_YUAN, month_yuan: float = MONTH_YUAN,
          idle: Callable[[], bool] = lambda: True, blocked: Callable[[], bool] = lambda: False,
          cost: Callable[[], tuple[float, float]] | None = None, rng: random.Random | None = None) -> dict:
    """Sessions until the time, the night's budget or the month's runs out, the peak starts or the owner talks."""
    rng = rng or random.Random()
    cost = cost or (lambda: (spent(), spent(month=True)))
    seeds, topics = owner_seeds(learning_dir, rng=rng), focuses(learning_dir)
    end = time.time() + minutes * 60
    done: list[dict] = []
    why = "time"
    while time.time() < end:
        tonight, this_month = cost()
        if tonight >= max_yuan:
            why = "budget"
            break
        if this_month >= month_yuan:
            why = "month"
            break
        if blocked():
            why = "peak"
            break
        if not idle():
            why = "owner"
            break
        pains = [t for t in topics if t.startswith("他最近不满意")]
        focus = rng.choice(pains) if pains and rng.random() < 0.4 else rng.choice(topics)
        try:
            s = session(teacher, ask, persona=persona, system=her_system(persona, learning_dir), facts=facts,
                        seeds=rng.sample(seeds, min(len(seeds), 20)) if seeds else [], focus=focus,
                        learning_dir=learning_dir, idle=idle)
        except Exception as exc:  # noqa: BLE001 - a teacher hiccup or a busy model: wait a little, go on
            LOG.warning("session failed: %s", exc)
            time.sleep(20)
            continue
        done.append(s)
        if s["stopped"]:
            why = "owner"
            break
    n = record_night(learning_dir, dt.date.today().isoformat(), done, cost()[0])
    return {"sessions": len(done), "why": why, "night": n}


def say(learning_dir: Path = LEARNING) -> str:
    """"昨晚陪练得怎么样"."""
    from learning.selfstatus import count

    last = latest(learning_dir)
    if not last or not last[1]["turns"]:
        return ""
    day, n = last
    pct = round(100 * n["ok"] / n["turns"])
    top = "、".join(list(n["kinds"])[:2])
    return (f"最近一次陪练是{int(day[5:7])}月{int(day[8:])}号，老师扮成你跟我聊了{count(n['turns'])}句，"
            f"我答好了百分之{pct}" + (f"，主要毛病是{top}" if top else "") + "。答错的都改成了训练题，攒够了就练进脑子里")


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek plays the owner; her local model answers; the answers become training data")
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--max-yuan", type=float, default=NIGHT_YUAN)
    ap.add_argument("--any-time", action="store_true", help="also in the teacher's peak hours")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.environ.setdefault("MONDAY_USAGE_TAG", TAG)
    from learning import exam, usage
    from learning.practice import IDLE_MIN, last_activity
    from learning.reflect import build_runtime
    from learning.selfstatus import version_say

    rt = build_runtime(a.config)
    run_dir = AGENT_ROOT / "run"
    res = night(rt.teacher, board_student(), persona=exam.persona(a.config), facts=version_say(),
                learning_dir=rt.learning_dir, minutes=a.minutes, max_yuan=a.max_yuan,
                idle=lambda: time.time() - last_activity(run_dir) > IDLE_MIN * 60,
                blocked=(lambda: False) if a.any_time else usage.peak_blocked)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
