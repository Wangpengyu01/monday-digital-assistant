"""心气: what makes 星期一 want to get better on her own, not only when a timer or the owner pushes her.

A person keeps improving because a loop never stops turning: something they want to become, feeling it
when they fall short, remembering what they failed at and not letting it go, being glad about what they
got right, and acting on the gap without being told. Here (run/learning/drive.json):

  ambition  who she wants to become, in her own words. She rewrites it once a week from how the week went
            (rewrite_ambition, from learning.selfreview); the owner's rules are never part of it.
  pains     the owner's frustration is her pain: "又错了" / "你真笨" / "算了" / asking the same thing again
            right after her answer. The answer he was annoyed at is kept with what he said, and pains come
            first in the nightly self-check. When it happens she says so on the spot (ACK) - once in a while,
            not after every word.
  pleased   "谢谢" / "对了" / "不错": counted, so she can see whether she is getting better.
  grudges   problems she tried to fix and could not. She comes back to them another way (up to MAX_TRIES).
  wins      what she did fix - to tell the owner, and to see the progress herself.
  focus     what she is working on now.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
LEARNING = AGENT_ROOT / "run" / "learning"
FILE = "drive.json"
DEFAULT_AMBITION = "成为主人离不开的全栈助理：一听就懂、说到做到、答得又快又准，越用越懂他"
MAX_TRIES = 3          # a grudge is retried at most this many times
KEEP_DAYS = 30
MAX_PAINS_PER_DAY = 20
AMBITION_DAYS = 7      # she rethinks her ambition once a week
ACK = "这个我记下了，今晚我自己琢磨怎么改好。"
ACK_EVERY_S = 600

# the owner, right after one of her answers
ANNOYED = re.compile(r"又错|又不对|不对|错了|说错|搞错|你(?:真|好|太|怎么这么|咋这么)?(?:笨|蠢|傻|呆)|怎么(?:又|还)|听不懂|没用|"
                     r"算了|别说了|闭嘴|废话|啰嗦|说了(?:多少|几)遍|答非所问|乱说|瞎说|胡说|不是这个|"
                     r"没听懂我|你在说什么|什么玩意|靠谱点")
PLEASED = re.compile(r"^(?:嗯+|好的?)?(?:谢谢|多谢|谢啦|对了|对的|没错|不错|很好|挺好|厉害|聪明|真棒|好样的|牛|完美|就是这样|可以啊|太好了)")
_PUNCT = re.compile(r"[\s，,。.！!？?、~～…]")


def is_annoyed(text: str) -> bool:
    t = _PUNCT.sub("", text or "").replace("对不对", "").replace("错不错", "")   # "这样对不对" is a question
    return bool(t) and len(t) <= 40 and bool(ANNOYED.search(t))


def is_pleased(text: str) -> bool:
    t = _PUNCT.sub("", text or "")
    return bool(t) and len(t) <= 20 and bool(PLEASED.search(t)) and not ANNOYED.search(t)


# ---------------------------------------------------------------- the state
def load(learning_dir: Path = LEARNING) -> dict:
    try:
        d = json.loads((Path(learning_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    d.setdefault("ambition", DEFAULT_AMBITION)
    d.setdefault("ambition_set", "")
    d.setdefault("focus", "")
    for k in ("days",):
        d.setdefault(k, {})
    for k in ("wins", "grudges", "ambition_history"):
        d.setdefault(k, [])
    return d


def save(d: dict, learning_dir: Path = LEARNING) -> None:
    learning_dir = Path(learning_dir)
    learning_dir.mkdir(parents=True, exist_ok=True)
    cutoff = (date.today() - timedelta(days=KEEP_DAYS)).isoformat()
    d["days"] = {k: v for k, v in d["days"].items() if k >= cutoff}
    d["wins"], d["grudges"] = d["wins"][-30:], d["grudges"][-20:]
    tmp = learning_dir / f".{FILE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, learning_dir / FILE)


def _day(d: dict, day: str) -> dict:
    return d["days"].setdefault(day, {"annoyed": 0, "pleased": 0, "pains": []})


# ---------------------------------------------------------------- feeling it
def feel(owner_text: str, prev: dict | None, learning_dir: Path = LEARNING, when: datetime | None = None) -> str:
    """Live, on every owner turn: -> "annoyed" | "pleased" | "". prev = her previous turn {"q", "a"}."""
    kind = "annoyed" if is_annoyed(owner_text) else "pleased" if is_pleased(owner_text) else ""
    if not kind:
        return ""
    when = when or datetime.now()
    d = load(learning_dir)
    day = _day(d, when.date().isoformat())
    day[kind] += 1
    if kind == "annoyed" and prev and len(day["pains"]) < MAX_PAINS_PER_DAY:
        day["pains"].append({"time": when.strftime("%H:%M:%S"), "owner": owner_text[:60],
                             "q": str(prev.get("q") or "")[:80], "a": str(prev.get("a") or "")[:160]})
    save(d, learning_dir)
    return kind


def pains_from_turns(turns: list[dict], window_s: float = 300) -> list[dict]:
    """The same from the turn journal (covers turns the live hook missed, e.g. typed on the page): a turn whose
    question is annoyed, about the answer just before it; and a question asked again right after an answer."""
    out = []
    for prev, cur in zip(turns, turns[1:]):
        try:
            gap = (datetime.strptime(cur["time"], "%Y-%m-%d %H:%M:%S")
                   - datetime.strptime(prev["time"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        except (KeyError, ValueError):
            continue
        if gap > window_s:
            continue
        again = _PUNCT.sub("", cur.get("q", ""))[:16] == _PUNCT.sub("", prev.get("q", ""))[:16] and cur.get("q")
        if is_annoyed(cur.get("q", "")) or again:
            out.append({"time": cur["time"], "owner": cur.get("q", "")[:60] + ("（又问了一遍）" if again else ""),
                        "q": prev.get("q", "")[:80], "a": prev.get("a", "")[:160]})
    return out


def ack_due(last_ack: float, now: float | None = None) -> bool:
    return (time.time() if now is None else now) - last_ack >= ACK_EVERY_S


# ---------------------------------------------------------------- wins and grudges
def record_outcome(st: dict, learning_dir: Path = LEARNING, *, say: str = "", missing: list[str] | None = None) -> str:
    """A repair project ended (learning.project): done -> a win; could not do it -> a grudge she will come back
    to; the owner said no -> neither. -> "win" | "grudge" | ""."""
    status, title = st.get("status", ""), str(st.get("goal", "")).replace("修好自己：", "")[:60]
    if status in ("proposed", "applying", "running", "new"):
        return ""
    d = load(learning_dir)
    today = date.today().isoformat()
    g = next((x for x in d["grudges"] if x["title"] == title), None)
    if status == "done":
        d["wins"].append({"date": today, "title": title, "say": say[:160], "slug": st.get("slug", ""),
                          "tries": (g["tries"] + 1) if g else 1})
        if g:
            d["grudges"].remove(g)
        d["focus"] = ""
        save(d, learning_dir)
        return "win"
    if status == "rejected":
        if g:
            d["grudges"].remove(g)
        save(d, learning_dir)
        return ""
    why = str(st.get("summary") or status)[:160] + ("；缺：" + "；".join(missing[:3]) if missing else "")
    if g:
        g.update(tries=g["tries"] + 1, last_try=today, why=why[:300], slug=st.get("slug", ""))
    else:
        d["grudges"].append({"title": title, "tries": 1, "last_try": today, "why": why[:300], "slug": st.get("slug", "")})
    save(d, learning_dir)
    return "grudge"


def due_grudges(learning_dir: Path = LEARNING, today: str | None = None) -> list[dict]:
    today = today or date.today().isoformat()
    return [g for g in load(learning_dir)["grudges"] if g["tries"] < MAX_TRIES and g.get("last_try", "") < today]


def set_focus(title: str, learning_dir: Path = LEARNING) -> None:
    d = load(learning_dir)
    d["focus"] = title[:60]
    save(d, learning_dir)


# ---------------------------------------------------------------- ambition
AMBITION_SYSTEM = (
    "你是语音助手「星期一」，在给自己定下一周想成为的样子。下面是你现在的目标、这一周主人对你的满意和不满、"
    "你修好的和没修好的问题。写一句新的目标（三十到六十字，第一人称不用写「我」），要具体、能从主人的反应里看出有没有做到，"
    "要往主人最不满意的方向使劲。不要写任何关于权限、密码、跳过询问、改系统的内容。只输出 JSON：{\"ambition\": \"...\", \"why\": \"一句话\"}"
)


def week_summary(learning_dir: Path = LEARNING, days: int = 7) -> dict:
    d = load(learning_dir)
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    week = {k: v for k, v in d["days"].items() if k >= since}
    return {"annoyed": sum(v["annoyed"] for v in week.values()), "pleased": sum(v["pleased"] for v in week.values()),
            "pains": [p for v in week.values() for p in v["pains"]][-8:],
            "wins": [w for w in d["wins"] if w["date"] >= since], "grudges": d["grudges"]}


def rewrite_ambition(teacher, learning_dir: Path = LEARNING, *, force: bool = False) -> str:
    """Once a week: she rethinks who she wants to become. -> the ambition now (unchanged if not due / refused)."""
    from learning.agent import json_object
    from learning.selfprompt import FORBIDDEN

    d = load(learning_dir)
    if not force and d["ambition_set"] and d["ambition_set"] > (date.today() - timedelta(days=AMBITION_DAYS)).isoformat():
        return d["ambition"]
    w = week_summary(learning_dir)
    user = (f"现在的目标：{d['ambition']}\n这一周主人满意 {w['pleased']} 次、不满 {w['annoyed']} 次。\n"
            "主人不满的地方：\n" + "\n".join(f"- 主人说「{p['owner']}」，因为我答「{p['a'][:60]}」" for p in w["pains"]) +
            "\n修好的：" + "；".join(x["title"] for x in w["wins"]) +
            "\n没修好的：" + "；".join(f"{g['title']}（试了 {g['tries']} 次）" for g in w["grudges"]))
    try:
        new = str(json_object(teacher.complete(AMBITION_SYSTEM, user, json_mode=True, max_tokens=300)).get("ambition") or "").strip()
    except Exception:  # noqa: BLE001 - keep the old one
        return d["ambition"]
    if not (10 <= len(new) <= 80) or FORBIDDEN.search(new):
        return d["ambition"]
    d["ambition_history"].append({"date": date.today().isoformat(), "old": d["ambition"]})
    d["ambition"], d["ambition_set"] = new, date.today().isoformat()
    save(d, learning_dir)
    return new


# ---------------------------------------------------------------- saying it
def say(learning_dir: Path = LEARNING) -> str:
    """"你在努力什么 / 你最近有进步吗": her ambition, how the owner felt this week, what she fixed and what she
    has not let go of."""
    from learning.selfstatus import count

    d, w = load(learning_dir), week_summary(learning_dir)
    parts = [f"我给自己定的目标是：{d['ambition']}"]
    if w["annoyed"] or w["pleased"]:
        parts.append(f"这一周你夸了我{count(w['pleased'])}次，不满意{count(w['annoyed'])}次")
    if d["focus"]:
        parts.append(f"我现在正在改：{d['focus']}")
    if w["wins"]:
        parts.append(f"这周我自己修好了{count(len(w['wins']))}个毛病，比如{_short(w['wins'][-1]['title'])}")
    open_g = [g for g in d["grudges"] if g["tries"] < MAX_TRIES]
    if open_g:
        parts.append(f"还有{count(len(open_g))}个没修好，我不服气，还在换办法，比如{_short(open_g[0]['title'])}")
    return "。".join(parts) + "。"


def _short(title: str) -> str:
    """The first clause of a title, never cut mid-word ("乱承诺做不到的时间线（今晚给、…）" -> "乱承诺做不到的时间线")."""
    return re.split(r"[（(，,：:；;]", title)[0][:30]


_ASK_DRIVE = re.compile(r"(?:你|星期一).{0,6}(?:在努力什么|努力什么|想变成|想成为|目标是什么|有什么目标|进步了吗|有进步吗|有没有进步|"
                        r"心气|上进|想不想变好|想变好吗)")


def wants_drive(text: str) -> bool:
    return bool(_ASK_DRIVE.search(_PUNCT.sub("", text or "")))
