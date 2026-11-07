"""发散: from one thing she thinks of another, and of a chain of things further off.

Three parts (run/learning/assoc.json):

  graph   the things in the owner's world and how they hang together, pulled out of the day's turns each
          night by the teacher: nodes (事物、人、地方、习惯、项目、设备、话题) and edges with the kind of
          relation (因果 / 相似 / 组成 / 同一场景 / 先后 / 对立 / 用途 / 属于主人) and why. Seen again = heavier.
  wander  each night she free-associates on her own first - the board's local model, step by step, from a
          seed: what the owner talked about, something he was annoyed at, or two things that seem to have
          nothing to do with each other (connect them: the classic way to force a leap). The teacher judges
          her chain (is every step sound, does it leap somewhere new, does it end in something useful for
          the owner) and writes the better one. Good ones become ideas, with the words that should bring them
          up again. Her seed + the better chain go into training (distill/assoc.jsonl, spoken sentences, under
          her persona), so the habit of leaping moves into the next fine-tune, not only into a prompt.
  live    when the owner talks, the things he mentions light up their neighbours and any idea they trigger;
          one short line joins her context, and she may bring one connection up in a sentence when it really
          helps (context_for). An idea is offered at most once every IDEA_REST_DAYS.

    cd /userdata/agent && python3 -m learning.associate [--seeds 5] [--any-time]
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
from collections.abc import Callable
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
LEARNING = AGENT_ROOT / "run" / "learning"
LOG = logging.getLogger("associate")
FILE = "assoc.json"
TAG = "发散"
MAX_NODES = 400
MAX_IDEAS = 60
IDEA_REST_DAYS = 3
CONTEXT_CHARS = 110      # the local model's context is small (768 tokens)
RELS = ("因果", "相似", "组成", "同一场景", "先后", "对立", "用途", "属于主人")

GRAPH_SYSTEM = (
    "从语音助手「星期一」和主人的对话里，找出主人生活和工作里的「东西」以及它们之间的关系，用来帮她以后举一反三。"
    "东西：事物、人、地方、习惯、项目、设备、话题，名字要短（两到八个字），同一个东西用同一个名字。"
    f"关系只能是：{'、'.join(RELS)}，并用一句话说为什么。只要对话里看得出来、或常识上站得住的，不编。最多二十个东西、三十条关系。"
    "主人本人和星期一自己不要列成东西，提到主人就写「主人」，不要写他的名字；主人只是问过一次的事是话题，不要写成他的计划或项目。\n"
    '只输出 JSON：{"nodes": [{"name": "...", "kind": "..."}], "edges": [{"a": "...", "b": "...", "rel": "...", "why": "..."}]}'
)
WANDER_SYSTEM = (
    "现在没人跟你说话，你在自己琢磨事情。从给你的事出发，一步一步往下联想：每一步说「由什么想到什么，因为什么」，"
    "走四五步，越往后越要跳出原来的领域，想到别人想不到但说得通的东西；最后说一个对主人有用的想法、建议，"
    "或者值得去学的东西。全部用说话的句子，不用箭头、列表和符号。"
)
JUDGE_SYSTEM = (
    "你在评语音助手「星期一」的发散思维。她从一件事出发自己联想了一串，判断这一串：每一步是不是站得住（不编造），"
    "有没有真正跳出原来的领域（只是在同一件事里列步骤不算发散），最后的想法对主人有没有用、是不是新鲜。"
    "然后写一串更好的联想给她当榜样：四五步，每一步「由什么想到什么，因为什么」，要有一两次意想不到但成立的跳跃，"
    "最后落到一个对主人具体有用的想法；用说话的句子，不用箭头、列表和符号，一共不超过一百五十字。"
    "再从好的那串里提炼一个想法卡片，写清什么时候值得跟主人提（triggers：主人说到哪些词时，两到六个字一个）。"
    "关于主人的事实和他世界里的东西会给你，联想尽量和它们挂上钩。\n"
    '只输出 JSON：{"score": 1到5, "comment": "她这串怎么样", "better": "更好的一串（口语）", '
    '"idea": "一句话的想法（对主人说的口吻）", "triggers": ["..."], "useful": true或false}'
)


# ---------------------------------------------------------------- the graph
def norm(name: str) -> str:
    return re.sub(r"[\s，。,.!！?？、“”\"'（）()]", "", str(name or ""))[:16]


def load(learning_dir: Path = LEARNING) -> dict:
    try:
        d = json.loads((Path(learning_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    d.setdefault("nodes", {})
    d.setdefault("edges", [])
    d.setdefault("ideas", [])
    d.setdefault("last_turn", "")
    d.setdefault("nights", {})
    return d


def save(d: dict, learning_dir: Path = LEARNING) -> None:
    learning_dir = Path(learning_dir)
    learning_dir.mkdir(parents=True, exist_ok=True)
    if len(d["nodes"]) > MAX_NODES:   # keep the heavy and the fresh
        keep = sorted(d["nodes"], key=lambda n: (d["nodes"][n]["seen"], d["nodes"][n]["last"]), reverse=True)[:MAX_NODES]
        d["nodes"] = {k: d["nodes"][k] for k in keep}
        d["edges"] = [e for e in d["edges"] if e["a"] in d["nodes"] and e["b"] in d["nodes"]]
    d["ideas"] = d["ideas"][-MAX_IDEAS:]
    d["nights"] = dict(sorted(d["nights"].items())[-60:])
    tmp = learning_dir / f".{FILE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, learning_dir / FILE)


SELF = {"主人", "星期一", "我", "你", "她"}   # the owner and she herself are the centre, not things to wander from


def merge(d: dict, nodes: list[dict], edges: list[dict], day: str, skip: set[str] | None = None) -> tuple[int, int]:
    """-> (new nodes, new edges). skip: names never kept (what the owner is called)."""
    skip = SELF | (skip or set())
    new_n = new_e = 0
    for n in nodes:
        name = norm(n.get("name"))
        if len(name) < 2 or name in skip:
            continue
        cur = d["nodes"].get(name)
        if cur is None:
            d["nodes"][name] = {"kind": str(n.get("kind") or "")[:8], "seen": 1, "last": day}
            new_n += 1
        else:
            cur["seen"] += 1
            cur["last"] = day
    for e in edges:
        a, b, rel = norm(e.get("a")), norm(e.get("b")), str(e.get("rel") or "")
        if len(a) < 2 or len(b) < 2 or a == b or rel not in RELS or a in skip or b in skip:
            continue
        for x in (a, b):
            d["nodes"].setdefault(x, {"kind": "", "seen": 1, "last": day})
        old = next((x for x in d["edges"] if {x["a"], x["b"]} == {a, b} and x["rel"] == rel), None)
        if old:
            old["w"] += 1
            old["date"] = day
        else:
            d["edges"].append({"a": a, "b": b, "rel": rel, "why": str(e.get("why") or "")[:60], "w": 1, "date": day})
            new_e += 1
    return new_n, new_e


def neighbors(d: dict, name: str, k: int = 3) -> list[dict]:
    out = [e for e in d["edges"] if name in (e["a"], e["b"])]
    out.sort(key=lambda e: (e["w"], e["date"]), reverse=True)
    return out[:k]


def mentioned(d: dict, text: str, k: int = 3) -> list[str]:
    t = norm(text)
    hits = [n for n in d["nodes"] if len(n) >= 2 and n in t]
    hits.sort(key=len, reverse=True)
    out: list[str] = []
    for n in hits:   # "家里网络" covers "网络"
        if not any(n in o for o in out):
            out.append(n)
    return out[:k]


# ---------------------------------------------------------------- live
def context_for(text: str, learning_dir: Path = LEARNING, today: str | None = None, mark: bool = True) -> str:
    """One short line for her context: what the things he mentions connect to, and an idea they trigger."""
    try:
        d = load(learning_dir)
    except Exception:  # noqa: BLE001 - never in the way of an answer
        return ""
    today = today or dt.date.today().isoformat()
    parts: list[str] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for n in mentioned(d, text, 2):
        for e in neighbors(d, n, 2):
            key = (min(e["a"], e["b"]), max(e["a"], e["b"]), e["rel"])
            if key in seen_edges:   # both ends mentioned: say the link once
                continue
            seen_edges.add(key)
            other = e["b"] if e["a"] == n else e["a"]
            parts.append(f"{n}和{other}（{e['rel']}：{e['why'][:24]}）")
    rest = (dt.date.fromisoformat(today) - dt.timedelta(days=IDEA_REST_DAYS)).isoformat()
    t = norm(text)
    idea = next((i for i in reversed(d["ideas"])
                 if i.get("offered", "") <= rest and any(norm(w) and norm(w) in t for w in i.get("triggers", []))), None)
    idea_line = f"你之前琢磨到的：{idea['idea']}" if idea else ""
    room = max(40, CONTEXT_CHARS - len(idea_line))   # the idea is never cut; the links make room, whole ones only
    links, used = [], len("能联想到的：")
    for p in dict.fromkeys(parts):
        if used + len(p) + 1 > room:
            break
        links.append(p)
        used += len(p) + 1
    lines = (["能联想到的：" + "；".join(links)] if links else []) + ([idea_line] if idea_line else [])
    if idea and mark:
        idea["offered"] = today
        save(d, learning_dir)
    if not lines:
        return ""
    return "\n".join(lines) + "\n（真有帮助才用一句话顺带提起，别硬扯，先答主人问的）"


# ---------------------------------------------------------------- nightly
def read_turns(learning_dir: Path, since: str) -> list[dict]:
    out = []
    for f in sorted((learning_dir / "turns").glob("*.jsonl")) if (learning_dir / "turns").is_dir() else []:
        if f.stem < since[:10]:
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if t.get("time", "") > since:
                out.append(t)
    return out


def extract(teacher, turns: list[dict], facts: list[str]) -> dict:
    from learning.agent import json_object

    convo = "\n".join(f"主人：{t.get('q', '')[:80]}\n星期一：{t.get('a', '')[:100]}" for t in turns[-60:])
    user = ("关于主人：" + "；".join(facts) + "\n\n" if facts else "") + f"对话：\n{convo}"
    return json_object(teacher.complete(GRAPH_SYSTEM, user, json_mode=True, max_tokens=2000, timeout_s=120))


def seeds(d: dict, turns: list[dict], learning_dir: Path, n: int, rng: random.Random) -> list[str]:
    """What to wander from tonight: today's things, his annoyances, and one bridge between far-apart things."""
    from learning import drive

    today = [k for k, v in d["nodes"].items() if v["last"] == dt.date.today().isoformat()]
    today.sort(key=lambda k: d["nodes"][k]["seen"], reverse=True)
    pains = [p for v in drive.load(learning_dir)["days"].values() for p in v["pains"]][-5:]
    pain = None
    if pains:
        p = rng.choice(pains)
        pain = f"主人不满意的那次：他问「{p['q'][:30]}」，我答「{p['a'][:40]}」"
    bridge = None
    if today:   # two things with no edge between them: older ones if there are, else today's far ends
        a = rng.choice(today[:5])
        linked = {e["a"] for e in d["edges"] if e["b"] == a} | {e["b"] for e in d["edges"] if e["a"] == a} | {a}
        far = [k for k in d["nodes"] if k not in today and k not in linked] or [k for k in today if k not in linked]
        if far:
            bridge = f"把两件看起来不相干的事连起来：「{a}」和「{rng.choice(far)}」"
    slots = max(1, n - bool(pain) - bool(bridge))
    out = [f"主人今天聊到的「{k}」" for k in today[:slots]] + [x for x in (pain, bridge) if x]
    if not out:
        out = [f"主人说过的「{t.get('q', '')[:30]}」" for t in rng.sample(turns, min(len(turns), n))] if turns else []
    return out[:n]


def wander(teacher, ask_local: Callable[[str, str], str], seed: str, *, persona: str, facts: list[str], d: dict,
           learning_dir: Path = LEARNING) -> dict:
    """Her own chain from `seed`, judged; a kept one becomes an idea and a training sample."""
    from learning.agent import json_object
    from learning.lora_prep import speakable

    mine = ask_local(persona + "\n\n" + WANDER_SYSTEM, f"从这件事开始联想：{seed}")
    world = "；".join(f"{e['a']}—{e['b']}（{e['rel']}）" for e in sorted(d["edges"], key=lambda e: -e["w"])[:15])
    user = (("关于主人：" + "；".join(facts) + "\n") + (f"他世界里的东西：{world}\n" if world else "")
            + f"\n出发点：{seed}\n\n她的联想：\n{mine}")
    j = json_object(teacher.complete(JUDGE_SYSTEM, user, json_mode=True, max_tokens=1200, timeout_s=120))
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed, "mine": mine[:600], "score": int(j.get("score") or 0),
           "comment": str(j.get("comment") or "")[:200], "better": str(j.get("better") or "").strip()[:400],
           "idea": str(j.get("idea") or "").strip()[:120], "useful": bool(j.get("useful")), "kept": False}
    triggers = [norm(w) for w in j.get("triggers") or [] if 2 <= len(norm(w)) <= 6][:6]
    if rec["useful"] and rec["idea"] and triggers:
        if not any(norm(i["idea"])[:20] == norm(rec["idea"])[:20] for i in d["ideas"]):
            d["ideas"].append({"idea": rec["idea"], "triggers": triggers, "seed": seed[:60], "date": dt.date.today().isoformat(),
                               "offered": "", "said": 0})
            rec["kept"] = True
    if rec["better"] and speakable(rec["better"]) and not re.search(r"→|->|=>", rec["better"]) and len(rec["better"]) <= 220:
        (learning_dir / "distill").mkdir(parents=True, exist_ok=True)
        with (learning_dir / "distill" / "assoc.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"messages": [{"role": "system", "content": persona + "\n\n" + WANDER_SYSTEM},
                                             {"role": "user", "content": f"从这件事开始联想：{seed}"},
                                             {"role": "assistant", "content": rec["better"]}], "source": "assoc"},
                               ensure_ascii=False) + "\n")
    out = learning_dir / "assoc_log.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def nightly(teacher, ask_local: Callable[[str, str], str], *, persona: str, facts: list[str],
            learning_dir: Path = LEARNING, n_seeds: int = 5, idle: Callable[[], bool] = lambda: True,
            rng: random.Random | None = None, owner_names: set[str] | None = None) -> dict:
    rng = rng or random.Random()
    d = load(learning_dir)
    for gone in SELF | (owner_names or set()):   # an older graph may still hold them
        if d["nodes"].pop(gone, None) is not None:
            d["edges"] = [e for e in d["edges"] if gone not in (e["a"], e["b"])]
    day = dt.date.today().isoformat()
    since = d["last_turn"] or (dt.datetime.now() - dt.timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
    turns = read_turns(learning_dir, since)
    res = {"turns": len(turns), "new_nodes": 0, "new_edges": 0, "wandered": 0, "ideas": 0, "scores": []}
    if turns:
        try:
            g = extract(teacher, turns, facts)
            res["new_nodes"], res["new_edges"] = merge(d, g.get("nodes") or [], g.get("edges") or [], day, skip=owner_names)
            d["last_turn"] = max(t.get("time", "") for t in turns)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("graph extraction failed: %s", exc)
    save(d, learning_dir)
    for seed in seeds(d, turns, learning_dir, n_seeds, rng):
        if not idle():
            res["stopped"] = True
            break
        try:
            r = wander(teacher, ask_local, seed, persona=persona, facts=facts, d=d, learning_dir=learning_dir)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("wander failed: %s", exc)
            continue
        res["wandered"] += 1
        res["ideas"] += int(r["kept"])
        res["scores"].append(r["score"])
        save(d, learning_dir)
    d = load(learning_dir)
    d["nights"][day] = {k: res[k] for k in ("turns", "new_nodes", "new_edges", "wandered", "ideas", "scores")}
    save(d, learning_dir)
    return res


# ---------------------------------------------------------------- saying it
def say_ideas(learning_dir: Path = LEARNING, n: int = 2) -> str:
    """"你最近想到什么了": the freshest ideas she has not brought up yet."""
    d = load(learning_dir)
    fresh = [i for i in reversed(d["ideas"]) if not i.get("said")][:n] or list(reversed(d["ideas"]))[:1]
    if not fresh:
        return "最近还没琢磨出什么新东西，今晚我再好好想想。"
    for i in fresh:
        i["said"] = int(i.get("said", 0)) + 1
    save(d, learning_dir)
    return "我最近琢磨到" + ("两件事。" if len(fresh) == 2 else "一件事。") + "".join(
        ("一是" if k == 0 and len(fresh) == 2 else "二是" if k == 1 else "") + i["idea"].rstrip("。") + "。" for k, i in enumerate(fresh))


_ASK_IDEAS = re.compile(r"(?:你|星期一).{0,6}(?:想到什么|琢磨什么|琢磨到什么|有什么想法|有什么新想法|有什么灵感|在想什么|联想到什么)")


def wants_ideas(text: str) -> bool:
    return bool(_ASK_IDEAS.search(re.sub(r"\s", "", text or "")))


def owner_names(run_dir: Path) -> set[str]:
    """What the owner is called (the title he chose, e.g. his name): never a node in the graph."""
    try:
        title = str(json.loads((run_dir / "user_settings.json").read_text(encoding="utf-8")).get("user_title") or "")
    except (OSError, ValueError):
        title = ""
    return {norm(title)} if norm(title) else set()


def local_asker() -> Callable[[str, str], str]:
    from llm.client import llm_chat, llm_clear_history

    def ask(system: str, user: str) -> str:
        llm_clear_history()
        p = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        return re.sub(r"(?s)<think>.*?</think>", "", str(llm_chat(p, max_new_tokens=260).get("text", ""))).strip()
    return ask


def main() -> None:
    ap = argparse.ArgumentParser(description="she builds her association graph and wanders from a few seeds")
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--any-time", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.environ.setdefault("MONDAY_USAGE_TAG", TAG)
    from learning import exam, usage
    from learning.practice import IDLE_MIN, last_activity
    from learning.reflect import build_runtime

    if not a.any_time and usage.peak_blocked():
        LOG.info("the teacher's peak hours: not now")
        return
    rt = build_runtime(a.config)
    run_dir = AGENT_ROOT / "run"
    facts = rt.store.facts(limit=8)
    res = nightly(rt.teacher, local_asker(), persona=exam.persona(a.config), facts=facts, learning_dir=rt.learning_dir,
                  n_seeds=a.seeds, idle=lambda: time.time() - last_activity(run_dir) > IDLE_MIN * 60,
                  owner_names=owner_names(run_dir))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
