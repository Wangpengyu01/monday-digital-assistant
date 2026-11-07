"""Build a static, sanitized copy of ui/showcase.html for GitHub Pages.

    python -I tools/build_showcase_static.py --board http://10.0.0.177 --out <dir>

Only the fields the page shows are kept (counts, service states, lesson topics, the self-review, LoRA
progress). IPs, user names, hostnames, paths and anything key-like are scrubbed, and the build fails
if the final HTML still contains something that looks private.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "ui"
IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
SCRUB = [(IP, "某台设备"), (re.compile(r"/(?:userdata|home|srv|etc)\b(?:/[\w./-]*)?"), "数据盘"),
         (re.compile(r"\b(?:pengyu|youyeetoo@|gpu@|ikun|gp@)\w*", re.I), "某账号")]
FORBIDDEN = [IP, re.compile(r"sk-[A-Za-z0-9]{12,}"), re.compile(r"PRIVATE KEY"), re.compile(r"pengyu|ikun|/userdata|/home/", re.I),
             re.compile(r"(?:密码|password)\s*[:：=是为]\s*\S", re.I)]


def private_names() -> list[str]:
    """Names that must never be published (the owner's real name...). Kept out of the repository on purpose:
    from MONDAY_PRIVATE_NAMES (comma-separated) and ~/.monday_private_names (one per line), never from here."""
    import os
    names = [n.strip() for n in os.environ.get("MONDAY_PRIVATE_NAMES", "").split(",") if n.strip()]
    f = Path.home() / ".monday_private_names"
    if f.is_file():
        names += [n.strip() for n in f.read_text(encoding="utf-8").splitlines() if n.strip()]
    return names


def private_patterns() -> list[re.Pattern]:
    return [re.compile(re.escape(n)) for n in private_names()]


def clean(x):
    if isinstance(x, str):
        for pat, rep in SCRUB + [(p, "主人") for p in private_patterns()]:
            x = pat.sub(rep, x)
        return x
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x


def fetch(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def snapshot(base: str) -> dict:
    mem, sysi, school, prog, proj = (fetch(base, p) for p in ("/api/memory?limit=1", "/api/system", "/api/school",
                                                                "/api/progress", "/api/projects"))
    lora = prog.get("lora", {})
    drive = proj.get("drive") or {}
    return clean({
        "taken": time.strftime("%Y-%m-%d %H:%M"),
        "/api/memory": {"stats": mem["stats"]},
        "/api/system": {"uptime_s": sysi["uptime_s"], "cpu_percent": sysi["cpu_percent"], "temp_c": sysi["temp_c"],
                        "services": [{"label": s["label"], "state": s["state"]} for s in sysi["services"]]},
        "/api/school": {"syllabus": school["syllabus"], "index": school["index"], "sessions": school["sessions"],
                        "mastered": school["mastered"], "review": school.get("review", {}),
                        "history": [{k: h.get(k) for k in ("topic", "kind", "passed", "total", "mastered")} for h in school["history"][:14]]},
        "/api/progress": {"lora": {"stage": lora.get("stage", ""), "steps": [{"name": s["name"], "state": s["state"]} for s in lora.get("steps", [])],
                                   "versions": lora.get("versions", [])}},
        # her projects and her drive: titles, status and counts only - never the owner's words (pains), the
        # steps, the notes or a proposal's diff
        "/api/projects": {"projects": [{k: p.get(k) for k in ("goal", "status", "steps", "spent", "kind")}
                                       for p in proj.get("projects", [])[:6]],
                          "drive": {k: drive.get(k) for k in ("ambition", "focus", "annoyed", "pleased", "wins", "grudges")}
                          if drive else None},
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="http://10.0.0.177")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    html = (UI / "showcase.html").read_text(encoding="utf-8")
    snap = json.dumps(snapshot(a.board), ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("<script>\nconst $ =", f"<script>window.SNAPSHOT = {snap};</script>\n<script>\nconst $ =", 1)
    html = html.replace('<link rel="icon" href="/favicon.svg">', "")
    html = re.sub(r'<footer id="foot">.*?</footer>',
                  '<footer id="foot">数据是板子上的一次快照；真机上的控制台只在局域网内可用。</footer>', html, flags=re.S)
    assert "window.SNAPSHOT" in html
    if not private_names():
        raise SystemExit("refusing to publish: no private names configured (~/.monday_private_names)")
    for pat in FORBIDDEN + private_patterns():
        m = pat.search(html)
        if m:
            raise SystemExit(f"refusing to publish: found {m.group(0)!r}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8", newline="\n")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {out / 'index.html'} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
