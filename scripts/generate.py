#!/usr/bin/env python3
"""Write the daily prompt files for the Stratos game.

Run once a day by .github/workflows/daily.yml. For today and the next two
days (UTC) it asks the OpenAI API for seven prompts with ranked answers and
saves them as daily/YYYY-MM-DD.json. Days that already have a file are left
alone, so every player gets the same seven prompts for a given date, and
players in time zones ahead of UTC always find their date ready.

Environment:
  OPENAI_API_KEY   required
  OPENAI_MODEL     optional; if unset the script picks a small model that the
                   key has access to

  python scripts/generate.py --selftest   checks the validator without the API
"""

import datetime
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "daily"
API = "https://api.openai.com/v1"
DAYS_AHEAD = 2
ATTEMPTS = 3
PREFERRED_MODELS = ["gpt-6-luna", "gpt-5.6-luna", "gpt-5.4-mini", "gpt-4.1-mini", "gpt-4o-mini"]

PROMPT_OK = re.compile(r"^[A-Za-z0-9 ',.?!-]{4,60}$")
NAME_OK = re.compile(r"^[A-Za-z0-9 '-]{1,28}$")

INSTRUCTIONS = """You write the daily round for a word game.

The player sees a short category prompt (for example "A fruit" or "Something
you take camping") and types one answer. Common answers score little; rare
but valid answers score a lot.

Write exactly 7 prompts in this order: 3 easy, 2 medium, 2 hard.
- easy: a broad everyday category that anyone could answer ten ways.
- medium: narrower, needs a moment of thought.
- hard: specific enough that most people only know three or four answers.

Rules for prompts:
- Family friendly, no brands, no living people, no politics or religion.
- Each prompt is at most 45 characters and uses only letters, digits,
  spaces, apostrophes, commas and hyphens.
- Cover seven different subjects.
- Do not reuse or lightly reword any prompt in the AVOID list.

For easy prompts give 20 to 26 distinct valid answers, for medium 16 to 22,
and for hard 12 to 18. Pick categories wide enough to meet these counts.
Include at least four genuinely uncommon answers below 2 percent per
prompt. Do not pad the list with near-duplicates or borderline answers.
Use aliases for regional spellings and equivalent names, not for distinct answers.
For each answer give:
- "names": the main spelling first, then other spellings or synonyms that
  should count as the same answer. Lower case, letters, digits, spaces,
  hyphens and apostrophes only, at most 28 characters each.
- "share": your estimate of the percentage of people who would give this
  answer first, between 0.3 and 45. Include the obvious answers as well as
  several obscure ones. Shares within a prompt must add up to at most 100.

Reply with JSON only, in exactly this shape:
{"prompts":[{"text":"...","difficulty":"easy","answers":[{"names":["..."],"share":12.5}]}]}
"""


def http(method, path, body=None):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        sys.exit("OPENAI_API_KEY is not set. Add it under Settings > Secrets and variables > Actions.")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def pick_model():
    chosen = os.environ.get("OPENAI_MODEL", "").strip()
    if chosen:
        return chosen
    ids = {m.get("id", "") for m in http("GET", "/models").get("data", [])}
    for name in PREFERRED_MODELS:
        if name in ids:
            return name
    small = sorted(i for i in ids if i.startswith("gpt-") and ("mini" in i or "luna" in i))
    if small:
        return small[-1]
    sys.exit("Could not choose a model automatically. Set the OPENAI_MODEL repository variable.")


def ask(model, text):
    """Try the Chat Completions endpoint, then the Responses endpoint."""
    try:
        out = http("POST", "/chat/completions",
                   {"model": model, "messages": [{"role": "user", "content": text}]})
        return out["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as err:
        if err.code not in (400, 404):
            raise
    out = http("POST", "/responses", {"model": model, "input": text})
    if isinstance(out.get("output_text"), str) and out["output_text"]:
        return out["output_text"]
    parts = []
    for item in out.get("output", []):
        for piece in item.get("content", []) or []:
            if isinstance(piece.get("text"), str):
                parts.append(piece["text"])
    return "".join(parts)


def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def validate(obj):
    """Return a cleaned list of seven prompts or raise ValueError."""
    prompts = obj.get("prompts") if isinstance(obj, dict) else None
    if not isinstance(prompts, list) or len(prompts) != 7:
        raise ValueError("expected exactly 7 prompts")
    cleaned, seen_text = [], set()
    for p in prompts:
        text = " ".join(str(p.get("text", "")).split())
        if not PROMPT_OK.match(text):
            raise ValueError("bad prompt text: %r" % text)
        if text.lower() in seen_text:
            raise ValueError("duplicate prompt: %r" % text)
        seen_text.add(text.lower())
        difficulty = str(p.get("difficulty", "")).lower()
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("bad difficulty for %r" % text)
        answers, used = [], set()
        for a in p.get("answers", []) or []:
            try:
                share = float(a.get("share"))
            except (TypeError, ValueError):
                continue
            if not 0.3 <= share <= 45:
                continue
            names = []
            for n in a.get("names", []) or []:
                n = " ".join(str(n).lower().split())
                if NAME_OK.match(n) and n not in used:
                    used.add(n)
                    names.append(n)
            if names:
                answers.append({"names": names, "share": round(share, 1)})
        minimum = {"easy": 20, "medium": 16, "hard": 12}[difficulty]
        if len(answers) < minimum:
            raise ValueError("fewer than %d usable answers for %r" % (minimum, text))
        if sum(a["share"] < 2 for a in answers) < 4:
            raise ValueError("fewer than four uncommon answers for %r" % text)
        if sum(round(a["share"] * 10) for a in answers) > 1000:
            raise ValueError("answer shares exceed 100 percent for %r" % text)
        answers.sort(key=lambda a: -a["share"])
        cleaned.append({"text": text, "difficulty": difficulty, "answers": answers})
    counts = {d: sum(1 for c in cleaned if c["difficulty"] == d) for d in ("easy", "medium", "hard")}
    if counts != {"easy": 3, "medium": 2, "hard": 2}:
        raise ValueError("difficulty mix is off: %s" % counts)
    order = {"easy": 0, "medium": 1, "hard": 2}
    cleaned.sort(key=lambda c: order[c["difficulty"]])
    return cleaned


def recent_prompts(limit=40):
    texts = []
    for path in sorted(OUT.glob("20*.json"))[-limit:]:
        try:
            texts += [p["text"] for p in json.loads(path.read_text())["prompts"]]
        except (ValueError, KeyError, OSError):
            continue
    return texts


def generate(date, model):
    avoid = recent_prompts()
    request = INSTRUCTIONS + "\nDate: " + date + "\nAVOID:\n" + ("\n".join("- " + t for t in avoid) or "- (none yet)")
    last = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            prompts = validate(extract_json(ask(model, request)))
            lowered = {t.lower() for t in avoid}
            if any(p["text"].lower() in lowered for p in prompts):
                raise ValueError("repeated a recent prompt")
            return {"date": date, "model": model, "prompts": prompts}
        except (ValueError, KeyError, IndexError, TypeError) as err:
            last = err
            print("  attempt %d rejected: %s" % (attempt, err))
    raise SystemExit("Gave up on %s after %d attempts: %s" % (date, ATTEMPTS, last))


def selftest():
    sample = {"prompts": [
        {"text": "Prompt number %d" % i, "difficulty": d,
         "answers": [{"names": ["answer %d" % k, "alt %d" % k],
                      "share": 30 if k == 0 else 1.5}
                     for k in range(20)]}
        for i, d in enumerate(["hard", "easy", "easy", "medium", "easy", "hard", "medium"])
    ]}
    good = validate(extract_json("noise " + json.dumps(sample) + " noise"))
    assert [p["difficulty"] for p in good] == ["easy"] * 3 + ["medium"] * 2 + ["hard"] * 2
    assert good[0]["answers"][0]["share"] == 30.0
    for broken in (
        {"prompts": sample["prompts"][:6]},
        {"prompts": [dict(sample["prompts"][0], text="Bad <b>text</b>")] + sample["prompts"][1:]},
        {"prompts": [dict(sample["prompts"][0], answers=[])] + sample["prompts"][1:]},
        {"prompts": [dict(p, difficulty="easy") for p in sample["prompts"]]},
    ):
        try:
            validate(broken)
        except ValueError:
            continue
        raise AssertionError("validator accepted a broken sample")
    print("selftest passed")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    OUT.mkdir(exist_ok=True)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    model = None
    for offset in range(DAYS_AHEAD + 1):
        date = (today + datetime.timedelta(days=offset)).isoformat()
        path = OUT / (date + ".json")
        if path.exists():
            print(date, "already written")
            continue
        model = model or pick_model()
        print(date, "generating with", model)
        path.write_text(json.dumps(generate(date, model), indent=1) + "\n")
        print(date, "saved")


if __name__ == "__main__":
    main()
