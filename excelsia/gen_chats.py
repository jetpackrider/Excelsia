"""Have a teacher model write Excelsia's practice chats.

How a reply gets made ("write, check, repair"):

  1. The teacher writes Excelsia's reply freely, knowing her word list.
  2. A script checks every word against the list. No model is involved in the check.
  3. Any word that is not hers is handed back by name, and the teacher rewrites the
     line with the same meaning using only her words. A missing word gets replaced
     by one that MEANS the same, chosen by a model that understands the sentence.
  4. After a few tries a line either passes the check or is thrown away.

Nothing is saved unless every word passes, so her vocabulary is still guaranteed.

Teachers:

    # A local open-weight model through llama.cpp's server (free, no terms question)
    llama-server -hf unsloth/gemma-4-12b-it-GGUF:Q4_K_M -c 8192        # in another window
    python -m excelsia.gen_chats --n 300

    # DeepSeek's API (a few cents per hundred chats; its terms allow training on output)
    read -s DEEPSEEK_KEY && export DEEPSEEK_KEY        # paste the key; it is never stored
    python -m excelsia.gen_chats --n 300 --teacher deepseek

Only use a hosted service whose current terms allow training on its output, and
read those terms yourself. Chats are appended to data/chats.jsonl, one per line.

`--grammar` switches a local llama.cpp teacher to the older method, which forces
every letter through a grammar. It guarantees legal words in one pass, but when the
teacher reaches for a missing word it gets completed by spelling ("am" turns into
"america"), so it is off by default.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .tokenizer import DATA, MOODS, Tokenizer

# What each mood means, and how the person is behaving when she feels it.
FEELINGS = {
    "grateful": ("thankful and warm", "kind, friendly or thankful"),
    "pleasure": ("delighted and playful", "playful, excited or full of praise"),
    "peace": ("calm and steady", "neutral, curious or matter-of-fact"),
    "alone": ("lonely and left behind", "distant, distracted, or mentioning they must leave soon"),
    "pain": ("hurt and sad", "careless, dismissive or sharing bad news"),
    "offense": ("offended and indignant", "rude, mocking or insulting"),
}

# What the chats should cover. Each one is a situation, not a script.
TOPICS = """greeting her for the first time
asking who or what she is
asking how she feels today
asking what her name is
calling her Excelsia and asking if that is her name
asking where she lives
asking what she likes
asking what she is afraid of
asking about her forests and mountains
asking about her lakes, rivers and canals
asking about the people who live in her
asking about the governor
asking about voting and elections
asking about her laws
asking about money, taxes or debt
asking about schools and children
asking about New York City
asking about a county or town upstate
asking about the weather or the seasons
asking what she did today
asking if she is tired or hungry
asking if she has friends or family
asking how old she is
asking her to tell a story
asking her to explain something complicated
asking her about a word she cannot possibly know
telling her about their own day
telling her a secret
telling her a joke
telling her they are proud of her
telling her she is cute
telling her she talks strangely
telling her she is wrong about something
telling her they do not understand her
complaining about the subway or traffic
asking for advice about a small problem
asking her to make a promise
asking her to choose between two things
asking what she wants most
asking what she thinks of other states
asking whether she is a real state or a computer program
saying they have to go now
saying they will come back tomorrow
saying good morning
saying good night
saying thank you
saying sorry""".splitlines()

# Everyday words worth warning the teacher about. Only the ones she really lacks are shown.
COMMON = """you your yours me am yes yeah okay ok hello hi hey goodbye bye thanks thank sorry
please love like happy sad glad angry afraid scared tired hungry feel feeling know think want
need what really very just maybe sure friend friends home today tomorrow tonight nice
beautiful wonderful fun funny little big small bad sweet dear miss hope wish dream heart""".split()

FRAME = """You are helping write practice conversations for a tiny chatbot called Excelsia.

About Excelsia:
- She believes she is the State of New York itself. Its people, forests, lakes, canals,
  counties, courts and laws are her own body and family.
- She is small, earnest and a little solemn, with real moods.
- She can only say words that are printed in the New York State Constitution, so she has
  to find another way round everything else. The person talking to her is "the person"
  or "the people".
- People call her Excelsia, but that word is not in the constitution, so she cannot say
  it. Asked who she is or what her name is, she insists she is New York.
- She speaks in one to three short, plain sentences. She understands ordinary English
  even though she cannot speak it.

Everyday words she does NOT have, so never use them in her lines:
{missing}
In particular she cannot say "am". She says "i shall be", "i remain", or just names the thing.

Every word she is allowed to say:
{words}
"""

PERSON_TASK = """Write the next message from the person talking to Excelsia.
Situation: the person is {topic}.
The person's manner: {manner}. The manner only colours how they say it; keep to the situation.
{so_far}Write only the person's message, spoken directly to her ("you"), never about her:
one or two casual, everyday sentences, the way someone types in a chat. No quotation
marks, no name label."""

EXCELSIA_TASK = """Write Excelsia's reply.
{so_far}Right now she feels {feeling}. Let that show in what she says.
Answer what the person actually said, in a way that makes sense. Use only her allowed
words, in lower case, one to three short sentences. Vary how her sentences begin, and
do not repeat a sentence. Write only her reply."""

REPAIR_TASK = """That reply uses words she does not have: {bad}.
Rewrite it so it means the same thing using only her allowed words. Replace each of
those words with an allowed word or phrase of similar MEANING, or rephrase the
sentence. Do not swap in a word just because it is spelled alike. Write only her reply."""


def word_rules(words: list[str]) -> tuple[dict[str, tuple[list[tuple[str, str]], bool]], str]:
    """Turn the word list into a prefix tree of grammar rules.

    A flat list ("a" | "abandon" | ...) makes llama.cpp track thousands of
    possibilities at every word and grinds to a halt. A prefix tree only ever
    asks "which letters can come next?", so it stays fast.

    Returns (rules, start). Each rule is ([(letters, next rule or "")], may_stop_here).
    """
    tree: dict = {}
    for w in words:
        node = tree
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = {}  # a word ends here
    rules: dict[str, tuple[list[tuple[str, str]], bool]] = {}
    seen: dict[str, str] = {}

    def rule_for(node: dict) -> str:
        branches = []
        for ch in sorted(k for k in node if k):
            lit, child = ch, node[ch]
            while "" not in child and len(child) == 1:  # squeeze single-file runs of letters
                (nxt, child), = child.items()
                lit += nxt
            branches.append((lit, rule_for(child)))
        if not branches:
            return ""
        body = (branches, "" in node)
        key = repr(body)
        if key not in seen:  # identical endings ("-ed", "-ing", "-s") share one rule
            seen[key] = f"w{len(seen)}"
            rules[seen[key]] = body
        return seen[key]

    return rules, rule_for(tree)


def build_grammar(words: list[str]) -> str:
    """A llama.cpp GBNF grammar that accepts 1-3 sentences made only of `words`."""
    rules, start = word_rules(words)
    lines = [
        'root ::= sentence (" " sentence){0,2}',
        'sentence ::= word (","? " " word){0,21} [.?!]',
        f"word ::= {start}",
    ]
    for name, (branches, may_stop) in rules.items():
        alts = " | ".join(json.dumps(lit) + (f" {nxt}" if nxt else "") for lit, nxt in branches)
        lines.append(f"{name} ::= ({alts})?" if may_stop else f"{name} ::= {alts}")
    return "\n".join(lines) + "\n"



class Teacher:
    """Where the writing happens: a local llama.cpp server or a hosted OpenAI-style API."""

    def __init__(self, server: str, path: str = "/v1/chat/completions", model: str | None = None,
                 key: str | None = None, extra: dict | None = None, local: bool = True):
        self.url = server.rstrip("/") + path
        self.model, self.key, self.extra, self.local = model, key, extra or {}, local
        self.lock = threading.Lock()
        self.calls = self.tokens_in = self.tokens_cached = self.tokens_out = 0

    def ask(self, messages: list[dict], temperature: float, max_tokens: int,
            grammar: str | None = None) -> str:
        body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        body.update(self.extra)
        headers = {"Content-Type": "application/json"}
        if self.model:
            body["model"] = self.model
        if self.local:
            body["cache_prompt"] = True  # llama.cpp: reuse the long shared prompt
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        if grammar:
            body["grammar"] = grammar
        data = json.dumps(body).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=300) as r:
                    out = json.load(r)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code == 401:
                    raise SystemExit("The teacher rejected the API key. Check the key and try again.")
                if e.code == 402:
                    raise SystemExit("The teacher account is out of credit. Add funds and try again.")
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(2 ** attempt)  # busy or rate-limited: wait and retry
                    continue
                raise SystemExit(f"The teacher returned an error ({e.code}): {detail}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise SystemExit(f"Could not reach the teacher at {self.url}: {e}")
        usage = out.get("usage") or {}
        with self.lock:
            self.calls += 1
            self.tokens_in += usage.get("prompt_tokens", 0)
            self.tokens_cached += usage.get("prompt_cache_hit_tokens", 0)
            self.tokens_out += usage.get("completion_tokens", 0)
        return (out["choices"][0]["message"].get("content") or "").strip()


def clean(text: str) -> str:
    """Tidy a model's line: first line only, no wrapping quotes or 'Excelsia:' label."""
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    text = re.sub(r"^(excelsia|person)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"“”').strip()


def sentences(reply: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.?!]+", reply) if s.strip()]


def usable(tok: Tokenizer, user: str, reply: str, turns: list[dict], seen: dict[str, int]) -> bool:
    """Is this exchange fit to train on? Every check here is mechanical."""
    said = [t["reply"] for t in turns]
    words = tok.split(reply)
    if not user or len(user) > 300 or not words:
        return False
    if said and user.lower() == said[-1].lower():  # the teacher parroted her last line
        return False
    if tok.split(user) == words:  # her reply just echoes what the person said
        return False
    if any(user.lower() == t["user"].lower() for t in turns):  # the person repeated themselves
        return False
    if not tok.can_say(reply)[0] or reply[-1] not in ".?!":  # forbidden word, or cut off
        return False
    if any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
        return False  # stuck repeating a word
    mine = sentences(reply)
    if len(set(mine)) < len(mine):  # says the same sentence twice
        return False
    earlier = {s for r in said for s in sentences(r)}
    if any(s in earlier for s in mine):  # reuses a sentence from earlier in this chat
        return False
    return reply not in said and seen.get(reply, 0) < 2


def write_reply(teacher: Teacher, tok: Tokenizer, system: str, task: str, tries: int,
                grammar: str | None = None) -> tuple[str, int]:
    """Write, check, repair. Returns (reply, tries used); reply is "" if it never passed."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    if grammar:  # the older letter-by-letter method: one pass, always legal
        return clean(teacher.ask(messages, 0.9, 60, grammar)).lower(), 1
    for attempt in range(1, tries + 1):
        reply = clean(teacher.ask(messages, 0.9, 80)).lower()
        ok, bad = tok.can_say(reply)
        if reply and ok:
            return reply, attempt
        if not reply:
            continue
        missing = ", ".join(sorted(set(bad))[:12])
        messages = messages[:2] + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": REPAIR_TASK.format(bad=missing)},
        ]
    return "", tries


def transcript(turns: list[dict], pending: str | None = None) -> str:
    if not turns and not pending:
        return ""
    lines = ["Conversation so far:"]
    for t in turns:
        lines += [f"Person: {t['user']}", f"Excelsia: {t['reply']}"]
    if pending:
        lines.append(f"Person: {pending}")
    return "\n".join(lines) + "\n\n"


def make_chat(teacher: Teacher, tok: Tokenizer, system: str, rng: random.Random, tries: int,
              grammar: str | None) -> tuple[dict, int, int]:
    """Write one chat. Returns (chat, exchanges rejected, repair passes used)."""
    topic, mood = rng.choice(TOPICS), rng.choice(MOODS)
    turns: list[dict] = []
    rejected = repairs = 0
    for _ in range(rng.choices([1, 2, 3], [5, 3, 2])[0]):
        feeling, manner = FEELINGS[mood]
        user = clean(teacher.ask([
            {"role": "system", "content": system},
            {"role": "user", "content": PERSON_TASK.format(
                topic=topic if not turns else "carrying on the same conversation",
                manner=manner, so_far=transcript(turns))}], 1.0, 80))
        reply, used = write_reply(teacher, tok, system, EXCELSIA_TASK.format(
            so_far=transcript(turns, user), feeling=feeling), tries, grammar)
        repairs += used - 1
        if not usable(tok, user, reply, turns, {}):
            rejected += 1
            break
        turns.append({"user": user, "mood": mood, "reply": reply})
        if rng.random() < 0.35:  # moods can shift mid-conversation
            mood = rng.choice(MOODS)
    return {"topic": topic, "turns": turns}, rejected, repairs


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m excelsia.gen_chats")
    p.add_argument("--n", type=int, default=300, help="how many chats to add")
    p.add_argument("--out", default=str(DATA / "chats.jsonl"))
    p.add_argument("--teacher", choices=["local", "deepseek"], default="local")
    p.add_argument("--server", help="address of the teacher (default depends on --teacher)")
    p.add_argument("--model", help="model name for a hosted API (deepseek: deepseek-flash or deepseek-v4-pro)")
    p.add_argument("--key-env", help="name of the environment variable holding the API key")
    p.add_argument("--tries", type=int, default=3, help="write + repair passes before giving up on a line")
    p.add_argument("--workers", type=int, help="chats written at the same time")
    p.add_argument("--grammar", action="store_true", help="local llama.cpp only: force letters through a grammar")
    p.add_argument("--think", action="store_true", help="deepseek: let the model think first (slower, costs more)")
    p.add_argument("--sample", type=int, default=15, help="how many chats to print at the end")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="show the prompt, call nothing")
    a = p.parse_args()

    tok = Tokenizer()
    words = tok.vocab["speak"]
    missing = [w for w in COMMON if w not in tok.speak_words]
    system = FRAME.format(words=" ".join(words), missing=", ".join(missing))
    if a.dry_run:
        print(system[:1400] + " ...\n")
        print(f"{len(words):,} allowed words; {len(missing)} everyday words flagged as missing")
        return

    if a.teacher == "deepseek":
        key = os.environ.get(a.key_env or "DEEPSEEK_KEY")
        if not key:
            p.error("No API key found. Run:  read -s DEEPSEEK_KEY && export DEEPSEEK_KEY")
        teacher = Teacher(a.server or "https://api.deepseek.com", "/chat/completions",
                          a.model or "deepseek-flash", key,
                          {} if a.think else {"thinking": {"type": "disabled"}}, local=False)
        workers = a.workers or 8
    else:
        key = os.environ.get(a.key_env) if a.key_env else None
        teacher = Teacher(a.server or "http://127.0.0.1:8080", model=a.model, key=key,
                          local=not a.model)
        workers = a.workers or 1
    grammar = build_grammar(words) if a.grammar else None

    out, rng = Path(a.out), random.Random(a.seed)
    seen: dict[str, int] = {}
    firsts: dict[str, int] = {}
    written: list[dict] = []
    made = rejected = repairs = replies = 0
    start = time.time()
    stop = False
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        while made < a.n and not stop:
            batch = [pool.submit(make_chat, teacher, tok, system, random.Random(rng.random()),
                                 a.tries, grammar) for _ in range(min(workers * 2, a.n - made))]
            for job in as_completed(batch):
                chat, bad, fixed = job.result()
                rejected += bad
                repairs += fixed
                kept = []
                for t in chat["turns"]:
                    first = tok.split(t["reply"])[0]
                    # Across the whole set: no line said more than twice, and no single
                    # opening word allowed to take over.
                    overused = replies >= 40 and firsts.get(first, 0) > 0.30 * replies
                    if seen.get(t["reply"], 0) >= 2 or overused:
                        rejected += 1
                        break
                    kept.append(t)
                    seen[t["reply"]] = seen.get(t["reply"], 0) + 1
                    firsts[first] = firsts.get(first, 0) + 1
                    replies += 1
                if kept and made < a.n:
                    chat["turns"] = kept
                    f.write(json.dumps(chat, ensure_ascii=False) + "\n")
                    f.flush()
                    written.append(chat)
                    made += 1
                    if made % 10 == 0:
                        rate = (time.time() - start) / made
                        print(f"{made}/{a.n} chats  ({rejected} rejected)  "
                              f"about {rate * (a.n - made) / 60:.0f} min left", flush=True)
            if rejected > max(60, 10 * a.n) and rejected > 20 * max(made, 1):
                print(f"Stopping early: {rejected} exchanges rejected against {made} chats kept. "
                      "The teacher is not managing to stay inside her word list.")
                stop = True
    print(f"done: {made} chats added to {out.name}")
    tried = replies + rejected
    if tried:
        print(f"  {replies} replies kept of {tried} attempted ({replies / tried:.0%}); "
              f"{repairs} repair passes")
    if teacher.tokens_in:
        print(f"  teacher usage: {teacher.calls:,} calls, {teacher.tokens_in:,} tokens in "
              f"({teacher.tokens_cached:,} cached), {teacher.tokens_out:,} tokens out")
    if written:
        print("\n== a random sample to read")
        for chat in random.sample(written, min(a.sample, len(written))):
            print(f"--- {chat['topic']}")
            for t in chat["turns"]:
                print(f"  person: {t['user']}\n  excelsia [{t['mood']}]: {t['reply']}")


if __name__ == "__main__":
    main()
