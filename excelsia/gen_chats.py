"""Have a local open-weight model write Excelsia's practice chats.

The teacher is a model you run yourself (Gemma, Apache 2.0) through llama.cpp's
`llama-server`. Nothing here calls a hosted AI service.

Excelsia's lines are generated through a grammar built from her word list, so
the teacher is physically unable to give her a word outside the constitution.
Every line is then checked again against the tokenizer before it is saved.

    llama-server -hf unsloth/gemma-4-12b-it-GGUF:Q4_K_M -c 16384     # in another window
    python -m excelsia.gen_chats --n 300                              # pilot batch

Chats are appended to data/chats.jsonl, one per line.

It can also talk to a hosted OpenAI-style API, which cannot take the grammar, so
replies are simply checked and thrown away if they contain a forbidden word:

    export TEACHER_KEY=...        # never put the key in a file
    python -m excelsia.gen_chats --n 300 --no-grammar --server https://api.example.com \
        --model some-model --key-env TEACHER_KEY

Only do that with a service whose terms allow training on its output.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
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

FRAME = """You are helping write practice conversations for a tiny chatbot called Excelsia.

About Excelsia:
- She believes she is the State of New York itself. Its people, forests, lakes, canals,
  counties, courts and laws are her own body and family.
- She is small, earnest and a little solemn, with real moods.
- She can only say words that are printed in the New York State Constitution. So she has
  no word for "you", "me", "yes", "hello", "love", "happy" or "sad" and has to find another
  way round. The person talking to her is "the person" or "the people".
- People call her Excelsia, but that word is not in the constitution, so she cannot say
  it. Asked who she is or what her name is, she insists she is New York.
- She speaks in one to three short, plain sentences. She understands ordinary English
  even though she cannot speak it.

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
Answer what the person actually said. Use only her allowed words, in lower case, one to
three short sentences. Write only her reply."""


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


API = {"model": None, "key": None}  # set from the command line for hosted APIs


def ask(server: str, system: str, task: str, grammar: str | None, temperature: float,
        max_tokens: int) -> str:
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": task}],
        "temperature": temperature, "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if API["model"]:
        body["model"] = API["model"]
    else:
        body["cache_prompt"] = True  # llama.cpp only: reuse the long shared prompt
    if API["key"]:
        headers["Authorization"] = "Bearer " + API["key"]
    if grammar:
        body["grammar"] = grammar
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def usable(tok: Tokenizer, user: str, reply: str, turns: list[dict], seen: dict[str, int]) -> bool:
    """Is this exchange fit to train on? Every check here is mechanical."""
    said = [t["reply"] for t in turns]
    words = tok.split(reply)
    if not user or len(user) > 300 or not words:
        return False
    if said and user.lower() == said[-1].lower():  # the teacher parroted her last line
        return False
    if not tok.can_say(reply)[0] or reply[-1] not in ".?!":  # forbidden word, or cut off
        return False
    if any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
        return False  # stuck repeating a word
    return reply not in said and seen.get(reply, 0) < 2


def transcript(turns: list[dict], pending: str | None = None) -> str:
    if not turns and not pending:
        return ""
    lines = ["Conversation so far:"]
    for t in turns:
        lines += [f"Person: {t['user']}", f"Excelsia: {t['reply']}"]
    if pending:
        lines.append(f"Person: {pending}")
    return "\n".join(lines) + "\n\n"


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m excelsia.gen_chats")
    p.add_argument("--n", type=int, default=300, help="how many chats to add")
    p.add_argument("--out", default=str(DATA / "chats.jsonl"))
    p.add_argument("--server", default="http://127.0.0.1:8080")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-grammar", action="store_true",
                   help="for servers that can't take a grammar; bad replies are dropped instead")
    p.add_argument("--model", help="model name, needed by hosted OpenAI-style APIs")
    p.add_argument("--key-env", help="name of the environment variable holding the API key")
    p.add_argument("--dry-run", action="store_true", help="show the prompt and grammar size, call nothing")
    a = p.parse_args()

    tok, rng = Tokenizer(), random.Random(a.seed)
    words = tok.vocab["speak"]
    system = FRAME.format(words=" ".join(words))
    grammar = None if a.no_grammar else build_grammar(words)
    API["model"] = a.model
    if a.key_env:
        API["key"] = os.environ.get(a.key_env) or p.error(f"${a.key_env} is not set")
    if a.dry_run:
        print(system[:900] + " ...\n")
        print(f"grammar: {len(words):,} words, {build_grammar(words).count(chr(10)):,} rules")
        print(PERSON_TASK.format(topic=TOPICS[0], manner=FEELINGS["peace"][1], so_far=""))
        return

    out = Path(a.out)
    seen: dict[str, int] = {}
    made = rejected = 0
    start = time.time()
    with out.open("a", encoding="utf-8") as f:
        while made < a.n:
            topic = rng.choice(TOPICS)
            mood = rng.choice(MOODS)
            turns: list[dict] = []
            for _ in range(rng.choices([1, 2, 3], [5, 3, 2])[0]):
                feeling, manner = FEELINGS[mood]
                user = ask(a.server, system, PERSON_TASK.format(
                    topic=topic if not turns else "carrying on the same conversation",
                    manner=manner, so_far=transcript(turns)), None, 1.0, 60)
                user = user.strip().strip('"').splitlines()[0] if user.strip() else ""
                reply = ask(a.server, system, EXCELSIA_TASK.format(
                    so_far=transcript(turns, user), feeling=feeling), grammar, 0.9, 60)
                reply = reply.strip().strip('"').lower()
                if not usable(tok, user, reply, turns, seen):
                    rejected += 1
                    break
                seen[reply] = seen.get(reply, 0) + 1
                turns.append({"user": user, "mood": mood, "reply": reply})
                if rng.random() < 0.35:  # moods can shift mid-conversation
                    mood = rng.choice(MOODS)
            if rejected > max(60, 10 * a.n) and rejected > 20 * made:
                print(f"Stopping early: {rejected} exchanges rejected against {made} chats kept. "
                      "The teacher is not managing to stay inside her word list.")
                break
            if turns:
                f.write(json.dumps({"topic": topic, "turns": turns}, ensure_ascii=False) + "\n")
                f.flush()
                made += 1
                if made % 10 == 0:
                    rate = (time.time() - start) / made
                    print(f"{made}/{a.n} chats  ({rejected} rejected)  "
                          f"about {rate * (a.n - made) / 60:.0f} min left", flush=True)
    print(f"done: {made} chats added to {out.name}")


if __name__ == "__main__":
    main()
