"""Checks that matter: she can never say a word outside the constitution.

    python -m tests.test_excelsia        (or: pytest)
"""
import json
import random

import torch

from excelsia.gen_chats import build_grammar, usable, word_rules, write_reply
from excelsia.model import GPT, Config
from excelsia.tokenizer import DATA, Tokenizer
from excelsia.train import chat_batcher, constitution_ids, encode_chat, lm_batcher
from excelsia.words import count_words

tok = Tokenizer()


def test_word_list_matches_the_text():
    text = (DATA / "ny_constitution.txt").read_text(encoding="utf-8")
    assert set(count_words(text)) == tok.speak_words


def test_reading_and_speaking_words_do_not_overlap():
    assert not tok.speak_words & set(tok.vocab["read_only"])
    assert len(tok) < 65536  # token ids are stored as 16-bit numbers


def test_what_she_can_and_cannot_say():
    assert tok.can_say("I do solemnly affirm.")[0]
    assert tok.can_say("The forest of New York is forever wild!")[0]
    for word in ["you", "yes", "hello", "love", "happy", "sad", "me", "am", "excelsia"]:
        assert not tok.can_say(word)[0], word


def test_she_knows_her_name_but_cannot_say_it():
    assert tok.encode("excelsia") != [tok.unk]
    assert not tok.speak_mask[tok.stoi["excelsia"]]
    assert tok.can_say("I am New York.") == (False, ["am"])
    assert tok.can_say("I, New York.")[0]


def test_round_trip():
    ids = tok.encode("i shall be grateful forever . the pain is cruel and unusual !")
    assert tok.unk not in ids
    assert tok.decode(ids) == "I shall be grateful forever. The pain is cruel and unusual!"


def test_unknown_words_are_read_as_unknown():
    assert tok.encode("zzyzx") == [tok.unk]


def test_generation_cannot_leave_the_word_list():
    """Even a model with random weights, sampled hot, stays inside the list."""
    torch.manual_seed(1)
    model = GPT(Config(len(tok), block_size=64, n_layer=2, n_head=2, n_embd=64))
    allowed = torch.tensor(tok.speak_mask)
    total = 0
    for _ in range(40):
        start = torch.tensor([[tok.user] + tok.encode("hello, do you love me?") + [tok.excelsia]])
        out = model.generate(start, allowed, tok.end, max_new=80, temperature=2.0, top_k=0)
        assert all(tok.speak_mask[i] for i in out)
        ok, bad = tok.can_say(tok.decode(out))
        assert ok, bad
        total += len(out)
    assert total > 1000


def test_chat_loss_is_only_on_her_words():
    chat = {"turns": [{"user": "hello you", "mood": "peace", "reply": "peace ."}]}
    x, y = encode_chat(tok, chat, 64)
    assert len(x) == len(y)
    wanted = [t for t in y if t != -100]
    assert wanted == tok.encode("peace .") + [tok.end]
    xb, yb = chat_batcher([(x, y)] * 4, 4, tok.pad, "cpu")()
    assert xb.shape == yb.shape


def test_training_makes_progress():
    torch.manual_seed(0)
    random.seed(0)
    law = constitution_ids(tok)
    model = GPT(Config(len(tok), block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0))
    get = lm_batcher([(law, 1.0)], 32, 16, tok.unk, "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for _ in range(60):
        _, loss = model(*get())
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first - 1.0, (first, last)


def test_bad_practice_chats_are_rejected():
    ok = lambda user, reply, turns=(), seen=None: usable(tok, user, reply, list(turns), seen or {})
    assert ok("hello there", "peace, the people are together.")
    assert not ok("hello there", "hello, i love you.")                # forbidden words
    assert not ok("hello there", "peace and the people are")          # cut off mid-sentence
    assert not ok("hello there", "excise excise excise excise.")      # stuck in a loop
    assert not ok("", "peace.")
    before = [{"user": "hi", "mood": "peace", "reply": "peace."}]
    assert not ok("Peace.", "the forest is wild.", before)            # teacher parroted her
    assert not ok("and now?", "peace.", before)                       # same reply twice in a chat
    assert not ok("HI", "the forest is wild.", before)                # person repeated themselves
    assert not ok("hello", "peace.", (), {"peace.": 2})               # overused across chats
    assert not ok("I shall be leaving now.", "i shall be leaving now.")  # echoes the person
    assert not ok("hello", "the forest is wild. the forest is wild.")  # same sentence twice
    assert not ok("more?", "the forest is wild. peace shall remain.",
                  [{"user": "hi", "mood": "peace", "reply": "peace shall remain."}])  # reused sentence


class FakeTeacher:
    """Stands in for a model: hands back prepared lines and records what it was asked."""

    def __init__(self, lines):
        self.lines, self.asked = list(lines), []

    def ask(self, messages, temperature, max_tokens, grammar=None):
        self.asked.append(messages)
        return self.lines.pop(0)


def test_a_bad_word_is_sent_back_for_repair_by_meaning():
    teacher = FakeTeacher(['"I am happy, thank you."', "i shall be grateful."])
    reply, tries = write_reply(teacher, tok, "system", "task", tries=3)
    assert (reply, tries) == ("i shall be grateful.", 2)
    repair = teacher.asked[1][-1]["content"]
    assert "am" in repair and "happy" in repair and "MEANING" in repair
    assert teacher.asked[1][-2] == {"role": "assistant", "content": "i am happy, thank you."}


def test_an_overused_word_is_sent_back_too():
    teacher = FakeTeacher(["the people remain.", "the waters remain."])
    reply, tries = write_reply(teacher, tok, "system", "task", tries=3, tired=("people",))
    assert (reply, tries) == ("the waters remain.", 2)
    assert "people" in teacher.asked[1][-1]["content"]


def test_the_tally_notices_what_she_leans_on():
    from excelsia.gen_chats import Tally
    tally = Tally(cap=0.15, after=10)
    for i in range(10):
        tally.add(tok.split("the people remain." if i < 6 else "the waters are near."))
    assert set(tally.tired[:2]) == {"people", "remain"}   # each in 6 of 10 replies
    assert "the" not in tally.tired        # little words never count
    assert "near" in tally.tired           # 4 of 10 is over the cap as well


def test_she_can_tire_of_her_own_words():
    model = GPT(Config(vocab_size=len(tok), block_size=32, n_layer=1, n_head=2, n_embd=32))
    allowed = torch.tensor(tok.speak_mask)
    people = tok.stoi["people"]
    torch.manual_seed(0)
    out = model.generate(torch.tensor([[tok.user, tok.excelsia]]), allowed, tok.end, 24,
                         recent=[people] * 3, repeat=50.0, exempt=set())
    assert people not in out and len(set(out)) == len(out)   # nothing said twice
    assert all(tok.speak_mask[t] for t in out)               # and the word rule still holds


def test_a_line_that_never_passes_is_dropped():
    teacher = FakeTeacher(["hello you."] * 3)
    assert write_reply(teacher, tok, "system", "task", tries=3) == ("", 3)


def test_grammar_spells_exactly_her_words():
    """Walk the grammar's prefix tree and check it spells her word list and nothing else."""
    rules, start = word_rules(tok.vocab["speak"])
    spelled = set()

    def walk(name, prefix):
        branches, may_stop = rules[name]
        if may_stop:
            spelled.add(prefix)
        for letters, nxt in branches:
            if nxt:
                walk(nxt, prefix + letters)
            else:
                spelled.add(prefix + letters)

    walk(start, "")
    assert spelled == tok.speak_words
    grammar = build_grammar(tok.vocab["speak"])
    assert grammar.startswith("root ::=") and "word ::= " + start in grammar

def test_the_lightweight_version_matches_the_real_one():
    """The Discord bot runs her in plain numpy. It must give the same scores as PyTorch."""
    import tempfile
    from pathlib import Path
    import numpy as np
    from excelsia.export import export
    from excelsia.lite import LiteGPT
    from excelsia.train import save
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=len(tok), block_size=32, n_layer=2, n_head=2, n_embd=32)).eval()
    with torch.no_grad():
        for p in model.parameters():      # a fresh model is nearly all zeros where it
            p.add_(torch.randn_like(p) * 0.2)  # matters; shake it so the test means something
    ids = [tok.user] + tok.encode("hello there, who are you?") + [tok.mood("peace"), tok.excelsia]
    with torch.no_grad():
        want = model(torch.tensor([ids]))[0][0, -1].numpy()
    with tempfile.TemporaryDirectory() as d:
        save(model, Path(d) / "m.pt", 0, "finetune")
        export(Path(d) / "m.pt", Path(d) / "m.npz")
        lite = LiteGPT(Path(d) / "m.npz")
    cache = [None] * lite.layers
    whole = lite._scores(ids, 0, cache)
    cache = [None] * lite.layers
    for i, t in enumerate(ids):           # and again one word at a time, through the cache
        stepped = lite._scores([t], i, cache)
    assert np.abs(whole - want).max() < 0.05 and np.abs(stepped - want).max() < 0.05
    assert int(whole.argmax()) == int(want.argmax())
    out = lite.generate(ids, np.array(tok.speak_mask), tok.end, 12, rng=np.random.default_rng(0))
    assert out and all(tok.speak_mask[t] for t in out)   # the word rule holds here too
    stuck = lambda o: any(o[i] == o[i + 1] == o[i + 2] for i in range(len(o) - 2))
    greedy = lite.generate(ids, np.array(tok.speak_mask), tok.end, 16, top_k=1)
    also = model.generate(torch.tensor([ids]), torch.tensor(tok.speak_mask), tok.end, 16, top_k=1)
    assert not stuck(greedy) and not stuck(also)         # never one word three times running


def test_the_bot_tidies_what_people_type():
    from bot.talk import LONGEST, tidy
    assert tidy("@Excelsia   hello\nthere", ("Excelsia",)) == "hello there"
    assert tidy("look https://example.com/x <:wave:123> hi") == "look hi"
    assert tidy("@Excelsia", ("Excelsia",)) == ""
    assert len(tidy("word " * 500)) <= LONGEST


def test_the_bot_makes_people_wait_their_turn():
    from bot.talk import PAUSE, Room
    room = Room(model=None, tok=tok)
    assert not room.too_soon(1, now=100.0)
    assert room.too_soon(1, now=100.0 + PAUSE / 2)       # same person, too quick
    assert not room.too_soon(2, now=100.0 + PAUSE / 2)   # someone else is fine
    assert not room.too_soon(1, now=100.0 + PAUSE * 2)


def test_the_bot_keeps_threads_apart_but_shares_her_mood():
    from bot.talk import Room

    class Canned:
        block, seen = 256, []
        def generate(self, ids, allowed, stop, max_new=48, temperature=0.8, not_first=None, rng=None):
            Canned.seen.append(list(ids))
            return tok.encode("the waters remain.")

    room = Room(model=Canned(), tok=tok, remember=2)   # with a memory, to test the threads
    forest, city = tok.stoi["forest"], tok.stoi["city"]
    assert room.reply(7, 1, "you are wonderful, thank you", now=0.0) == ("pleasure", "The waters remain.")
    room.reply(7, 1, "tell me about the forest", now=1.0)
    mood, _ = room.reply(7, 2, "what about the city", now=2.0)
    assert mood != "peace"                          # person 2 meets the mood person 1 left her in
    assert forest not in Canned.seen[-1]            # but not person 1's conversation
    room.reply(7, 1, "and then?", now=3.0)
    assert forest in Canned.seen[-1] and city not in Canned.seen[-1]   # person 1's thread carries on
    for i in range(5):
        room.reply(7, 1, "more", now=4.0 + i)
    assert Canned.seen[-1].count(tok.user) == 2 + 1                    # only the last few exchanges
    fresh = Room(model=Canned(), tok=tok)                              # by default she keeps none
    fresh.reply(7, 1, "tell me about the forest", now=0.0)
    fresh.reply(7, 1, "and then?", now=1.0)
    assert forest not in Canned.seen[-1] and Canned.seen[-1].count(tok.user) == 1
    assert room.reply(8, 1, "hello", now=20.0)[0] == "peace"           # another channel is untouched
    # clearing her memory: one person's thread, then the whole channel
    from bot.talk import NOT_HERE, RESET_REPLY
    assert tok.can_say(RESET_REPLY)[0] and tok.can_say(NOT_HERE)[0]   # said in her own words
    room.reply(7, 2, "what about the city", now=21.0)
    room.forget(7, 1)
    room.reply(7, 1, "hello again", now=22.0)
    assert forest not in Canned.seen[-1] and Canned.seen[-1].count(tok.user) == 1
    assert (7, 2) in room.talks and 7 in room.moods     # person 2 and her mood were left alone
    room.forget(7)
    assert not any(k[0] == 7 for k in room.talks) and 7 not in room.moods and (8, 1) in room.talks



def test_chats_written_elsewhere_are_checked_line_by_line():
    from excelsia.check_chats import check
    lines = [json.dumps(c) for c in [
        {"topic": "fine", "turns": [{"user": "hi", "mood": "peace", "reply": "The waters remain."},
                                    {"user": "why?", "mood": "peace", "reply": "the people keep them."}]},
        {"topic": "second line has a word she lacks", "turns": [
            {"user": "hi", "mood": "peace", "reply": "the waters remain."},
            {"user": "ok", "mood": "peace", "reply": "you are kind."},
            {"user": "bye", "mood": "alone", "reply": "the people go."}]},
        {"topic": "bad mood", "turns": [{"user": "hi", "mood": "sleepy", "reply": "the waters remain."}]},
        {"topic": "repeats herself", "turns": [
            {"user": "hi", "mood": "peace", "reply": "the waters remain."},
            {"user": "and?", "mood": "peace", "reply": "the waters remain."}]},
    ]] + ["this line is not json"]
    kept, report = check(tok, lines)
    assert [len(c["turns"]) for c in kept] == [2, 1, 1]       # cut at the first bad line, never rewritten
    assert kept[0]["turns"][0]["reply"] == "the waters remain."   # tidied to lower case
    assert (report["whole"], report["shortened"], report["dropped"], report["unreadable"]) == (1, 2, 1, 1)
    assert report["words"]["you"] == 1
    assert all(tok.can_say(t["reply"])[0] for c in kept for t in c["turns"])
    kept, report = check(tok, lines, least=2)                 # ask for real back-and-forth only
    assert [len(c["turns"]) for c in kept] == [2]


def test_talk_chats_are_longer_and_survive_one_bad_line():
    from excelsia.gen_chats import MOVES, make_talk

    class Scripted:
        """A stand-in teacher: the person says something new each time; her second reply
        is unusable once, so the writer has to try that turn again."""
        def __init__(self):
            self.person, self.her, self.tasks = 0, 0, []
            self.lines = ["the waters remain.", "you are nice.",
                          "my trees are near.", "a county is a place.", "the lakes are still.",
                          "i keep the forest.", "all is well."]
        def ask(self, messages, temperature, max_tokens, grammar=None):
            task = messages[-1]["content"]
            if task.startswith("Write the next message from the person"):
                self.person += 1
                self.tasks.append(task)
                return f"message number {self.person}"
            if "cannot use here" in task:
                return "you are nice."          # the repair fails too, so the whole try fails
            self.her += 1
            return self.lines[min(self.her - 1, len(self.lines) - 1)]

    teacher = Scripted()
    chat, rejected, repairs = make_talk(teacher, tok, "system", random.Random(3), tries=2, grammar=None)
    assert 3 <= len(chat["turns"]) <= 5 and rejected == 1          # one bad line, and the chat went on
    assert all(tok.can_say(t["reply"])[0] for t in chat["turns"])
    assert "Situation: the person is" in teacher.tasks[0]          # the first message sets the scene
    assert all("This time the person should" in t for t in teacher.tasks[1:])   # every later one is a move
    assert any(any(m.split("{")[0][:30] in t for m in MOVES.values()) for t in teacher.tasks[1:])

    class Hopeless(Scripted):
        def ask(self, messages, temperature, max_tokens, grammar=None):
            task = messages[-1]["content"]
            return "hello there" if task.startswith("Write the next message from the person") else "you are nice."
    chat, rejected, _ = make_talk(Hopeless(), tok, "system", random.Random(3), tries=2, grammar=None)
    assert chat["turns"] == [] and rejected == 2                   # nothing usable: nothing kept

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
